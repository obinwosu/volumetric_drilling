#ifndef HAPTIC_MONITOR_FEED_H
#define HAPTIC_MONITOR_FEED_H

// Feed for the haptic monitor window (scripts/haptic_monitor/haptic_monitor.py)
// while the simulator owns the Touch.
//
// The physics thread only stores a few numbers in atomics (physicsTick). A
// separate thread, ~20 times a second:
//  - reads each Touch CHAI3D opened (handles from libhdPhantom's phantomDevices)
//    with one hdScheduleSynchronous callback: encoders, angles, buttons,
//    temperature, servo rate, calibration, force output, HD errors
//    (haptic_device_probe.h; a few microseconds in the servo thread, no frame,
//    no force set);
//  - writes /tmp/fivrs-rtx/haptic_state.json (20 Hz while the monitor window
//    is open, 2 Hz otherwise);
//  - appends health events to ~/.local/share/fivrs/haptic_health.log, always.
// It never changes a force, a button or the device.

#include "haptic_device_probe.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

class HapticMonitorFeed {
public:
    // libhdPhantom's per-device record (ambf/core/external/chai3d/extras/hdPhantom/
    // hdPhantom.cpp). Only handle and enabled are read; the layout is checked
    // against the 4 x 200 bytes the library reserves for it.
    struct PhantomRecord {
        unsigned int handle;
        double position[3];
        double linearVelocity[3];
        double rotation[9];
        double force[3];
        double torque[3];
        int button;
        bool enabled;
        bool initialized;
        double workspaceRadius;
        void* sync;
    };
    static_assert(sizeof(PhantomRecord) == 200, "libhdPhantom device record layout changed");

    ~HapticMonitorFeed() { stop(); }

    void start()
    {
        if (m_thread.joinable()) return;
        m_stop = false;
        m_thread = std::thread([this] { run(); });
    }

    // Called from close(). Waits at most 0.5 s: a servo loop that has already
    // stopped must not hang AMBF's shutdown.
    void stop()
    {
        if (!m_thread.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(m_wakeMutex);
            m_stop = true;
        }
        m_wake.notify_all();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (!m_exited && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (m_exited) m_thread.join();
        else m_thread.detach();
    }

    // Physics thread, every step. Relaxed atomics only.
    void physicsTick(double dt, const double requested[3], const double sent[3], const double suction[3],
                     bool paused, bool contactSuppressed, bool inContact, bool drillOn,
                     bool safetyTripped, int safetyTrips, bool suctionTracked)
    {
        for (int i = 0; i < 3; i++) {
            m_requested[i].store(requested[i], std::memory_order_relaxed);
            m_sent[i].store(sent[i], std::memory_order_relaxed);
            m_suction[i].store(suction[i], std::memory_order_relaxed);
        }
        m_paused.store(paused, std::memory_order_relaxed);
        m_suppressed.store(contactSuppressed, std::memory_order_relaxed);
        m_inContact.store(inContact, std::memory_order_relaxed);
        m_drillOn.store(drillOn, std::memory_order_relaxed);
        m_safetyTripped.store(safetyTripped, std::memory_order_relaxed);
        m_safetyTrips.store(safetyTrips, std::memory_order_relaxed);
        m_suctionTracked.store(suctionTracked, std::memory_order_relaxed);
        m_ticks.fetch_add(1, std::memory_order_relaxed);
        // Longest wall-clock gap between physics steps since the feed last looked:
        // while the physics thread stalls the servo loop keeps sending the last force.
        const int64_t now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        const int64_t last = m_lastTickUs.exchange(now, std::memory_order_relaxed);
        if (last > 0) {
            const int64_t gap = now - last;
            int64_t seen = m_maxGapUs.load(std::memory_order_relaxed);
            while (gap > seen && !m_maxGapUs.compare_exchange_weak(seen, gap, std::memory_order_relaxed)) {}
        }
        (void)dt;
    }

private:
    static constexpr double kPhysicsStallMs = 20.0;
    static constexpr double kPhysicsStoppedMs = 250.0;
    static constexpr double kDropRequestN = 0.5;

    struct Device {
        unsigned int handle = 0;
        std::string role;
        haptic_probe::Info info;
        haptic_probe::Sample sample;
        haptic_probe::HealthTracker tracker{nullptr, ""};
    };

    static haptic_probe::HDCallbackCode readCallback(void* user)
    {
        auto* self = static_cast<HapticMonitorFeed*>(user);
        for (Device& d : self->m_devices) {
            haptic_probe::read(self->m_hd, d.handle, d.sample);
        }
        return haptic_probe::HD_CALLBACK_DONE;
    }

    // Find the devices CHAI3D opened. False while libhdPhantom has not started the servo loop.
    bool attach()
    {
        if (!m_hd.ok) {
            if (!m_hd.load(RTLD_DEFAULT)) return false;
        }
        m_phantoms = reinterpret_cast<PhantomRecord*>(dlsym(RTLD_DEFAULT, "phantomDevices"));
        m_numPhantoms = reinterpret_cast<int*>(dlsym(RTLD_DEFAULT, "numPhantomDevices"));
        m_servoStarted = reinterpret_cast<bool*>(dlsym(RTLD_DEFAULT, "servoStarted"));
        if (!m_phantoms || !m_numPhantoms || !m_servoStarted || !*m_servoStarted) return false;
        const int count = *m_numPhantoms;
        if (count < 1 || count > 4) return false;
        m_devices.clear();
        m_devices.reserve(count);
        static const char* roles[4] = {"drill", "suction", "device 3", "device 4"};
        for (int i = 0; i < count; i++) {
            if (!m_phantoms[i].enabled) continue;
            Device d;
            d.handle = m_phantoms[i].handle;
            d.role = roles[i];
            m_devices.push_back(d);
        }
        if (m_devices.empty()) return false;
        // Identification strings are read in the servo thread too (one more sync call).
        m_hd.scheduleSynchronous(&HapticMonitorFeed::infoCallback, this, haptic_probe::HD_MIN_SCHEDULER_PRIORITY);
        for (Device& d : m_devices) {
            d.tracker = haptic_probe::HealthTracker(&m_log, d.role + " (SN " + d.info.serial + ")");
            d.tracker.event("simulator started reading it. Model " + d.info.model + ", driver " + d.info.driver);
        }
        return true;
    }

    static haptic_probe::HDCallbackCode infoCallback(void* user)
    {
        auto* self = static_cast<HapticMonitorFeed*>(user);
        for (Device& d : self->m_devices) {
            d.info = haptic_probe::readInfo(self->m_hd, d.handle);
        }
        return haptic_probe::HD_CALLBACK_DONE;
    }

    static bool monitorOpen()
    {
        struct stat st;
        if (stat("/tmp/fivrs-rtx/haptic_monitor.alive", &st) != 0) return false;
        return haptic_probe::wallSeconds() - st.st_mtime < 3.0;
    }

    void vec(std::ostringstream& o, const char* key, const std::atomic<double>* v)
    {
        o << ",\"" << key << "\":[" << v[0].load(std::memory_order_relaxed) << ","
          << v[1].load(std::memory_order_relaxed) << "," << v[2].load(std::memory_order_relaxed) << "]";
    }

    static double norm(const std::atomic<double>* v)
    {
        const double x = v[0].load(std::memory_order_relaxed);
        const double y = v[1].load(std::memory_order_relaxed);
        const double z = v[2].load(std::memory_order_relaxed);
        return std::sqrt(x * x + y * y + z * z);
    }

    void run()
    {
        using namespace std::chrono;
        auto lastWrite = steady_clock::now() - seconds(10);
        auto lastAttachTry = steady_clock::now() - seconds(10);
        auto lastRateTime = steady_clock::now();
        uint64_t lastTicks = m_ticks.load();
        double physicsHz = 0.0;
        int lastTrips = 0;
        double dropSince = -1.0;
        bool dropLogged = false;
        double worstGapMinute = 0.0;
        int stallCount = 0;
        double stallWorst = 0.0;
        double lastStallLog = 0.0;
        bool physicsStopped = false;
        double physicsStoppedAt = 0.0;
        std::string attachNote = "waiting for CHAI3D to open the Touch";
        m_log.line("simulator", "haptic feed started (health log " + m_log.path() + ")");

        while (true) {
            {
                std::unique_lock<std::mutex> lock(m_wakeMutex);
                m_wake.wait_for(lock, milliseconds(50), [this] { return m_stop.load(); });
                if (m_stop) break;
            }
            const auto now = steady_clock::now();
            const double wall = haptic_probe::wallSeconds();

            if (!m_attached && now - lastAttachTry > seconds(2)) {
                lastAttachTry = now;
                m_attached = attach();
                if (!m_attached && !m_hd.ok) {
                    attachNote = "OpenHaptics is not loaded in AMBF (no Touch, or a simulated device): forces only";
                }
            }

            // Physics loop rate and stalls.
            const double span = duration<double>(now - lastRateTime).count();
            if (span >= 0.5) {
                const uint64_t ticks = m_ticks.load(std::memory_order_relaxed);
                physicsHz = (ticks - lastTicks) / span;
                lastTicks = ticks;
                lastRateTime = now;
            }
            const double gapMs = m_maxGapUs.exchange(0, std::memory_order_relaxed) / 1000.0;
            m_lastGapMs = gapMs;
            worstGapMinute = std::max(worstGapMinute, gapMs);
            if (gapMs > kPhysicsStallMs) {
                stallCount += 1;
                stallWorst = std::max(stallWorst, gapMs);
            }
            if (stallCount > 0 && wall - lastStallLog >= 1.0) {
                std::ostringstream o;
                o.precision(1);
                o << std::fixed << "physics loop stalled " << stallCount << "x, longest " << stallWorst
                  << " ms (the Touch kept the last force; contact forces were not updated)";
                m_log.line("simulator", o.str());
                lastStallLog = wall;
                stallCount = 0;
                stallWorst = 0.0;
            }
            const int trips = m_safetyTrips.load(std::memory_order_relaxed);
            if (trips != lastTrips) {
                if (trips > lastTrips) {
                    m_log.line("drill", "haptic safety tripped (trip " + std::to_string(trips)
                               + "): unstable contact, drill force faded to zero until out of contact");
                }
                lastTrips = trips;
            }
            // Contact asks for force but none is sent (not paused, not a safety trip).
            const double requested = norm(m_requested);
            const double sent = norm(m_sent);
            const bool paused = m_paused.load(std::memory_order_relaxed);
            const bool tripped = m_safetyTripped.load(std::memory_order_relaxed);
            if (!paused && requested >= kDropRequestN && sent < 0.2 * requested) {
                if (dropSince < 0.0) dropSince = wall;
                if (!dropLogged && wall - dropSince > 0.1) {
                    dropLogged = true;
                    std::ostringstream o;
                    o.precision(2);
                    o << std::fixed << "force dropped: contact asks " << requested << " N, " << sent << " N sent"
                      << (tripped ? " (haptic safety)" : "");
                    m_log.line("drill", o.str());
                }
            } else {
                if (dropLogged) {
                    std::ostringstream o;
                    o.precision(2);
                    o << std::fixed << "force back after " << (wall - dropSince) << " s";
                    m_log.line("drill", o.str());
                }
                dropSince = -1.0;
                dropLogged = false;
            }
            // A physics loop that stops altogether never closes a gap: watch the time since its last step.
            const int64_t lastTick = m_lastTickUs.load(std::memory_order_relaxed);
            if (lastTick > 0) {
                const int64_t nowUs = duration_cast<microseconds>(now.time_since_epoch()).count();
                const double idleMs = (nowUs - lastTick) / 1000.0;
                if (idleMs > kPhysicsStoppedMs && !physicsStopped) {
                    physicsStopped = true;
                    physicsStoppedAt = wall - idleMs / 1000.0;
                    m_log.line("simulator", "physics loop STOPPED: no step for " + std::to_string(static_cast<int>(idleMs))
                               + " ms (the Touch keeps the last force)");
                } else if (idleMs < kPhysicsStallMs && physicsStopped) {
                    physicsStopped = false;
                    std::ostringstream o;
                    o.precision(2);
                    o << std::fixed << "physics loop running again after " << (wall - physicsStoppedAt) << " s";
                    m_log.line("simulator", o.str());
                }
                worstGapMinute = std::max(worstGapMinute, idleMs);
            }
            bool summaryDue = false;
            for (Device& d : m_devices) summaryDue = summaryDue || d.tracker.summaryDue(wall);
            if (summaryDue) {
                std::ostringstream o;
                o.precision(1);
                o << std::fixed << "physics " << physicsHz << " Hz, longest physics gap " << worstGapMinute
                  << " ms, safety trips " << trips;
                for (Device& d : m_devices) d.tracker.setSummaryExtra(o.str());
                worstGapMinute = 0.0;
            }

            if (m_attached) {
                if (!*m_servoStarted) {
                    m_attached = false;
                    m_log.line("simulator", "servo loop stopped");
                } else {
                    m_hd.scheduleSynchronous(&HapticMonitorFeed::readCallback, this, haptic_probe::HD_MIN_SCHEDULER_PRIORITY);
                    for (Device& d : m_devices) d.tracker.update(m_hd, d.sample, wall);
                }
            }

            const auto writeEvery = monitorOpen() ? milliseconds(45) : milliseconds(500);
            if (now - lastWrite < writeEvery) continue;
            lastWrite = now;
            std::ostringstream o;
            o.setf(std::ios::fixed);
            o.precision(4);
            o << "{\"source\":\"sim\",\"wall_time\":" << wall
              << ",\"status\":\"" << (m_attached ? "ok" : "no_device") << "\""
              << ",\"error\":" << haptic_probe::jsonString(m_attached ? "" : attachNote)
              << ",\"health_log\":" << haptic_probe::jsonString(m_log.path())
              << ",\"physics_hz\":" << physicsHz
              << ",\"physics_gap_ms\":" << m_lastGapMs
              << ",\"paused\":" << (paused ? "true" : "false")
              << ",\"contact_suppressed\":" << (m_suppressed.load() ? "true" : "false")
              << ",\"drill_in_contact\":" << (m_inContact.load() ? "true" : "false")
              << ",\"drill_on\":" << (m_drillOn.load() ? "true" : "false")
              << ",\"safety_tripped\":" << (tripped ? "true" : "false")
              << ",\"safety_trips\":" << trips
              << ",\"suction_tracked\":" << (m_suctionTracked.load() ? "true" : "false");
            vec(o, "drill_requested_n", m_requested);
            vec(o, "drill_sent_n", m_sent);
            vec(o, "suction_sent_n", m_suction);
            o << ",\"devices\":[";
            bool first = true;
            if (m_attached) {
                for (Device& d : m_devices) {
                    o << (first ? "" : ",") << haptic_probe::deviceJson(d.role, d.info, d.sample);
                    first = false;
                }
            }
            o << "]}\n";
            haptic_probe::writeAtomically("/tmp/fivrs-rtx/haptic_state.json", o.str());
        }
        m_log.line("simulator", "haptic feed stopped");
        m_exited = true;
    }

    haptic_probe::Api m_hd;
    haptic_probe::HealthLog m_log{"sim"};
    std::vector<Device> m_devices;
    PhantomRecord* m_phantoms = nullptr;
    int* m_numPhantoms = nullptr;
    bool* m_servoStarted = nullptr;
    bool m_attached = false;
    double m_lastGapMs = 0.0;

    std::thread m_thread;
    std::mutex m_wakeMutex;
    std::condition_variable m_wake;
    std::atomic<bool> m_stop{false};
    std::atomic<bool> m_exited{false};

    std::atomic<double> m_requested[3] = {{0.0}, {0.0}, {0.0}};
    std::atomic<double> m_sent[3] = {{0.0}, {0.0}, {0.0}};
    std::atomic<double> m_suction[3] = {{0.0}, {0.0}, {0.0}};
    std::atomic<bool> m_paused{false}, m_suppressed{false}, m_inContact{false}, m_drillOn{false};
    std::atomic<bool> m_safetyTripped{false}, m_suctionTracked{false};
    std::atomic<int> m_safetyTrips{0};
    std::atomic<uint64_t> m_ticks{0};
    std::atomic<int64_t> m_lastTickUs{0};
    std::atomic<int64_t> m_maxGapUs{0};
};

#endif
