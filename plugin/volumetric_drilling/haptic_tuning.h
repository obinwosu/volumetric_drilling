#ifndef HAPTIC_TUNING_H
#define HAPTIC_TUNING_H

// Live force settings for the haptics tuner (scripts/haptics_tuner).
//
// Off unless the simulator is started with FIVRS_HAPTIC_TUNER=1: every other launch
// keeps the stock removal, stiffness and hum. The tuner window owns two shared files:
//   /dev/shm/fivrs_haptic_tuning    settings, read every physics tick (seqlock)
//   /dev/shm/fivrs_haptic_telemetry ring of per-tick samples for the plots
// The layout is mirrored in scripts/haptics_tuner/tuning_shm.py.
//
// Every value is clamped here, so a bad file cannot command more than these limits.
// This header has no CHAI3D dependency so it can be tested on its own.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace haptic_tuning {

constexpr uint32_t kParamMagic = 0x4E555448;      // "HTUN"
constexpr uint32_t kTelemetryMagic = 0x4C455448;  // "HTEL"
constexpr uint32_t kVersion = 1;
constexpr int kParamCount = 18;
constexpr int kTelemetryRows = 4096;
constexpr int kTelemetryFields = 8;
constexpr size_t kParamHeaderBytes = 16;
constexpr size_t kTelemetryHeaderBytes = 24;

constexpr double kDeviceForceN = 3.3;       // Touch maximum
constexpr double kMaxHumN = 0.6;            // hard cap on the hum, at any load
constexpr double kMaxHumHz = 400.0;         // physics runs at about 1 kHz
constexpr double kMinHumHz = 20.0;
constexpr double kMaxCutRateMm3 = 400.0;
constexpr int kMaxVoxelsPerTick = 40;
constexpr double kMaxStiffnessMult = 1.85;  // 400 N/m at the hand, the Touch's rated stiffness
constexpr double kArmQuietN = 0.1;          // force starts only from a request this small
constexpr double kArmQuietS = 0.25;
constexpr double kContinuousN = 0.88;       // the Touch's rated continuous force
constexpr double kSustainedTripN = 1.5;     // RMS over about 3 s that starts the limit
constexpr double kSustainedClearN = 1.0;
constexpr double kSustainedTauS = 3.0;

enum HumDirection { HUM_DIAGONAL = 0, HUM_SHAFT = 1, HUM_RADIAL = 2 };

// Order = file order. Defaults reproduce today's haptics.
struct Params {
    double cutGate = 0.0;        // 0 = stock removal, 1 = force decides the cut rate
    double cutStartN = 0.4;      // below this force the burr does not cut
    double cutFullN = 1.6;       // at this force it cuts at the full rate
    double cutCurve = 1.0;       // 1 = linear, > 1 = slow start
    double rateMm3[4] = {4.0, 12.0, 40.0, 80.0};  // full rate for the 1, 2, 4, 6 mm burr
    double stiffnessMult = 1.0;  // 1 = today's contact stiffness
    double humOn = 1.0;
    double humAmpN = 0.1386;     // stock: 0.08 N on each of three axes
    double humHz = 79.577;       // stock: 500 rad/s
    double humDirection = HUM_DIAGONAL;
    double humSpinUpS = 0.0;
    double humSpinDownS = 0.0;
    double humLoadAmp = 0.0;     // extra hum at full device force (0.5 = +50%)
    double humLoadPitch = 0.0;   // frequency drop at full device force (0.2 = -20%)
    double forceEnable = 0.0;    // on while the tuner runs the simulator

    void clamp()
    {
        const Params stock;
        cutGate = fix(cutGate, 0.0, 1.0, stock.cutGate) >= 0.5 ? 1.0 : 0.0;
        cutStartN = fix(cutStartN, 0.0, kDeviceForceN - 0.1, stock.cutStartN);
        cutFullN = fix(cutFullN, 0.05, kDeviceForceN, stock.cutFullN);
        cutFullN = std::max(cutFullN, cutStartN + 0.05);
        cutCurve = fix(cutCurve, 0.25, 4.0, stock.cutCurve);
        for (int i = 0; i < 4; i++) {
            rateMm3[i] = fix(rateMm3[i], 0.0, kMaxCutRateMm3, stock.rateMm3[i]);
        }
        stiffnessMult = fix(stiffnessMult, 0.1, kMaxStiffnessMult, stock.stiffnessMult);
        humOn = fix(humOn, 0.0, 1.0, stock.humOn) >= 0.5 ? 1.0 : 0.0;
        humAmpN = fix(humAmpN, 0.0, kMaxHumN, stock.humAmpN);
        humHz = fix(humHz, kMinHumHz, kMaxHumHz, stock.humHz);
        humDirection = std::floor(fix(humDirection, 0.0, 2.0, stock.humDirection) + 0.5);
        humSpinUpS = fix(humSpinUpS, 0.0, 3.0, stock.humSpinUpS);
        humSpinDownS = fix(humSpinDownS, 0.0, 3.0, stock.humSpinDownS);
        humLoadAmp = fix(humLoadAmp, -1.0, 2.0, stock.humLoadAmp);
        humLoadPitch = fix(humLoadPitch, 0.0, 0.6, stock.humLoadPitch);
        forceEnable = fix(forceEnable, 0.0, 1.0, 0.0) >= 0.5 ? 1.0 : 0.0;
    }

    static double fix(double value, double low, double high, double fallback)
    {
        if (!std::isfinite(value)) {
            return fallback;
        }
        return std::min(std::max(value, low), high);
    }
};

// Share of the full cut rate at this contact force.
inline double cutFraction(const Params& p, double forceN)
{
    if (forceN <= p.cutStartN) {
        return 0.0;
    }
    const double x = (forceN - p.cutStartN) / (p.cutFullN - p.cutStartN);
    return std::pow(std::min(x, 1.0), p.cutCurve);
}

inline int burrIndex(double burrMm)
{
    if (burrMm < 1.5) return 0;
    if (burrMm < 3.0) return 1;
    if (burrMm < 5.0) return 2;
    return 3;
}

// Force-gated removal: the rate in mm3/s becomes a running voxel allowance.
class CutBudget {
public:
    // Voxels this tick may remove, at most `available`.
    int allow(const Params& p, double forceN, double burrMm, double voxelMm3, double dt, int available)
    {
        m_fraction = cutFraction(p, forceN);
        if (voxelMm3 <= 0.0 || available <= 0) {
            return 0;
        }
        const double step = std::min(std::max(dt, 0.0), 0.004);
        m_budget += p.rateMm3[burrIndex(burrMm)] * m_fraction * step / voxelMm3;
        // A stall must not save up a burst of removal.
        m_budget = std::min(m_budget, static_cast<double>(kMaxVoxelsPerTick));
        return std::min({static_cast<int>(m_budget), available, kMaxVoxelsPerTick});
    }
    void spend(int voxels) { m_budget = std::max(0.0, m_budget - voxels); }
    void idle()
    {
        m_budget = 0.0;
        m_fraction = 0.0;
    }
    double fraction() const { return m_fraction; }

private:
    double m_budget = 0.0;
    double m_fraction = 0.0;
};

// The motor hum. It is added after the safety guard, which would read a strong
// vibration as unstable contact and cut all force. It has its own hard cap.
class Hum {
public:
    // shaft: unit stylus axis in the device frame. out: force in the device frame.
    void generate(const Params& p, double dt, bool spinning, bool guardTripped,
                  const double shaft[3], double contactN, double out[3])
    {
        const double step = std::min(std::max(dt, 0.0), 0.004);
        const bool on = spinning && p.humOn > 0.5 && !guardTripped;
        if (guardTripped) {
            m_envelope = 0.0;
        } else if (on) {
            m_envelope = p.humSpinUpS > 1e-4 ? std::min(1.0, m_envelope + step / p.humSpinUpS) : 1.0;
        } else {
            m_envelope = p.humSpinDownS > 1e-4 ? std::max(0.0, m_envelope - step / p.humSpinDownS) : 0.0;
        }
        out[0] = out[1] = out[2] = 0.0;
        m_lastN = 0.0;
        if (m_envelope <= 0.0) {
            m_phase = 0.0;
            return;
        }
        const double load = std::min(std::max(contactN, 0.0) / kDeviceForceN, 1.0);
        const double amp = std::min(std::max(p.humAmpN * m_envelope * (1.0 + p.humLoadAmp * load), 0.0), kMaxHumN);
        const double hz = std::max(kMinHumHz, p.humHz * (1.0 - p.humLoadPitch * load));
        m_phase = std::fmod(m_phase + 2.0 * M_PI * hz * step, 2.0 * M_PI);

        const int direction = static_cast<int>(p.humDirection);
        if (direction == HUM_RADIAL) {
            // A spinning imbalance: the force turns around the shaft.
            double a[3], b[3];
            perpendicular(shaft, a, b);
            for (int i = 0; i < 3; i++) {
                out[i] = amp * (a[i] * std::cos(m_phase) + b[i] * std::sin(m_phase));
            }
        } else if (direction == HUM_SHAFT) {
            for (int i = 0; i < 3; i++) {
                out[i] = amp * shaft[i] * std::sin(m_phase);
            }
        } else {
            const double axis = amp * std::sin(m_phase) / std::sqrt(3.0);
            out[0] = out[1] = out[2] = axis;
        }
        m_lastN = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
    }
    void reset()
    {
        m_envelope = 0.0;
        m_phase = 0.0;
        m_lastN = 0.0;
    }
    double lastN() const { return m_lastN; }
    double envelope() const { return m_envelope; }

private:
    static void perpendicular(const double n[3], double a[3], double b[3])
    {
        const double ref[3] = {std::fabs(n[0]) < 0.8 ? 1.0 : 0.0, std::fabs(n[0]) < 0.8 ? 0.0 : 1.0, 0.0};
        a[0] = n[1] * ref[2] - n[2] * ref[1];
        a[1] = n[2] * ref[0] - n[0] * ref[2];
        a[2] = n[0] * ref[1] - n[1] * ref[0];
        const double length = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        for (int i = 0; i < 3; i++) {
            a[i] /= length > 1e-9 ? length : 1.0;
        }
        b[0] = n[1] * a[2] - n[2] * a[1];
        b[1] = n[2] * a[0] - n[0] * a[2];
        b[2] = n[0] * a[1] - n[1] * a[0];
    }

    double m_envelope = 0.0;
    double m_phase = 0.0;
    double m_lastN = 0.0;
};

// Force starts only from a near-zero request, so a burr that starts inside the bone cannot push on a docked stylus.
class ForceGate {
public:
    bool update(bool enabled, double requestedN, double dt)
    {
        if (!enabled) {
            m_open = false;
            m_quiet = 0.0;
        } else if (!m_open) {
            m_quiet = requestedN < kArmQuietN ? m_quiet + std::max(dt, 0.0) : 0.0;
            m_open = m_quiet >= kArmQuietS;
        }
        return m_open;
    }
    bool open() const { return m_open; }

private:
    bool m_open = false;
    double m_quiet = 0.0;
};

// Protects the motors: a force held well above the continuous rating is eased
// down to it. Short hard pushes pass untouched.
class SustainedLimit {
public:
    double update(double forceN, double dt)
    {
        const double step = std::min(std::max(dt, 0.0), 0.004);
        const double rms = std::sqrt(m_power);
        if (rms >= kSustainedTripN) m_limiting = true;
        else if (rms <= kSustainedClearN) m_limiting = false;
        const double target = (m_limiting && forceN > kContinuousN) ? kContinuousN / forceN : 1.0;
        const double maxChange = 5.0 * step;  // full range in 0.2 s
        m_scale += std::min(std::max(target - m_scale, -maxChange), maxChange);
        const double sent = forceN * m_scale;
        m_power += (sent * sent - m_power) * (1.0 - std::exp(-step / kSustainedTauS));
        return m_scale;
    }
    bool limiting() const { return m_limiting; }

private:
    double m_power = 0.0;
    double m_scale = 1.0;
    bool m_limiting = false;
};

// The two shared files. Physics thread only.
class Link {
public:
    ~Link() { close(); }

    bool open(const std::string& paramPath = "/dev/shm/fivrs_haptic_tuning",
              const std::string& telemetryPath = "/dev/shm/fivrs_haptic_telemetry")
    {
        const size_t paramBytes = kParamHeaderBytes + kParamCount * sizeof(double);
        const size_t telemetryBytes = kTelemetryHeaderBytes + kTelemetryRows * kTelemetryFields * sizeof(float);
        m_params = map(paramPath, paramBytes);
        m_telemetry = map(telemetryPath, telemetryBytes);
        if (!m_params || !m_telemetry
            || word(m_params, 0) != kParamMagic || word(m_params, 1) != kVersion
            || word(m_telemetry, 0) != kTelemetryMagic || word(m_telemetry, 1) != kVersion) {
            close();
            return false;
        }
        return true;
    }
    bool isOpen() const { return m_params != nullptr; }

    // Keeps the last good settings if the writer is mid-update.
    const Params& read()
    {
        if (!m_params) {
            return m_current;
        }
        auto* seq = reinterpret_cast<std::atomic<uint32_t>*>(m_params + 8);
        for (int attempt = 0; attempt < 3; attempt++) {
            const uint32_t before = seq->load(std::memory_order_acquire);
            if (before & 1u) {
                continue;
            }
            double raw[kParamCount];
            std::memcpy(raw, m_params + kParamHeaderBytes, sizeof(raw));
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq->load(std::memory_order_relaxed) != before) {
                continue;
            }
            if (before != m_lastSeq || !m_haveRead) {
                m_lastSeq = before;
                m_haveRead = true;
                Params p;
                int i = 0;
                p.cutGate = raw[i++];
                p.cutStartN = raw[i++];
                p.cutFullN = raw[i++];
                p.cutCurve = raw[i++];
                for (double& rate : p.rateMm3) {
                    rate = raw[i++];
                }
                p.stiffnessMult = raw[i++];
                p.humOn = raw[i++];
                p.humAmpN = raw[i++];
                p.humHz = raw[i++];
                p.humDirection = raw[i++];
                p.humSpinUpS = raw[i++];
                p.humSpinDownS = raw[i++];
                p.humLoadAmp = raw[i++];
                p.humLoadPitch = raw[i++];
                p.forceEnable = raw[i++];
                p.clamp();
                m_current = p;
            }
            break;
        }
        return m_current;
    }

    // time, contact N, hum N, sent N, cut fraction, voxels removed, flags, burr mm
    void log(const float row[kTelemetryFields])
    {
        if (!m_telemetry) {
            return;
        }
        auto* count = reinterpret_cast<std::atomic<uint64_t>*>(m_telemetry + 16);
        const uint64_t n = count->load(std::memory_order_relaxed);
        auto* rows = reinterpret_cast<float*>(m_telemetry + kTelemetryHeaderBytes);
        std::memcpy(rows + (n % kTelemetryRows) * kTelemetryFields, row, kTelemetryFields * sizeof(float));
        count->store(n + 1, std::memory_order_release);
    }

private:
    static uint32_t word(const unsigned char* base, int index)
    {
        uint32_t value;
        std::memcpy(&value, base + index * 4, 4);
        return value;
    }
    unsigned char* map(const std::string& path, size_t bytes)
    {
        const int fd = ::open(path.c_str(), O_RDWR);
        if (fd < 0) {
            return nullptr;
        }
        struct stat info;
        void* memory = MAP_FAILED;
        if (fstat(fd, &info) == 0 && static_cast<size_t>(info.st_size) == bytes) {
            memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        }
        ::close(fd);
        return memory == MAP_FAILED ? nullptr : static_cast<unsigned char*>(memory);
    }
    void close()
    {
        if (m_params) {
            munmap(m_params, kParamHeaderBytes + kParamCount * sizeof(double));
        }
        if (m_telemetry) {
            munmap(m_telemetry, kTelemetryHeaderBytes + kTelemetryRows * kTelemetryFields * sizeof(float));
        }
        m_params = nullptr;
        m_telemetry = nullptr;
    }

    unsigned char* m_params = nullptr;
    unsigned char* m_telemetry = nullptr;
    Params m_current;
    uint32_t m_lastSeq = 0;
    bool m_haveRead = false;
};

}  // namespace haptic_tuning

#endif
