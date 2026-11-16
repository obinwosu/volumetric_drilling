#ifndef HAPTIC_DEVICE_PROBE_H
#define HAPTIC_DEVICE_PROBE_H

// Reads what the 3D Systems Touch Diagnostic app shows (encoders, joint and gimbal
// angles, buttons, inkwell, motor temperature, calibration, servo rate) plus the
// health signals behind "force drops out": force output enabled, the force the
// servo loop last commanded, and OpenHaptics errors raised in the servo thread.
//
// Shared by the drilling plugin (inside AMBF, which owns the device) and the
// standalone probe scripts/haptic_monitor/haptic_probe.cpp (simulator closed).
// OpenHaptics is reached through dlsym, so the plugin does not link libHD and a
// box without it (or a simulated device) just reports "HD not loaded".
//
// Cost in the servo thread: read() runs as one hdScheduleSynchronous callback
// (a handful of hdGet calls, a few microseconds) about 20 times a second. It never
// calls hdBeginFrame/hdEndFrame and never sets a force, so the frame CHAI3D runs
// on the same tick is untouched. Measured on the Touch: values read this way are
// live (match the Diagnostic app), and the servo rate stays ~1 kHz.
//
// Not readable through HD on this box: firmware version (HD returns 0 / error
// 0x103), so it is reported as unknown rather than faked.

#include <dlfcn.h>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace haptic_probe {

// --- OpenHaptics types and constants (hdDefines.h, OpenHaptics 3.4) ---
typedef unsigned int HHD;
typedef unsigned int HDenum;
typedef unsigned int HDerror;
typedef unsigned char HDboolean;
typedef unsigned int HDCallbackCode;
typedef unsigned long HDSchedulerHandle;
typedef HDCallbackCode (*HDSchedulerCallback)(void*);
struct HDErrorInfo { HDerror errorCode; int internalErrorCode; HHD hHD; };

constexpr HDenum HD_CURRENT_BUTTONS = 0x2000;
constexpr HDenum HD_CURRENT_SAFETY_SWITCH = 0x2001;
constexpr HDenum HD_CURRENT_INKWELL_SWITCH = 0x2002;
constexpr HDenum HD_CURRENT_ENCODER_VALUES = 0x2010;
constexpr HDenum HD_CURRENT_POSITION = 0x2050;
constexpr HDenum HD_CURRENT_VELOCITY = 0x2051;
constexpr HDenum HD_CURRENT_JOINT_ANGLES = 0x2100;
constexpr HDenum HD_CURRENT_GIMBAL_ANGLES = 0x2150;
constexpr HDenum HD_DEVICE_MODEL_TYPE = 0x2501;
constexpr HDenum HD_DEVICE_DRIVER_VERSION = 0x2502;
constexpr HDenum HD_DEVICE_VENDOR = 0x2503;
constexpr HDenum HD_DEVICE_SERIAL_NUMBER = 0x2504;
constexpr HDenum HD_CALIBRATION_STYLE = 0x2555;
constexpr HDenum HD_INSTANTANEOUS_UPDATE_RATE = 0x2601;
constexpr HDenum HD_NOMINAL_MAX_FORCE = 0x2603;
constexpr HDenum HD_NOMINAL_MAX_CONTINUOUS_FORCE = 0x2604;
constexpr HDenum HD_MOTOR_TEMPERATURE = 0x2605;
constexpr HDenum HD_CURRENT_FORCE = 0x2700;
constexpr HDenum HD_FORCE_OUTPUT = 0x4000;
constexpr HDenum HD_CALIBRATION_OK = 0x5000;
constexpr HDenum HD_CALIBRATION_NEEDS_UPDATE = 0x5001;
constexpr HDenum HD_CALIBRATION_NEEDS_MANUAL_INPUT = 0x5002;
constexpr HDCallbackCode HD_CALLBACK_DONE = 0;
constexpr HDCallbackCode HD_CALLBACK_CONTINUE = 1;
constexpr unsigned short HD_MIN_SCHEDULER_PRIORITY = 0;
constexpr unsigned short HD_DEFAULT_SCHEDULER_PRIORITY = 32767;
constexpr HDerror HD_SUCCESS = 0;
constexpr int kMaxErrorsPerRead = 16;

struct Api {
    bool ok = false;
    HHD (*initDevice)(const char*) = nullptr;
    void (*disableDevice)(HHD) = nullptr;
    void (*makeCurrentDevice)(HHD) = nullptr;
    void (*beginFrame)(HHD) = nullptr;
    void (*endFrame)(HHD) = nullptr;
    HDErrorInfo (*getError)() = nullptr;
    const char* (*getErrorString)(HDerror) = nullptr;
    void (*enable)(HDenum) = nullptr;
    void (*disable)(HDenum) = nullptr;
    HDboolean (*isEnabled)(HDenum) = nullptr;
    void (*getBooleanv)(HDenum, HDboolean*) = nullptr;
    void (*getIntegerv)(HDenum, int*) = nullptr;
    void (*getDoublev)(HDenum, double*) = nullptr;
    void (*getLongv)(HDenum, long*) = nullptr;
    const char* (*getString)(HDenum) = nullptr;
    void (*setDoublev)(HDenum, const double*) = nullptr;
    HDenum (*checkCalibration)() = nullptr;
    void (*startScheduler)() = nullptr;
    void (*stopScheduler)() = nullptr;
    void (*scheduleSynchronous)(HDSchedulerCallback, void*, unsigned short) = nullptr;
    HDSchedulerHandle (*scheduleAsynchronous)(HDSchedulerCallback, void*, unsigned short) = nullptr;
    void (*unschedule)(HDSchedulerHandle) = nullptr;

    // handle: dlopen("libHD.so") for the probe, RTLD_DEFAULT inside AMBF (libHD
    // is already loaded there by CHAI3D's libhdPhantom).
    bool load(void* handle)
    {
        auto get = [&](const char* name) { return dlsym(handle, name); };
#define HP_LOAD(field, name) field = reinterpret_cast<decltype(field)>(get(name)); if (!field) { ok = false; return false; }
        HP_LOAD(initDevice, "hdInitDevice")
        HP_LOAD(disableDevice, "hdDisableDevice")
        HP_LOAD(makeCurrentDevice, "hdMakeCurrentDevice")
        HP_LOAD(beginFrame, "hdBeginFrame")
        HP_LOAD(endFrame, "hdEndFrame")
        HP_LOAD(getError, "hdGetError")
        HP_LOAD(getErrorString, "hdGetErrorString")
        HP_LOAD(enable, "hdEnable")
        HP_LOAD(disable, "hdDisable")
        HP_LOAD(isEnabled, "hdIsEnabled")
        HP_LOAD(getBooleanv, "hdGetBooleanv")
        HP_LOAD(getIntegerv, "hdGetIntegerv")
        HP_LOAD(getDoublev, "hdGetDoublev")
        HP_LOAD(getLongv, "hdGetLongv")
        HP_LOAD(getString, "hdGetString")
        HP_LOAD(setDoublev, "hdSetDoublev")
        HP_LOAD(checkCalibration, "hdCheckCalibration")
        HP_LOAD(startScheduler, "hdStartScheduler")
        HP_LOAD(stopScheduler, "hdStopScheduler")
        HP_LOAD(scheduleSynchronous, "hdScheduleSynchronous")
        HP_LOAD(scheduleAsynchronous, "hdScheduleAsynchronous")
        HP_LOAD(unschedule, "hdUnschedule")
#undef HP_LOAD
        ok = true;
        return true;
    }

    std::string errorText(HDerror code) const
    {
        const char* text = getErrorString ? getErrorString(code) : nullptr;
        return text ? std::string(text) : std::string("unknown");
    }
};

// Static identification, read once outside the scheduler.
struct Info {
    std::string model, vendor, serial, driver;
    double maxForceN = 0.0;
    double maxContinuousForceN = 0.0;
    int calibrationStyle = 0;
};

// One read in the servo thread.
struct Sample {
    bool valid = false;
    long encoders[6] = {0, 0, 0, 0, 0, 0};      // turret, thigh, shin, yaw, pitch, roll
    double joints[3] = {0, 0, 0};               // rad: turret, thigh, shin
    double gimbal[3] = {0, 0, 0};               // rad: yaw, pitch, roll
    double temperature[6] = {-1, -1, -1, -1, -1, -1}; // 0..1 motor heat model, -1 = not reported
    double positionMm[3] = {0, 0, 0};
    double velocityMmS[3] = {0, 0, 0};
    double forceN[3] = {0, 0, 0};               // force the servo loop last set
    int buttons = 0;
    bool inkwell = false;
    bool safetySwitch = false;
    int rateHz = 0;
    HDenum calibration = 0;
    bool forceOutput = false;
    int errorCount = 0;
    HDerror errors[kMaxErrorsPerRead] = {};
};

// Call inside a scheduler callback (servo thread). No frame, no force set.
// Errors raised by the frame CHAI3D runs in this thread (e.g. max velocity
// exceeded, which can switch force output off) sit on this thread's error
// stack, so draining it here is how they are seen at all: CHAI3D never reads it.
inline void read(const Api& hd, HHD device, Sample& s)
{
    hd.makeCurrentDevice(device);
    hd.getLongv(HD_CURRENT_ENCODER_VALUES, s.encoders);
    hd.getDoublev(HD_CURRENT_JOINT_ANGLES, s.joints);
    hd.getDoublev(HD_CURRENT_GIMBAL_ANGLES, s.gimbal);
    hd.getDoublev(HD_MOTOR_TEMPERATURE, s.temperature);
    hd.getDoublev(HD_CURRENT_POSITION, s.positionMm);
    hd.getDoublev(HD_CURRENT_VELOCITY, s.velocityMmS);
    hd.getDoublev(HD_CURRENT_FORCE, s.forceN);
    hd.getIntegerv(HD_CURRENT_BUTTONS, &s.buttons);
    HDboolean flag = 0;
    hd.getBooleanv(HD_CURRENT_INKWELL_SWITCH, &flag);
    s.inkwell = flag != 0;
    flag = 0;
    hd.getBooleanv(HD_CURRENT_SAFETY_SWITCH, &flag);
    s.safetySwitch = flag != 0;
    hd.getIntegerv(HD_INSTANTANEOUS_UPDATE_RATE, &s.rateHz);
    s.calibration = hd.checkCalibration();
    s.forceOutput = hd.isEnabled(HD_FORCE_OUTPUT) != 0;
    s.errorCount = 0;
    for (int i = 0; i < 64; i++) {
        const HDErrorInfo e = hd.getError();
        if (e.errorCode == HD_SUCCESS) {
            break;
        }
        if (s.errorCount < kMaxErrorsPerRead) {
            s.errors[s.errorCount++] = e.errorCode;
        }
    }
    s.valid = true;
}

inline Info readInfo(const Api& hd, HHD device)
{
    Info info;
    hd.makeCurrentDevice(device);
    auto str = [&](HDenum name) {
        const char* text = hd.getString(name);
        return text ? std::string(text) : std::string();
    };
    info.model = str(HD_DEVICE_MODEL_TYPE);
    info.vendor = str(HD_DEVICE_VENDOR);
    info.serial = str(HD_DEVICE_SERIAL_NUMBER);
    info.driver = str(HD_DEVICE_DRIVER_VERSION);
    hd.getDoublev(HD_NOMINAL_MAX_FORCE, &info.maxForceN);
    hd.getDoublev(HD_NOMINAL_MAX_CONTINUOUS_FORCE, &info.maxContinuousForceN);
    hd.getIntegerv(HD_CALIBRATION_STYLE, &info.calibrationStyle);
    while (hd.getError().errorCode != HD_SUCCESS) {
    }
    return info;
}

inline const char* calibrationText(HDenum code)
{
    switch (code) {
    case HD_CALIBRATION_OK: return "OK";
    case HD_CALIBRATION_NEEDS_UPDATE: return "needs update";
    case HD_CALIBRATION_NEEDS_MANUAL_INPUT: return "needs manual input (put the stylus in the inkwell)";
    default: return "unknown";
    }
}

inline double wallSeconds()
{
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline std::string jsonString(const std::string& in)
{
    std::string out = "\"";
    for (char c : in) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { out += ' '; }
        else { out += c; }
    }
    return out + "\"";
}

// JSON object for one device. extra: more "key":value pairs (starting with ',').
inline std::string deviceJson(const std::string& role, const Info& info, const Sample& s,
                              const std::string& extra = "")
{
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(4);
    o << "{\"role\":" << jsonString(role)
      << ",\"model\":" << jsonString(info.model)
      << ",\"vendor\":" << jsonString(info.vendor)
      << ",\"serial\":" << jsonString(info.serial)
      << ",\"driver\":" << jsonString(info.driver)
      << ",\"max_force_n\":" << info.maxForceN
      << ",\"max_continuous_force_n\":" << info.maxContinuousForceN
      << ",\"valid\":" << (s.valid ? "true" : "false");
    o << ",\"encoders\":[";
    for (int i = 0; i < 6; i++) o << (i ? "," : "") << s.encoders[i];
    o << "],\"joint_deg\":[";
    for (int i = 0; i < 3; i++) o << (i ? "," : "") << s.joints[i] * 180.0 / M_PI;
    o << "],\"gimbal_deg\":[";
    for (int i = 0; i < 3; i++) o << (i ? "," : "") << s.gimbal[i] * 180.0 / M_PI;
    o << "],\"temperature\":[";
    bool first = true;
    for (int i = 0; i < 6; i++) {
        if (s.temperature[i] < 0.0) continue;
        o << (first ? "" : ",") << s.temperature[i];
        first = false;
    }
    o << "],\"position_mm\":[" << s.positionMm[0] << "," << s.positionMm[1] << "," << s.positionMm[2]
      << "],\"velocity_mm_s\":[" << s.velocityMmS[0] << "," << s.velocityMmS[1] << "," << s.velocityMmS[2]
      << "],\"device_force_n\":[" << s.forceN[0] << "," << s.forceN[1] << "," << s.forceN[2] << "]"
      << ",\"buttons\":" << s.buttons
      << ",\"inkwell\":" << (s.inkwell ? "true" : "false")
      << ",\"safety_switch\":" << (s.safetySwitch ? "true" : "false")
      << ",\"servo_hz\":" << s.rateHz
      << ",\"calibration\":" << jsonString(calibrationText(s.calibration))
      << ",\"force_output\":" << (s.forceOutput ? "true" : "false")
      << extra << "}";
    return o.str();
}

inline void writeAtomically(const std::string& path, const std::string& text)
{
    const std::string temp = path + ".tmp";
    {
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << text;
    }
    std::rename(temp.c_str(), path.c_str());
}

inline std::string defaultHealthLogPath()
{
    const char* home = std::getenv("HOME");
    std::string dir = std::string(home ? home : "/tmp") + "/.local/share/fivrs";
    std::string partial;
    std::stringstream ss(dir);
    std::string piece;
    while (std::getline(ss, piece, '/')) {
        if (piece.empty()) continue;
        partial += "/" + piece;
        mkdir(partial.c_str(), 0755);
    }
    return dir + "/haptic_health.log";
}

// Health log: one line per change (connected, servo rate low, force output off,
// HD errors, calibration, temperature, plugin events) and a summary every
// minute. Always written while a device is read, whether or not the monitor
// window is open. Rotates to .1 at 5 MB.
class HealthLog {
public:
    explicit HealthLog(std::string source, std::string path = defaultHealthLogPath())
        : m_source(std::move(source)), m_path(std::move(path)) {}

    const std::string& path() const { return m_path; }

    void line(const std::string& device, const std::string& text)
    {
        struct stat st;
        if (stat(m_path.c_str(), &st) == 0 && st.st_size > 5 * 1024 * 1024) {
            std::rename(m_path.c_str(), (m_path + ".1").c_str());
        }
        std::ofstream out(m_path, std::ios::app);
        if (!out) return;
        const double now = wallSeconds();
        const std::time_t secs = static_cast<std::time_t>(now);
        std::tm local{};
        localtime_r(&secs, &local);
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &local);
        char millis[8];
        std::snprintf(millis, sizeof(millis), ".%03d", static_cast<int>((now - std::floor(now)) * 1000.0));
        out << stamp << millis << " [" << m_source << "] " << device << ": " << text << "\n";
    }

private:
    std::string m_source;
    std::string m_path;
};

// Turns samples into health events for one device. Thresholds: servo below 900 Hz
// for 3 reads in a row (~150 ms) is "low"; motor heat model over 0.8 warns.
class HealthTracker {
public:
    HealthTracker(HealthLog* log, std::string device) : m_log(log), m_device(std::move(device)) {}

    void setDevice(const std::string& device) { m_device = device; }

    void event(const std::string& text) { if (m_log) m_log->line(m_device, text); }

    void update(const Api& hd, const Sample& s, double now)
    {
        if (!m_log) return;
        if (!s.valid) return;
        // The very first read after opening can report a stale calibration state.
        if (++m_reads == 1) return;
        if (!m_started) {
            m_started = true;
            m_windowStart = now;
            m_forceOutput = s.forceOutput;
            m_calibration = s.calibration;
            std::ostringstream o;
            o << "reading: servo " << s.rateHz << " Hz, force output " << (s.forceOutput ? "on" : "OFF")
              << ", calibration " << calibrationText(s.calibration);
            event(o.str());
        }
        // Servo rate.
        if (s.rateHz > 0 && s.rateHz < kLowRateHz) {
            if (++m_lowReads == 3) {
                m_lowSince = now;
                event("servo loop slow: " + std::to_string(s.rateHz) + " Hz (normal ~1000)");
            }
        } else {
            if (m_lowReads >= 3) {
                std::ostringstream o;
                o.precision(2);
                o << std::fixed << "servo loop back to " << s.rateHz << " Hz after " << (now - m_lowSince) << " s";
                event(o.str());
            }
            m_lowReads = 0;
        }
        if (s.forceOutput != m_forceOutput) {
            m_forceOutput = s.forceOutput;
            event(s.forceOutput ? "force output back ON"
                                : "FORCE OUTPUT OFF: OpenHaptics disabled the motors (no force reaches the hand)");
        }
        if (s.calibration != m_calibration) {
            m_calibration = s.calibration;
            event(std::string("calibration: ") + calibrationText(s.calibration));
        }
        for (int i = 0; i < s.errorCount; i++) {
            const HDerror code = s.errors[i];
            m_errorTotal += 1;
            auto& last = m_lastErrorLogged[code];
            m_errorCounts[code] += 1;
            if (now - last >= 1.0) {
                last = now;
                char hex[16];
                std::snprintf(hex, sizeof(hex), "0x%03X", code);
                event(std::string("HD error ") + hex + " " + hd.errorText(code)
                      + " (" + std::to_string(m_errorCounts[code]) + " so far)");
            }
        }
        double hottest = -1.0;
        for (double t : s.temperature) hottest = std::max(hottest, t);
        if (hottest > kHotTemperature && !m_hot) {
            m_hot = true;
            event("motor hot: heat model " + std::to_string(static_cast<int>(hottest * 100)) + "%");
        } else if (hottest >= 0.0 && hottest < kHotTemperature - 0.1 && m_hot) {
            m_hot = false;
            event("motor cooled: " + std::to_string(static_cast<int>(hottest * 100)) + "%");
        }
        // Minute summary.
        if (s.rateHz > 0) {
            m_rateMin = m_rateCount ? std::min(m_rateMin, s.rateHz) : s.rateHz;
            m_rateSum += s.rateHz;
            m_rateCount += 1;
        }
        m_hottest = std::max(m_hottest, hottest);
        if (now - m_windowStart >= 60.0) {
            std::ostringstream o;
            o << "summary 60 s: servo mean " << (m_rateCount ? m_rateSum / m_rateCount : 0)
              << " Hz min " << (m_rateCount ? m_rateMin : 0) << " Hz, force output "
              << (s.forceOutput ? "on" : "OFF") << ", HD errors " << m_errorTotal
              << ", hottest motor " << static_cast<int>(std::max(0.0, m_hottest) * 100) << "%";
            if (!m_summaryExtra.empty()) o << ", " << m_summaryExtra;
            event(o.str());
            m_windowStart = now;
            m_rateMin = 0;
            m_rateSum = 0;
            m_rateCount = 0;
            m_errorTotal = 0;
            m_hottest = -1.0;
        }
    }

    // Plugin adds e.g. physics rate / longest physics gap to the minute summary.
    void setSummaryExtra(const std::string& text) { m_summaryExtra = text; }
    bool summaryDue(double now) const { return m_started && now - m_windowStart >= 60.0; }

    static constexpr int kLowRateHz = 900;
    static constexpr double kHotTemperature = 0.8;

private:
    HealthLog* m_log;
    std::string m_device;
    bool m_started = false;
    long m_reads = 0;
    int m_lowReads = 0;
    double m_lowSince = 0.0;
    bool m_forceOutput = true;
    HDenum m_calibration = 0;
    bool m_hot = false;
    std::map<HDerror, double> m_lastErrorLogged;
    std::map<HDerror, long> m_errorCounts;
    double m_windowStart = 0.0;
    int m_rateMin = 0;
    long m_rateSum = 0;
    int m_rateCount = 0;
    long m_errorTotal = 0;
    double m_hottest = -1.0;
    std::string m_summaryExtra;
};

} // namespace haptic_probe

#endif
