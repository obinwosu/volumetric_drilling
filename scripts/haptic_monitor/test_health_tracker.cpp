// Health log events from haptic_device_probe.h, without a device.
// Built and run by scripts/study_gui/test_haptic_monitor.py.
#include "../../plugin/volumetric_drilling/haptic_device_probe.h"

#include <cassert>
#include <iostream>

using namespace haptic_probe;

static const char* fakeErrorString(HDerror code)
{
    return code == 0x203 ? "Exceeded max velocity" : "other";
}

static std::string readAll(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static bool has(const std::string& text, const std::string& part)
{
    return text.find(part) != std::string::npos;
}

int main(int argc, char** argv)
{
    const std::string path = argc > 1 ? argv[1] : "/tmp/test_haptic_health.log";
    std::remove(path.c_str());
    Api hd;
    hd.getErrorString = fakeErrorString;
    HealthLog log("test", path);
    HealthTracker tracker(&log, "drill (SN 1)");

    Sample s;
    s.valid = true;
    s.rateHz = 1000;
    s.forceOutput = true;
    s.calibration = HD_CALIBRATION_OK;
    s.temperature[0] = 0.1;
    double t = 1000.0;
    tracker.update(hd, s, t);   // first read after opening is skipped
    assert(readAll(path).empty());
    tracker.update(hd, s, t);
    std::string text = readAll(path);
    assert(has(text, "[test] drill (SN 1): reading: servo 1000 Hz, force output on, calibration OK"));

    // Two slow reads are noise; the third is logged, and the recovery too.
    s.rateHz = 600;
    tracker.update(hd, s, t += 0.05);
    tracker.update(hd, s, t += 0.05);
    assert(!has(readAll(path), "servo loop slow"));
    tracker.update(hd, s, t += 0.05);
    assert(has(readAll(path), "servo loop slow: 600 Hz"));
    s.rateHz = 1000;
    tracker.update(hd, s, t += 0.05);
    assert(has(readAll(path), "servo loop back to 1000 Hz"));

    // Force output switched off by OpenHaptics, then back.
    s.forceOutput = false;
    tracker.update(hd, s, t += 0.05);
    assert(has(readAll(path), "FORCE OUTPUT OFF"));
    s.forceOutput = true;
    tracker.update(hd, s, t += 0.05);
    assert(has(readAll(path), "force output back ON"));

    // Errors: logged with their text, repeats within a second only counted.
    s.errorCount = 2;
    s.errors[0] = 0x203;
    s.errors[1] = 0x203;
    tracker.update(hd, s, t += 0.05);
    text = readAll(path);
    assert(has(text, "HD error 0x203 Exceeded max velocity (1 so far)"));
    assert(!has(text, "(2 so far)"));
    s.errorCount = 1;
    tracker.update(hd, s, t += 1.5);
    assert(has(readAll(path), "(3 so far)"));
    s.errorCount = 0;

    // Calibration change and a hot motor.
    s.calibration = HD_CALIBRATION_NEEDS_MANUAL_INPUT;
    s.temperature[1] = 0.85;
    tracker.update(hd, s, t += 0.05);
    text = readAll(path);
    assert(has(text, "calibration: needs manual input"));
    assert(has(text, "motor hot: heat model 85%"));

    // Minute summary with the plugin's extra text.
    tracker.setSummaryExtra("physics 1000.0 Hz");
    tracker.update(hd, s, t += 61.0);
    text = readAll(path);
    assert(has(text, "summary 60 s: servo mean"));
    assert(has(text, "HD errors 3"));
    assert(has(text, "physics 1000.0 Hz"));

    // The JSON a device turns into has the fields the window reads.
    Info info;
    info.model = "Touch";
    info.serial = "23059000020";
    const std::string json = deviceJson("drill", info, s);
    assert(has(json, "\"encoders\":[") && has(json, "\"servo_hz\":1000") && has(json, "\"temperature\":["));
    std::cout << "health tracker ok" << std::endl;
    return 0;
}
