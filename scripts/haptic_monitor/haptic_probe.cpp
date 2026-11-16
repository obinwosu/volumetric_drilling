// Standalone Touch reader for the haptic monitor when the simulator is closed.
//
// Opens the Touch(es) through OpenHaptics, never sends force (force output is
// switched off), and about 20 times a second writes /tmp/fivrs-rtx/haptic_direct.json
// and appends health events to ~/.local/share/fivrs/haptic_health.log.
//
// Only one process can own a Touch, so this exits (and frees the device) when
//  - /tmp/fivrs-rtx/haptic_probe.release appears (the study GUI writes it before
//    Start simulation),
//  - an ambf_simulator process appears,
//  - its parent (the monitor window) exits, or on SIGINT / SIGTERM.
// While it owns the device it holds an flock on /tmp/fivrs-rtx/haptic_probe.lock,
// so the GUI can wait for the device to be free.
//
// Build: scripts/haptic_monitor/build_probe.sh (the monitor builds it when missing).

#include "../../plugin/volumetric_drilling/haptic_device_probe.h"

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <thread>

using namespace haptic_probe;

static volatile sig_atomic_t g_stop = 0;
static void onSignal(int) { g_stop = 1; }

static const char* kDir = "/tmp/fivrs-rtx";

static bool ambfRunning()
{
    DIR* proc = opendir("/proc");
    if (!proc) return false;
    bool found = false;
    while (dirent* entry = readdir(proc)) {
        if (entry->d_name[0] < '0' || entry->d_name[0] > '9') continue;
        char path[300];
        std::snprintf(path, sizeof(path), "/proc/%s/comm", entry->d_name);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        char comm[64] = {0};
        if (std::fgets(comm, sizeof(comm), f)) {
            if (std::strncmp(comm, "ambf_simulator", 14) == 0) found = true;
        }
        std::fclose(f);
        if (found) break;
    }
    closedir(proc);
    return found;
}

static bool releaseRequested()
{
    struct stat st;
    const std::string path = std::string(kDir) + "/haptic_probe.release";
    if (stat(path.c_str(), &st) != 0) return false;
    return wallSeconds() - st.st_mtime < 60.0;
}

struct Device {
    std::string role;
    std::string configName;
    HHD handle = 0;
    bool open = false;
    Info info;
    Sample sample;
    HealthTracker tracker{nullptr, ""};
};

static Api g_hd;
static std::vector<Device*> g_devices;

static HDCallbackCode frameCallback(void*)
{
    // Keep the device state fresh; no force is set (force output is off).
    for (Device* d : g_devices) {
        g_hd.makeCurrentDevice(d->handle);
        g_hd.beginFrame(d->handle);
        g_hd.endFrame(d->handle);
    }
    return HD_CALLBACK_CONTINUE;
}

static HDCallbackCode readCallback(void*)
{
    for (Device* d : g_devices) {
        read(g_hd, d->handle, d->sample);
    }
    return HD_CALLBACK_DONE;
}

static void writeState(const std::string& status, const std::string& error)
{
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(3);
    o << "{\"source\":\"direct\",\"wall_time\":" << wallSeconds()
      << ",\"pid\":" << getpid()
      << ",\"status\":" << jsonString(status)
      << ",\"error\":" << jsonString(error)
      << ",\"devices\":[";
    bool first = true;
    for (Device* d : g_devices) {
        o << (first ? "" : ",") << deviceJson(d->role, d->info, d->sample,
                                              ",\"force_output_note\":\"off: the monitor sends no force\"");
        first = false;
    }
    o << "]}\n";
    writeAtomically(std::string(kDir) + "/haptic_direct.json", o.str());
}

int main(int argc, char** argv)
{
    double hz = 20.0;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--hz") == 0 && i + 1 < argc) hz = std::max(1.0, std::atof(argv[++i]));
    }
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    mkdir(kDir, 0755);

    HealthLog log("direct");
    if (ambfRunning()) {
        writeState("sim_running", "The simulator owns the Touch; the monitor reads the simulator's feed.");
        return 2;
    }
    const int lockFd = open((std::string(kDir) + "/haptic_probe.lock").c_str(), O_CREAT | O_RDWR, 0644);
    if (lockFd < 0 || flock(lockFd, LOCK_EX | LOCK_NB) != 0) {
        std::cerr << "haptic_probe: another probe owns the device" << std::endl;
        return 3;
    }

    void* lib = dlopen("libHD.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib || !g_hd.load(lib)) {
        const std::string why = std::string("OpenHaptics libHD.so not loadable: ") + (dlerror() ? dlerror() : "missing symbols");
        writeState("no_library", why);
        log.line("probe", why);
        return 4;
    }

    Device drill;
    drill.role = "drill";
    drill.configName = "";      // HD_DEFAULT_DEVICE
    Device suction;
    suction.role = "suction";
    suction.configName = "Phantom2";

    std::string initError;
    for (Device* d : {&drill, &suction}) {
        const HHD handle = g_hd.initDevice(d->configName.empty() ? nullptr : d->configName.c_str());
        const HDErrorInfo e = g_hd.getError();
        if (e.errorCode != HD_SUCCESS || handle == 0xFFFFFFFFu) {
            if (d == &drill) {
                char hex[16];
                std::snprintf(hex, sizeof(hex), "0x%03X", e.errorCode);
                initError = std::string("could not open the drill Touch (default device): HD error ") + hex + " "
                    + g_hd.errorText(e.errorCode);
                log.line("probe", initError);
            }
            while (g_hd.getError().errorCode != HD_SUCCESS) {}
            continue;
        }
        d->handle = handle;
        d->open = true;
        d->info = readInfo(g_hd, handle);
        g_hd.makeCurrentDevice(handle);
        g_hd.disable(HD_FORCE_OUTPUT);
        d->tracker = HealthTracker(&log, d->role + " (SN " + d->info.serial + ")");
        g_devices.push_back(d);
    }
    if (g_devices.empty()) {
        writeState("no_device", initError);
        flock(lockFd, LOCK_UN);
        return 5;
    }
    for (Device* d : g_devices) {
        d->tracker.event("opened by the haptic monitor (simulator closed; force output off). Model "
                         + d->info.model + ", driver " + d->info.driver);
    }
    const HDSchedulerHandle frame = g_hd.scheduleAsynchronous(frameCallback, nullptr, HD_DEFAULT_SCHEDULER_PRIORITY);
    g_hd.startScheduler();
    // After OpenHaptics is up: it installs its own SIGINT/SIGTERM handlers that
    // exit at once, which would skip closing the device and the log line.
    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);

    const auto period = std::chrono::duration<double>(1.0 / hz);
    std::string reason = "monitor closed";
    int checkTick = 0;
    while (!g_stop) {
        const auto start = std::chrono::steady_clock::now();
        g_hd.scheduleSynchronous(readCallback, nullptr, HD_MIN_SCHEDULER_PRIORITY);
        const double now = wallSeconds();
        for (Device* d : g_devices) {
            d->tracker.update(g_hd, d->sample, now);
        }
        writeState("ok", initError);
        if (releaseRequested()) { reason = "released for the simulator"; break; }
        if (++checkTick % 2 == 0 && ambfRunning()) { reason = "simulator started"; break; }
        std::this_thread::sleep_until(start + period);
    }
    g_hd.stopScheduler();
    g_hd.unschedule(frame);
    for (Device* d : g_devices) {
        d->tracker.event("closed by the haptic monitor (" + reason + ")");
        g_hd.disableDevice(d->handle);
    }
    writeState("released", reason);
    flock(lockFd, LOCK_UN);
    close(lockFd);
    return 0;
}
