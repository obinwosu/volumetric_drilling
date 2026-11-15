// Unattended force check: does the device actually move when force is sent?
// Stylus should be DOCKED in the inkwell, nobody holding it.
// Per open: settle 0.6 s, then for each of 3 directions send a 0.3 N pulse for
// 50 ms and record max displacement over the next 250 ms. Force is cut if
// speed > 100 mm/s. Logs safety switch / force output per open.
// Usage: pulse_test <opens> [device name]   (default device = drill)
#include <HD/hd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#define PULSE_N   0.8
#define PULSE_S   0.100
#define WATCH_S   0.250
#define VMAX      100.0
#define MOVED_MM  0.1

static volatile double g_force[3];
static volatile double g_pos[3];
static volatile int g_cut, g_ss, g_err;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static HDCallbackCode HDCALLBACK servo(void *u) {
    HHD h = *(HHD *)u;
    hdBeginFrame(h);
    double p[3], v[3]; HDboolean ss = 0;
    hdGetDoublev(HD_CURRENT_POSITION, p);
    hdGetDoublev(HD_CURRENT_VELOCITY, v);
    hdGetBooleanv(HD_CURRENT_SAFETY_SWITCH, &ss);
    g_ss = ss;
    if (sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) > VMAX) g_cut = 1;
    double f[3] = {0, 0, 0};
    if (!g_cut) { f[0] = g_force[0]; f[1] = g_force[1]; f[2] = g_force[2]; }
    hdSetDoublev(HD_CURRENT_FORCE, f);
    hdEndFrame(h);
    HDErrorInfo e = hdGetError(); if (HD_DEVICE_ERROR(e)) g_err = e.errorCode;
    g_pos[0] = p[0]; g_pos[1] = p[1]; g_pos[2] = p[2];
    return HD_CALLBACK_CONTINUE;
}

static double pulse(int axis, double sign) {
    double base[3] = {g_pos[0], g_pos[1], g_pos[2]}, maxd = 0;
    g_force[0] = g_force[1] = g_force[2] = 0;
    g_force[axis] = sign * PULSE_N;
    double t0 = now();
    while (now() - t0 < WATCH_S) {
        if (now() - t0 >= PULSE_S) g_force[axis] = 0;
        double dx = g_pos[0]-base[0], dy = g_pos[1]-base[1], dz = g_pos[2]-base[2];
        double d = sqrt(dx*dx+dy*dy+dz*dz); if (d > maxd) maxd = d;
        usleep(1000);
    }
    g_force[axis] = 0;
    usleep(300000);   // let it settle back
    return maxd;
}

int main(int argc, char **argv) {
    int opens = argc > 1 ? atoi(argv[1]) : 10;
    const char *name = (argc > 2 && strcmp(argv[2], "-") != 0) ? argv[2] : HD_DEFAULT_DEVICE;
    int warm = argc > 3 ? atoi(argv[3]) : 0;      // quick open/close cycles first: no scheduler, no force
    double tw = now();
    for (int w = 0; w < warm; w++) {
        HHD h = hdInitDevice(name);
        HDErrorInfo e = hdGetError();
        if (HD_DEVICE_ERROR(e)) { printf("warm-up %d: INIT FAIL %s\n", w + 1, hdGetErrorString(e.errorCode)); continue; }
        hdDisableDevice(h);
    }
    if (warm) printf("warm-up: %d quick open/close cycles in %.2f s\n", warm, now() - tw);
    int moved_n = 0, fail_n = 0;
    printf("pulse test: %d opens of %s, %.1f N x %.0f ms pulses, cut at %.0f mm/s\n",
           opens, name ? name : "(default)", PULSE_N, PULSE_S * 1000, VMAX);
    for (int r = 1; r <= opens; r++) {
        g_cut = 0; g_err = 0; g_force[0] = g_force[1] = g_force[2] = 0;
        HHD h = hdInitDevice(name);
        HDErrorInfo e = hdGetError();
        if (HD_DEVICE_ERROR(e)) { printf("open %2d: INIT FAIL %s\n", r, hdGetErrorString(e.errorCode)); fail_n++; fflush(stdout); sleep(2); continue; }
        hdMakeCurrentDevice(h);
        hdEnable(HD_FORCE_OUTPUT);
        HDSchedulerHandle cb = hdScheduleAsynchronous(servo, &h, HD_MAX_SCHEDULER_PRIORITY);
        hdStartScheduler();
        usleep(600000);
        g_cut = 0;   // ignore the start-up position jump; arm the speed cut only now
        int ss0 = g_ss; HDboolean fo = hdIsEnabled(HD_FORCE_OUTPUT);
        double dy = pulse(1, +1), dz = pulse(2, +1), dzn = pulse(2, -1);
        double best = fmax(dy, fmax(dz, dzn));
        int moved = best > MOVED_MM; moved_n += moved;
        printf("open %2d: %-8s max move %.2f mm (+y %.2f, +z %.2f, -z %.2f)  safety_switch=%d force_output=%s  HD err=%s%s\n",
               r, moved ? "MOVED" : "NO MOVE", best, dy, dz, dzn, ss0, fo ? "on" : "OFF",
               g_err ? hdGetErrorString(g_err) : "none", g_cut ? "  (speed cut)" : "");
        fflush(stdout);
        hdStopScheduler(); hdUnschedule(cb); hdDisableDevice(h);
        sleep(2);
    }
    printf("SUMMARY: %d/%d opens moved, %d init failures\n", moved_n, opens, fail_n);
    return 0;
}
