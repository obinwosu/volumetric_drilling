// Read-only calibration-state monitor for Geomagic Touch devices.
// Forces are DISABLED. Never calls hdUpdateCalibration, so it only OBSERVES
// what the driver reports; it does not change calibration.
// Usage: calib_check <seconds> <device name> [<device name> ...]
#include <HD/hd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAXDEV 4
typedef struct { const char *name; HHD h; int calib; double pos[3]; } Dev;
static Dev devs[MAXDEV];
static int ndev;

static const char *cstr(int c) {
    switch (c) {
    case HD_CALIBRATION_OK: return "OK";
    case HD_CALIBRATION_NEEDS_UPDATE: return "NEEDS_UPDATE";
    case HD_CALIBRATION_NEEDS_MANUAL_INPUT: return "NEEDS_MANUAL_INPUT (dock in inkwell)";
    default: return "UNKNOWN";
    }
}

static HDCallbackCode HDCALLBACK sample(void *u) {
    (void)u;
    for (int i = 0; i < ndev; i++) {
        hdBeginFrame(devs[i].h);
        hdDisable(HD_FORCE_OUTPUT);
        devs[i].calib = hdCheckCalibration();
        hdGetDoublev(HD_CURRENT_POSITION, devs[i].pos);
        hdEndFrame(devs[i].h);
    }
    return HD_CALLBACK_DONE;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s seconds dev [dev...]\n", argv[0]); return 2; }
    double secs = atof(argv[1]);
    for (int a = 2; a < argc && ndev < MAXDEV; a++) {
        HHD h = hdInitDevice(argv[a]);
        HDErrorInfo e = hdGetError();
        if (HD_DEVICE_ERROR(e)) { fprintf(stderr, "INIT FAIL %s: %s\n", argv[a], hdGetErrorString(e.errorCode)); return 1; }
        hdMakeCurrentDevice(h); hdDisable(HD_FORCE_OUTPUT);
        HDint style = 0; hdGetIntegerv(HD_CALIBRATION_STYLE, &style);
        printf("opened %s (forces disabled), calibration style=%d%s\n", argv[a], style,
               style == HD_CALIBRATION_INKWELL ? " (inkwell)" : "");
        devs[ndev].name = argv[a]; devs[ndev].h = h; devs[ndev].calib = -1; ndev++;
    }
    hdStartScheduler();
    int last[MAXDEV]; for (int i = 0; i < MAXDEV; i++) last[i] = -2;
    for (double t = 0; t < secs; t += 0.25) {
        hdScheduleSynchronous(sample, NULL, HD_DEFAULT_SCHEDULER_PRIORITY);
        for (int i = 0; i < ndev; i++) {
            Dev *d = &devs[i];
            int changed = d->calib != last[i];
            if (changed || ((int)(t * 4) % 4 == 0))
                printf("t=%6.2fs %-15s calib=%-36s z=%8.2f y=%8.2f x=%8.2f%s\n", t, d->name, cstr(d->calib),
                       d->pos[2], d->pos[1], d->pos[0], changed && last[i] != -2 ? "   <-- CHANGED" : "");
            last[i] = d->calib;
        }
        fflush(stdout);
        usleep(250000);
    }
    hdStopScheduler();
    for (int i = 0; i < ndev; i++) hdDisableDevice(devs[i].h);
    return 0;
}
