// Tests a built libhdPhantom.so exactly as CHAI3D uses it: dlopen (runs the
// library's init + warm-up), hdPhantomOpen(0), start servo, one gentle pulse
// via hdPhantomSetForce, measure motion. Stylus must rest free (not docked).
// Usage: bridge_test <path to libhdPhantom.so>
#include <dlfcn.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

typedef int  (*fOpen)(int);
typedef int  (*fPos)(int, double*, double*, double*);
typedef int  (*fForce)(int, double*, double*, double*);
typedef void (*fVoid)(void);
typedef int  (*fNum)(void);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s libhdPhantom.so\n", argv[0]); return 2; }
    dlopen("libHD.so", RTLD_NOW | RTLD_GLOBAL);
    double t0 = now();
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { printf("RESULT dlopen-fail %s\n", dlerror()); return 1; }
    double tload = now() - t0;
    fNum num = (fNum)dlsym(lib, "hdPhantomGetNumDevices");
    fOpen open = (fOpen)dlsym(lib, "hdPhantomOpen"), close = (fOpen)dlsym(lib, "hdPhantomClose");
    fPos pos = (fPos)dlsym(lib, "hdPhantomGetPosition"), vel = (fPos)dlsym(lib, "hdPhantomGetLinearVelocity");
    fForce force = (fForce)dlsym(lib, "hdPhantomSetForce");
    fVoid start = (fVoid)dlsym(lib, "hdPhantomStartServo"), stop = (fVoid)dlsym(lib, "hdPhantomStopServo");
    int n = num();
    if (n < 1) { printf("RESULT no-device load=%.2fs\n", tload); return 1; }
    open(0); start();
    usleep(600000);
    double x0, y0, z0, x, y, z, vx, vy, vz, maxd = 0, fx = 0, fy = 0.8, fz = 0, zero = 0;
    pos(0, &x0, &y0, &z0);
    double tp = now(); int cut = 0;
    while (now() - tp < 0.35) {
        vel(0, &vx, &vy, &vz);
        if (sqrt(vx*vx + vy*vy + vz*vz) > 0.1) cut = 1;            // 100 mm/s
        if (!cut && now() - tp < 0.10) force(0, &fx, &fy, &fz); else force(0, &zero, &zero, &zero);
        pos(0, &x, &y, &z);
        double d = 1000 * sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
        if (d > maxd) maxd = d;
        usleep(1000);
    }
    force(0, &zero, &zero, &zero);
    usleep(300000);
    printf("RESULT %s move=%.2fmm devices=%d load+warmup=%.2fs\n", maxd > 0.1 ? "MOVED" : "NO_MOVE", maxd, n, tload);
    fflush(stdout);
    for (int i = 0; i < n; i++) close(i);
    stop();
    usleep(100000);
    return 0;
}
