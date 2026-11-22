// g++ -std=c++17 -I plugin/volumetric_drilling scripts/haptics_tuner/test_haptic_tuning.cpp -o /tmp/t && /tmp/t
#include "haptic_tuning.h"
#include <cassert>
#include <cstdio>
#include <limits>
using namespace haptic_tuning;

static double norm(const double v[3]) { return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

int main()
{
    Params p;
    p.cutGate = 1; p.cutStartN = 0.5; p.cutFullN = 1.5; p.cutCurve = 1; p.clamp();
    assert(cutFraction(p, 0.5) == 0.0);
    assert(near(cutFraction(p, 1.0), 0.5, 1e-9));
    assert(cutFraction(p, 3.0) == 1.0);

    // Below the threshold nothing is cut, however long the burr presses.
    CutBudget budget;
    for (int i = 0; i < 5000; i++) assert(budget.allow(p, 0.49, 4.0, 0.0156, 0.001, 100) == 0);

    // At full force the mean rate is the set mm3/s.
    p.rateMm3[2] = 40.0;
    budget.idle();
    long removed = 0;
    for (int i = 0; i < 10000; i++) { int n = budget.allow(p, 2.0, 4.0, 0.015625, 0.001, 100); budget.spend(n); removed += n; }
    assert(near(removed * 0.015625 / 10.0, 40.0, 0.2));

    // Limited by the voxels in contact and the per-tick cap.
    budget.idle();
    p.rateMm3[2] = kMaxCutRateMm3;
    assert(budget.allow(p, 3.0, 4.0, 1e-4, 0.004, 3) == 3);
    assert(budget.allow(p, 3.0, 4.0, 1e-4, 0.004, 1000) <= kMaxVoxelsPerTick);

    // Bad values cannot exceed the limits.
    Params bad;
    bad.humAmpN = 99; bad.humHz = 1e6; bad.stiffnessMult = 50; bad.cutFullN = -1;
    bad.cutStartN = std::numeric_limits<double>::quiet_NaN();
    bad.clamp();
    assert(bad.humAmpN == kMaxHumN && bad.humHz == kMaxHumHz && bad.stiffnessMult == kMaxStiffnessMult);
    assert(bad.cutFullN > bad.cutStartN);

    // Hum: free-air amplitude, cap under load, spin-down, guard trip.
    const double shaft[3] = {0, 0, 1};
    double out[3];
    Params h; h.humAmpN = 0.3; h.humHz = 200; h.humDirection = HUM_RADIAL; h.clamp();
    Hum hum;
    for (int i = 0; i < 20; i++) {
        hum.generate(h, 0.001, true, false, shaft, 0.0, out);
        assert(near(norm(out), 0.3, 1e-9));            // a turning force has constant size
        assert(near(out[2], 0.0, 1e-9));               // and stays across the shaft
    }
    h.humAmpN = 0.6; h.humLoadAmp = 2.0;
    hum.generate(h, 0.001, true, false, shaft, 3.3, out);
    assert(norm(out) <= kMaxHumN + 1e-9);
    hum.generate(h, 0.001, true, true, shaft, 0.0, out);
    assert(norm(out) == 0.0);

    Params spin; spin.humAmpN = 0.2; spin.humHz = 100; spin.humDirection = HUM_RADIAL; spin.humSpinUpS = 0.1; spin.humSpinDownS = 0.1; spin.clamp();
    Hum ramp;
    ramp.generate(spin, 0.001, true, false, shaft, 0, out);
    assert(near(ramp.envelope(), 0.01, 1e-9));
    for (int i = 0; i < 200; i++) ramp.generate(spin, 0.001, true, false, shaft, 0, out);
    assert(ramp.envelope() == 1.0);
    for (int i = 0; i < 200; i++) ramp.generate(spin, 0.001, false, false, shaft, 0, out);
    assert(ramp.envelope() == 0.0 && norm(out) == 0.0);

    // Defaults are the stock hum: 0.08 N per axis at 500 rad/s.
    Params stock; stock.clamp();
    assert(stock.cutGate == 0.0 && stock.stiffnessMult == 1.0);
    assert(near(stock.humAmpN / std::sqrt(3.0), 0.08, 1e-3) && near(stock.humHz * 2 * M_PI, 500, 0.01));
    // No force before the motors pass, or while the burr starts inside the bone.
    ForceGate gate;
    for (int i = 0; i < 2000; i++) assert(!gate.update(false, 0.0, 0.001));
    for (int i = 0; i < 2000; i++) assert(!gate.update(true, 3.3, 0.001));
    for (int i = 0; i < 240; i++) assert(!gate.update(true, 0.0, 0.001));
    for (int i = 0; i < 20; i++) gate.update(true, 0.0, 0.001);
    assert(gate.open() && gate.update(true, 3.3, 0.001));
    assert(!gate.update(false, 0.0, 0.001));

    // A short hard push passes; a held one is eased to the continuous rating.
    SustainedLimit shortPush;
    for (int i = 0; i < 300; i++) assert(shortPush.update(3.3, 0.001) == 1.0);
    SustainedLimit held;
    double scale = 1.0;
    for (int i = 0; i < 3000; i++) scale = held.update(3.3, 0.001);
    assert(held.limiting() && near(3.3 * scale, kContinuousN, 0.01));
    puts("haptic_tuning: all checks passed");
}
