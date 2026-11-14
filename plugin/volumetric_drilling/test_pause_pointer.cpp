// g++ -std=c++17 plugin/volumetric_drilling/test_pause_pointer.cpp -o /tmp/test_pause_pointer && /tmp/test_pause_pointer
#include "pause_pointer.h"
#include <cassert>
#include <cstdio>

using namespace pause_pointer;

int main()
{
    Cursor c;
    c.line = 5.0;
    c.across = 0.5;
    // Up one line pitch: one line up the menu. Down: one line down.
    Cursor up = step(c, 0.0, kLinePitchM, 20);
    assert(lineUnder(up) == 4);
    Cursor down = step(c, 0.0, -kLinePitchM, 20);
    assert(lineUnder(down) == 6);
    // Right: across grows. Depth is not an input at all.
    Cursor right = step(c, 0.01, 0.0, 20);
    assert(right.across > c.across && right.line == c.line);
    // Kept on the panel, and back at once from the edge (like a mouse at the screen edge).
    Cursor top = step(c, 0.0, 1.0, 20);
    assert(top.line == 0.0);
    assert(lineUnder(step(top, 0.0, -kLinePitchM, 20)) == 1);
    Cursor far = step(c, 5.0, -5.0, 20);
    assert(far.across == 1.0 && far.line == 19.0);
    // A resting hand is still; a real move is not.
    assert(!moved(0.0001, 0.0));
    assert(moved(0.0, 0.001));
    std::puts("pause_pointer: all checks passed");
    return 0;
}
