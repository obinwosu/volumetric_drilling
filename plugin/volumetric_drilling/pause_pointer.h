#ifndef PAUSE_POINTER_H
#define PAUSE_POINTER_H
// The stylus as a mouse in the pause menu.
//
// While the menu is open the burr stays frozen and the stylus drives a flat
// cursor on the menu panel: up / down / left / right move it, toward and away
// from you is ignored. The action button clicks whatever row is under the
// cursor. The pedals keep working as before. No forces: the Touch moves freely.
//
// The burr does not jump when the menu closes: drill_manager moves the burr by
// the stylus's velocity, and that is not applied while the simulator is paused,
// so hand travel in the menu is dropped (like lifting a mouse).
//
// The cursor is in panel units: `line` is a text line of the pause HUD (0 = the
// title, fractions in between) and `across` is 0..1 of the panel's text width.
// The plugin hit-tests rows by line; the headset HUD draws the cursor there.
//
// Plain doubles, no CHAI3D, so it can be tested on its own (test_pause_pointer.cpp).
// Touch device frame (CHAI3D): x toward the user, y right, z up.
#include <algorithm>
#include <cmath>

namespace pause_pointer {

// Stylus travel per text line, and across the whole panel width.
constexpr double kLinePitchM = 0.008;
constexpr double kPanelWidthM = 0.12;
// Flip a sign here if a rig's stylus moves the cursor the wrong way.
constexpr double kRightSign = 1.0;  // +y (right) moves the cursor right
constexpr double kUpSign = 1.0;     // +z (up) moves the cursor up
// Slower than this between two frames counts as still (hand tremor, noise),
// so a resting stylus does not take the highlight back from the pedals.
constexpr double kStillM = 0.0002;

struct Cursor {
    double line = 0.0;
    double across = 0.3;
};

// Move the cursor by one frame's stylus motion (dy right, dz up), kept on the panel.
inline Cursor step(Cursor c, double dy, double dz, int lines)
{
    c.line -= kUpSign * dz / kLinePitchM;
    c.across += kRightSign * dy / kPanelWidthM;
    c.line = std::clamp(c.line, 0.0, std::max(0, lines - 1) * 1.0);
    c.across = std::clamp(c.across, 0.0, 1.0);
    return c;
}

// The text line under the cursor.
inline int lineUnder(const Cursor& c)
{
    return static_cast<int>(std::lround(c.line));
}

inline bool moved(double dy, double dz)
{
    return std::hypot(dy, dz) > kStillM;
}

}  // namespace pause_pointer

#endif
