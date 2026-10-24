//==============================================================================
/*
    Software License Agreement (BSD License)
    Copyright (c) 2019-2022, AMBF
    (https://github.com/WPI-AIM/ambf)

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

    * Neither the name of authors nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    \author    <amunawar@jhu.edu>
    \author    Adnan Munawar
*/
//==============================================================================
#ifndef FOOTPEDAL_H
#define FOOTPEDAL_H

#include "joystick.h"

// Ikkegol / PCsensor 3-pedal unit: left→a, middle→b, right→c (factory mapping).
// Also used for joystick button indices when a true js device is present.
enum class FootPedalJoystickMap{
    BURR_STYLE = 0,       // Left  (key a): diamond <-> cutting
    DRILL_ON = 1,         // Middle (key b): hold-to-run drill
    CHANGE_BURR_SIZE = 2  // Right (key c): cycle burr size
};

// Slots filled from KEY_A/B/C (or legacy F17/F18/F19) in joystick.cpp.
enum class FootPedalKeyboardMap{
    BURR_STYLE = 3,        // Left  = a
    DRILL_ON = 4,          // Middle = b
    CHANGE_BURR_SIZE = 5   // Right = c
};

class FootPedal: public JoyStick{
public:
    // True only while the drill pedal is held (surgical hold-to-run).
    bool isDrillOn();

    // Rising edge on far-right pedal (cycle 1/2/4/6 mm).
    bool isChangeBurrSizePressed();

    // Rising edge on left pedal (diamond <-> cutting).
    bool isChangeBurrStylePressed();

    bool isCamClutchPressed();

    bool isDeviceClutchPressed();

private:
    bool risingEdge(bool curr, bool& prev);

    bool m_burrChangeBtnPrevState = false;
    bool m_burrStyleBtnPrevState = false;
};

#endif
