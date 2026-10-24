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
#include "footpedal.h"

bool FootPedal::risingEdge(bool curr, bool& prev)
{
    bool edged = (!prev && curr);
    prev = curr;
    return edged;
}

bool FootPedal::isDrillOn()
{
    // Surgical hold-to-run: drill is active only while the pedal is depressed.
    if (m_isKeyboard) {
        return getButtonState(static_cast<int>(FootPedalKeyboardMap::DRILL_ON));
    }

    bool drillPedalHeld = getButtonState(static_cast<int>(FootPedalJoystickMap::DRILL_ON));
    // Analog pedal rest is about -1; require a clear press past 0.
    bool analogHeld = getPedalState(0) > 0.0;
    return drillPedalHeld || analogHeld;
}

bool FootPedal::isChangeBurrSizePressed()
{
    bool curr;
    if (m_isKeyboard) {
        curr = getButtonState(static_cast<int>(FootPedalKeyboardMap::CHANGE_BURR_SIZE));
    } else {
        curr = getButtonState(static_cast<int>(FootPedalJoystickMap::CHANGE_BURR_SIZE));
    }
    return risingEdge(curr, m_burrChangeBtnPrevState);
}

bool FootPedal::isChangeBurrStylePressed()
{
    bool curr;
    if (m_isKeyboard) {
        curr = getButtonState(static_cast<int>(FootPedalKeyboardMap::BURR_STYLE));
    } else {
        curr = getButtonState(static_cast<int>(FootPedalJoystickMap::BURR_STYLE));
    }
    return risingEdge(curr, m_burrStyleBtnPrevState);
}

bool FootPedal::isCamClutchPressed()
{
    return false;
}

bool FootPedal::isDeviceClutchPressed()
{
    return false;
}
