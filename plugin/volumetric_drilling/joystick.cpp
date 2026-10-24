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

#include "joystick.h"
#include <string>
#include <iostream>
#include <linux/input.h>
#include <cerrno>
#include <climits>
#include <cstring>

using namespace std;

JoyState::JoyState(){
    m_buttons.resize(16);
    m_axes.resize(8);

    for (int i = 0 ; i < m_buttons.size(); i++){
        m_buttons[i] = false;
    }

    for (int i = 0 ; i < m_axes.size(); i++){
        m_axes[i] = -1.0;
    }
}

void JoyState::print(){
    for (int i = 0 ; i < m_buttons.size(); i++){
        printf("\tButton %u %s\n", i, m_buttons[i] ? "pressed" : "released");

    }

    for (int i = 0 ; i < m_axes.size(); i++){
    printf("\tAxis at (%6d, %6f)\n", i, m_axes[i]);

    }
}

void JoyState::reset(){
    for (int i = 0 ; i < m_buttons.size(); i++){
        m_buttons[i] = false;
    }

    for (int i = 0 ; i < m_axes.size(); i++){
        m_axes[i] = -1.0;
    }
}

///
/// \brief JoyStick::JoyStick
///
JoyStick::JoyStick(){
}


///
/// \brief JoyStick::~JoyStick
///
JoyStick::~JoyStick(){
    close(m_js);
}

///
/// \brief JoyStick::findFootSwitchDevice
/// \return device path if found, empty string otherwise
///
std::string JoyStick::findFootSwitchDevice() {
    std::string devicePath;
    
    std::cout << "Looking for FootSwitch in /dev/input/by-id..." << std::endl;
    
    DIR *dir = opendir("/dev/input/by-id");
    if (dir == nullptr) {
        perror("Could not open /dev/input/by-id");
        return "";
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string entryName(entry->d_name);
        std::cout << "Checking: " << entryName << std::endl;
        
        // Look for the FootSwitch keyboard event device
        if (entryName.find("PCsensor_FootSwitch") != std::string::npos && 
            entryName.find("event-kbd") != std::string::npos) {
            
            std::string fullPath = std::string("/dev/input/by-id/") + entryName;
            std::cout << "*** FOUND FootSwitch: " << fullPath << std::endl;
            
            // Resolve the symlink to get the actual device path
            char actualPath[PATH_MAX];
            ssize_t len = readlink(fullPath.c_str(), actualPath, sizeof(actualPath)-1);
            if (len != -1) {
                actualPath[len] = '\0';
                
                // If it's a relative path, make it absolute relative to by-id directory
                if (actualPath[0] != '/') {
                    std::string relativePath = std::string("/dev/input/by-id/") + actualPath;
                    // Canonicalize the path to resolve ".."
                    char canonicalPath[PATH_MAX];
                    if (realpath(relativePath.c_str(), canonicalPath)) {
                        devicePath = std::string(canonicalPath);
                    } else {
                        // Fallback: try to resolve manually
                        devicePath = "/dev/input/" + std::string(actualPath).substr(3); // remove "../"
                    }
                } else {
                    devicePath = std::string(actualPath);
                }
            } else {
                // If readlink fails, use the symlink path directly
                devicePath = fullPath;
            }
            
            std::cout << "Resolved to: " << devicePath << std::endl;
            break;
        }
    }
    closedir(dir);
    
    if (devicePath.empty()) {
        std::cout << "Could not find FootSwitch device in /dev/input/by-id" << std::endl;
    }
    
    return devicePath;
}

///
/// \brief JoyStick::init
/// \param dev_name
/// \return
///
int JoyStick::init(string dev_name){
    auto isKeyboardPath = [](const string& path) -> bool {
        return path.find("event-kbd") != string::npos
            || path.find("FootSwitch") != string::npos
            || path.find("/event") != string::npos;
    };
    auto printAccessHint = [](){
        std::cout << "ERROR! No permission to read the footpedal. Drill pedals need /dev/input access.\n"
                  << "       Fix with:\n"
                  << "         sudo cp scripts/udev/99-volumetric-drilling-input.rules /etc/udev/rules.d/\n"
                  << "         sudo usermod -aG input \"$USER\"\n"
                  << "         sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                  << "       then log out and back in. Falling back to X11 key polling." << std::endl;
    };

    m_js = open(dev_name.c_str(), O_RDONLY | O_NONBLOCK);

    if (m_js != -1){
        m_isKeyboard = isKeyboardPath(dev_name);
        std::cout << "Successfully opened "
                  << (m_isKeyboard ? "keyboard footpedal" : "joystick")
                  << " device: " << dev_name << std::endl;
        return 1;
    }

    const int openErrno = errno;
    std::cout << "Could not open " << dev_name
              << " (" << strerror(openErrno) << ")" << std::endl;
    if (openErrno == EACCES){
        printAccessHint();
    }

    // If configured path failed, look for PCsensor FootSwitch keyboard HID.
    std::string footSwitchDevice = findFootSwitchDevice();
    if (!footSwitchDevice.empty() && footSwitchDevice != dev_name) {
        m_js = open(footSwitchDevice.c_str(), O_RDONLY | O_NONBLOCK);
        if (m_js != -1) {
            std::cout << "Successfully opened FootSwitch device: " << footSwitchDevice << std::endl;
            m_isKeyboard = true;
            return 1;
        }
        const int fsErrno = errno;
        std::cout << "Failed to open FootSwitch device " << footSwitchDevice
                  << " (error: " << strerror(fsErrno) << ")" << std::endl;
        if (fsErrno == EACCES){
            printAccessHint();
        }
    }

    // Keep analog at released (-1) so the drill stays off until a pedal is held.
    std::cout << "Could not open a footpedal device; X11 key fallback may still work." << std::endl;
    m_state.m_axes[0] = -1.0;
    return -1;
}

bool JoyStick::isAvailable(){
    return m_js == -1 ? false : true;
}


///
/// \brief JoyStick::readEvent
/// \param fd
/// \param event
/// \return
///
int JoyStick::readEvent(int fd, struct js_event *event){
    ssize_t bytes;
    bytes = read(fd, event, sizeof(*event));

    if (bytes == (ssize_t)sizeof(*event))
        return 0;

    /* Error, could not read full event. */
    return -1;
}

///
/// \brief JoyStick::readKeyboardEvent
/// \param fd
/// \param event
/// \return
///
int JoyStick::readKeyboardEvent(int fd, struct input_event *event){
    ssize_t bytes;
    bytes = read(fd, event, sizeof(*event));

    if (bytes == sizeof(*event))
        return 0;

    /* Error, could not read full event. */
    return -1;
}


///
/// \brief JoyStick::getButtonState
/// \param button_index
/// \return
///
bool JoyStick::getButtonState(int button_index){
    if (button_index < 0 || button_index >= (int)m_state.m_buttons.size()){
        return false;
    }
    return m_state.m_buttons[button_index];
}


///
/// \brief JoyStick::getPedalState
/// \param pedal_index
/// \return
///
double JoyStick::getPedalState(int pedal_index){
    if (pedal_index < 0 || pedal_index >= (int)m_state.m_axes.size()){
        return -1.0;
    }
    return m_state.m_axes[pedal_index];
}

void JoyStick::poll()
{
    if (m_isKeyboard) {
        // Drain the keyboard event queue (same idea as joystick).
        for (int n = 0; n < 32; n++){
            if (readKeyboardEvent(m_js, &m_keyboard_event) != 0) {
                break;
            }
            if (m_keyboard_event.type != EV_KEY) {
                continue;
            }
            int button_index = -1;
            // Ikkegol factory map: left=a, middle=b, right=c.
            if (m_keyboard_event.code == KEY_A) {
                button_index = static_cast<int>(3); // FootPedalKeyboardMap::BURR_STYLE
            }
            else if (m_keyboard_event.code == KEY_B) {
                button_index = static_cast<int>(4); // FootPedalKeyboardMap::DRILL_ON
            }
            else if (m_keyboard_event.code == KEY_C) {
                button_index = static_cast<int>(5); // FootPedalKeyboardMap::CHANGE_BURR_SIZE
            }
            if (button_index != -1 && button_index < (int)m_state.m_buttons.size()) {
                m_state.m_buttons[button_index] = m_keyboard_event.value;
            }
        }
    }
    else {
        // Handle joystick events (drain the queue; include INIT so rest pose is known).
        for (int n = 0; n < 32; n++){
            if (readEvent(m_js, &m_event) != 0){
                break;
            }
            const unsigned char evType = m_event.type & ~JS_EVENT_INIT;
            switch (evType) {
            case JS_EVENT_BUTTON:
                if (m_event.number < m_state.m_buttons.size()) {
                    m_state.m_buttons[m_event.number] = m_event.value;
                }
                if ((m_event.type & JS_EVENT_INIT) == 0){
                    printf("Button %u %s\n", m_event.number, m_event.value ? "pressed" : "released");
                }
                break;
            case JS_EVENT_AXIS:
                if (m_event.number < m_state.m_axes.size()) {
                    m_state.m_axes[m_event.number] = double (m_event.value / 32768.0);
                }
                if ((m_event.type & JS_EVENT_INIT) == 0){
                    printf("Axis at (%6d, %6f)\n", m_event.number, double (m_event.value / 32768.0));
                }
                break;
            default:
                break;
            }
        }
    }
}
