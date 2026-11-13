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

// To silence warnings on MacOS
#define GL_SILENCE_DEPRECATION
#include <afFramework.h>
#include "footpedal.h"
#include "camera_panel_manager.h"
#include "wave_generator.h"
#include "haptic_safety.h"
#include "gaze_marker_controller_jagged.h" // change bewteen gaze_marker_controller_smooth vs. gaze_marker_controller_jagged
#include "drill_manager.h"
#include "video_recording_controller.h"

#include <vector>

using namespace std;
using namespace ambf;


class afVolmetricDrillingPlugin: public afSimulatorPlugin{
public:
    afVolmetricDrillingPlugin();
    virtual int init(int argc, char** argv, const afWorldPtr a_afWorld) override;
    virtual void keyboardUpdate(GLFWwindow* a_window, int a_key, int a_scancode, int a_action, int a_mods) override;
    virtual void mouseBtnsUpdate(GLFWwindow* a_window, int a_button, int a_action, int a_modes) override;
    virtual void mousePosUpdate(GLFWwindow* a_window, double x_pos, double y_pos) override;
    virtual void mouseScrollUpdate(GLFWwindow* a_window, double x_pos, double y_pos) override;
    virtual void graphicsUpdate() override;
    virtual void physicsUpdate(double dt) override;
    virtual void reset() override;
    virtual bool close() override;

protected:
    void sliceVolume(int axisIdx, double delta);

    void makeVRWindowFullscreen(afCameraPtr vrCam, int monitor_number=-1);

    void updateButtons();

    void initializeLabels();

    afCameraPtr findAndAppendCamera(string cam_name);

    // Roll main_camera about the camera-to-volume axis. Volume pose is unchanged.
    void setCameraRoll(double clockwiseDeg);
    void applyCameraRoll(double clockwiseDeg);

    // Move main_camera along the current look axis. Positive = dolly in (keep orientation).
    void dollyCamera(double distanceTowardVolume);
    // Narrow (factor < 1) or widen the main_camera field of view, like microscope zoom.
    // The camera does not move, so the scope light on the field is unchanged.
    void zoomCamera(double factor);
    void writeRtxFov(double fovRad);

    void applyVolumeLightingUniforms();
    // Empty-space color behind the microscope beam. 0 = black, 20 = white.
    void applyRoomGray();
    void nudgeRoomGray(int deltaSteps);
    // Optional RTX window. Writes a status file and, on request, the current
    // occupancy. It does not change drilling.
    void publishRtxStatus(double now);
    void publishDrillVisualState();
    void publishSuctionVisualState();
    void serviceRtxCamera();
    void serviceRtxPauseMenu();
    void applySimPause(bool paused);
    bool takeRtxCameraFile(double& yawDeg, double& pitchDeg, double& panRight, double& panUp, double& dolly);
    void panCamera(double rightAmount, double upAmount);

    struct CanonicalView {
        string id;
        string label;
        string laterality; // "left" or "right"
        bool valid = false;
        cTransform T_rel;
    };

    struct PauseViewEntry {
        string id;      // "home" or CanonicalView.id
        string label;
        bool valid = false;
        bool isHome = false;
    };

    string canonicalViewsPath();
    void loadCanonicalViews();
    void saveCanonicalViews();
    CanonicalView* findCanonicalView(const string& id);
    void saveCurrentIntoView(const string& id);
    bool applyNamedView(const string& id);
    void applyHomeView();
    void applyStartupView();
    void nudgeCameraRoll(double clockwiseDeg);
    // Yaw-orbit around the current look-at, about camera up (keeps roll).
    void nudgeCameraOrbitYaw(double rightDeg);
    // Pitch-orbit around the current look-at, about camera right.
    void nudgeCameraOrbitPitch(double upDeg);
    // AMBF scroll always dollies first; reverse that when we consume the wheel.
    void undoAmbfWheelDolly(GLFWwindow* a_window, double y_pos);
    vector<PauseViewEntry> buildPauseViewList() const;
    void applyHighlightedPauseView();
    void saveHighlightedPauseView();
    // Apply saved or rough view for anatomy laterality (startup / F3-F4).
    void applyLateralityStartupView();
    string findAnatomyAdfPathFromArgs(int argc, char** argv) const;
    string readLateralityFromAdf(const string& adfPath) const;
    void showNotice(const string& text, double durationSec = 4.0);
    // Desktop + VR pause / session menu (Ctrl+P or hold the left pedal).
    void applyStereoCenteredHud();
    void ensurePauseMenuLabel();
    void setVrHudVisible(bool visible);
    void refreshVrHudText();
    // Pause menu text for the RTX headset view (/tmp/fivrs-rtx/pause_hud.txt); empty removes it.
    void writeRtxPauseHud(const string& text);
    string m_rtxPauseHudText;
    string controlHelpText(bool paused) const;
    void cyclePauseViewHighlight(int delta);
    // Pause menu: views, then Burr type and Burr size, each opening a list.
    enum class PauseSubmenu { NONE, VIEWS, BURR_TYPE, BURR_SIZE };
    PauseSubmenu m_pauseSubmenu = PauseSubmenu::NONE;
    int m_pauseSubIndex = 0;
    // "Close pause menu and return to simulator?" Holding the left pedal in the menu
    // asks this; the left pedal toggles Yes / No and the middle pedal selects.
    bool m_pauseConfirmClose = false;
    bool m_pauseConfirmYes = true;
    int pauseViewCount();
    int pauseRowCount();
    int pauseSubmenuCount() const;
    vector<int> burrSizeOrder() const;
    void clampPauseIndex();
    void pauseMenuSelect();
    void pauseMenuBack();
    string pauseMenuListText();
    // True if Ctrl is held system-wide (works even when study_gui / VR has focus).
    bool isCtrlHeld() const;
    // True if Shift is held system-wide (works even when study_gui / VR has focus).
    bool isShiftHeld() const;
    // True if Alt is held — used as cam-clutch when an HMD eyepiece is loaded.
    bool isAltHeld() const;
    // True if digit "1" is held (SayoDevice macro key → camera clutch).
    bool isDigit1Held() const;
    // X11 keymap helpers (work without /dev/input permission).
    bool isXKeycodeHeld(unsigned int keycode) const;
    // True if letter key is held system-wide (Ikkegol a/b/c pedals).
    bool isLetterKeyHeld(int glfwKey) const;
    void requestDrillReposition();
    void requestSuctionReposition();

private:

    cVoxelObject* m_voxelObj;

    int m_renderingMode = 0;

    double m_opticalDensity;

    cMutex m_mutexVoxel;

    cCollisionAABBBox m_volumeUpdate;

    cColorb m_zeroColor;

    bool m_flagStart = true;

    int m_counter = 0;

    cGenericObject* m_selectedObject = NULL;

    bool m_flagMarkVolumeForUpdate = false;

    afVolumePtr m_volumeObject;

    // camera to render the world
    afCameraPtr m_mainCamera, m_cameraL, m_cameraR, m_stereoCamera;
    // Present when ADF 14 (OpenVR) or ADF 15 (OpenXR eyepiece) is loaded.
    afCameraPtr m_openvrCamera = nullptr;
    afCameraPtr m_openxrEyepieceCamera = nullptr;
    typedef void (*HmdCamClutchFn)(int);
    typedef void (*OpenxrVoidFn)(void);
    typedef void (*OpenxrIntFn)(int);
    HmdCamClutchFn m_openvrSetCamClutch = nullptr;
    HmdCamClutchFn m_openxrSetCamClutch = nullptr;
    OpenxrVoidFn m_openxrRecenterToHead = nullptr;
    OpenxrVoidFn m_openxrRecenterLap = nullptr;
    OpenxrIntFn m_openxrSetPlaceDrag = nullptr;
    void ensureOpenxrPlacementFns();

    map<string, afCameraPtr> m_cameras;

    // warning pop-up label
    cLabel* m_warningLabel;

    // color property of bone
    cColorb m_boneColor;

    // get color of voxels at (x,y,z)
    cColorb m_storedColor;

    bool m_enableVolumeSmoothing = false;

    int m_volumeSmoothingLevel = 2;

    // 0 = no distance falloff. 1 = mild clamped falloff. See dvr/shader.fs.
    int m_attenuationMode = 0;
    double m_attConstant = 1.0;
    double m_attLinear = 0.02;
    double m_attQuadratic = 0.0;
    double m_attClampMin = 0.85;
    double m_attClampMax = 1.0;
    bool m_revisedSpecular = true;
    double m_specularStrength = 0.45;
    double m_shininess = 48.0;
    // Half-angle of the microscope beam. 0.15 rad is about 8.6 degrees.
    double m_spotCutoffRad = 0.15;
    // 1 = smooth fade to the cone edge. Higher pulls brightness toward the axis.
    double m_spotFocus = 2.0;
    // Room gray in 5% steps. 3 = 15%, the surreal simulator room (0.14–0.15).
    int m_roomGrayStep = 3;
    /// FIVRS_LOOK=upstream (graphics study): AMBF's look before the FIVRS changes.
    bool m_upstreamLook = false;
    std::vector<cBackground*> m_roomBackdrops;

    cLabel* m_volumeSmoothingLabel;

    cVector3d m_maxVolCorner, m_minVolCorner;

    cVector3d m_maxTexCoord, m_minTexCoord;

    cVector3d m_textureCoordScale; // Scale between volume corners extent and texture coordinates extent

    DrillManager m_drillManager;

    FootPedal m_footpedal;

    // Rising-edge trackers for X11 FootSwitch fallback (linux KEY_F17/18/19 + 8).
    bool m_xFootStylePrev = false;
    bool m_xFootSizePrev = false;
    // With the microscope button held, the left / middle pedals zoom in / out.
    bool m_pedalDrillPrev = false;
    bool m_pedalDrillSuppress = false;  // drill pedal held through the pause menu: wait for release
    // Left pedal: a tap moves on (tutorial) or down (pause menu); a long hold opens the menu.
    double m_leftPedalDownTime = -1.0;
    bool m_leftPedalLongFired = false;
    bool m_leftPedalZoomed = false;  // the microscope button was held during this press

    // Study mode: /tmp/fivrs-rtx/study_drill_lock stops the drill (timed trial over).
    bool m_studyDrillLock = false;
    double m_studyLockPollTime = -1.0;
    // Counters for the study tutorial, written to drill_state.json.
    long m_zoomSteps = 0;
    double m_zoomTravel = 0.0;  // total zoom applied, in doublings
    double m_zoomInTravel = 0.0;
    double m_zoomOutTravel = 0.0;
    int m_pedalAdvanceCount = 0;  // left-pedal taps outside the menu (study tutorial)
    int m_tutorialSkipCount = 0;  // right-arrow presses outside the menu (study tutorial)
    bool m_rightArrowPrev = false;
    long m_drillRecenterCount = 0;

    WaveGenerator m_waveGenerator;

    GazeMarkerController m_gazeMarkerController;

    CameraPanelManager m_panelManager;

    VideoRecordingController m_videoRecordingController; // Adapting ambf video recording plugin
    double m_timeSinceLastCalibration;
    bool m_isRecording = false;
    
    bool m_isCountdownActive; // For pulling back on the camera during recalibration
    double m_countdownTimer;
    cTransform m_originalCameraTransform;
    cVector3d m_volumeLookAt;
    cVector3d m_cameraHomeUp;
    double m_activeCameraRollDeg = 0.0;
    bool m_customViewActive = false;
    bool m_arcballMouseDown = false;
    bool m_rightMouseDown = false;
    cMatrix3d m_rightDragRot;
    cVector3d m_preservedUp;
    vector<CanonicalView> m_canonicalViews;
    string m_startupViewId;
    int m_pauseViewIndex = 0;
    string m_anatomyLaterality; // "left", "right", or empty
    string m_sessionUser;       // from study GUI --session_user
    string m_sessionAnatomy;    // from study GUI --session_anatomy
    cLabel* m_cameraPresetLabel;
    bool m_helpVisible = true;
    // Last totals read from the RTX camera-nudge file. The difference is applied.
    unsigned long long m_cameraNudgeSeq = 0;
    unsigned long long m_rtxPauseSeq = 0;
    double m_cameraNudgeYaw = 0.0;
    double m_cameraNudgePitch = 0.0;
    double m_cameraNudgePanRight = 0.0;
    double m_cameraNudgePanUp = 0.0;
    double m_cameraNudgeDolly = 0.0;
    unsigned long long m_rtxFovSeq = 0;
    bool m_rtxFovWritten = false;
    cLabel* m_calibrationWarningLabel;
    cLabel* m_noticeLabel;
    cLabel* m_vrHudLabel = nullptr;
    double m_noticeTimer = 0.0;
    bool m_vrHudVisible = false;
    bool m_simPaused = false;
    double m_wallTimeAtPause = -1.0;
    bool m_btn2Prev = false;
    double m_btn2LastClickTime = -1.0;
    bool m_suctionBtn2Prev = false;
    double m_suctionBtn2LastClickTime = -1.0;
    bool m_rtxView = false;
    bool m_rtxWindowSized = false;
    double m_rtxStatusTimer = 0.0;
    // Last contact force, held across the brief gap after a voxel is deleted.
    cVector3d m_heldHapticForce = cVector3d(0.0, 0.0, 0.0);
    double m_hapticHoldSeconds = 0.0;
    // Rate limit + oscillation cutoff on the drill force (haptic_safety.h).
    HapticSafety m_hapticSafety;
    cVector3d m_heldSuctionForce = cVector3d(0.0, 0.0, 0.0);
    double m_suctionHoldSeconds = 0.0;

    // RTX deletions are queued in physics and published once per graphics frame.
    // A trial file clears named voxels through that same queue.
    void serviceRtxRestore();
    void serviceRtxTrial();
    void serviceRtxFlush();
    void serviceRtxCatchup();
    bool dumpOccupancy(const std::string& path);
    // Clears air-label voxels (the EAC plug) on a volume texture AMBF has
    // just loaded or reset. Runs once per texture.
    void clearAirLabels();
    bool loadRtxTrial(const std::string& path);
    void enqueueRemoved(int x, int y, int z, const std::string& trialId, bool silent);
    void writeTrialLine(const std::string& id, int cleared, uint64_t batchId, double wallClear, double wallPublish, double clearMs, bool dropped);
    struct RtxTrialBatch {
        std::string id;
        double release_s = 0.0;
        std::vector<int> indices;
        bool drop = false;
    };
    struct RtxPendingVoxel {
        std::string trialId;
        double wallClear = 0.0;
        int x = 0;
        int y = 0;
        int z = 0;
        bool silent = false;
    };
    struct RtxBatch {
        uint64_t id = 0;
        bool published = false;
        std::vector<int> indices;
    };
    std::vector<RtxTrialBatch> m_rtxTrial;
    std::vector<RtxPendingVoxel> m_rtxPending;
    std::vector<RtxBatch> m_rtxHistory;
    size_t m_rtxTrialNext = 0;
    double m_rtxTrialStart = 0.0;
    bool m_rtxTrialLoaded = false;
    int m_rtxTrialScan = 0;
    int m_rtxRestoreWait = 0;
    uint64_t m_rtxNextBatch = 1;
    // Texture the air labels were last cleared on. A reset swaps in a new one.
    const cTexture3d* m_airClearedTexture = nullptr;
};


AF_REGISTER_SIMULATOR_PLUGIN(afVolmetricDrillingPlugin)
