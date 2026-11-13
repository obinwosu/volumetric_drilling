//==============================================================================
/*
    AMBF OpenXR eyepiece camera plugin — stereo of main_camera via WiVRn.
    Cam-clutch: HMD 6DoF → main_camera while held.
    Scene placement: LOCAL-space pose (lap default) with recenter / drag.
*/
//==============================================================================

#define GL_SILENCE_DEPRECATION
#include <afFramework.h>

// X11 / GLX types must exist before openxr_platform.h pulls in the GLX binding.
#include <X11/Xlib.h>
#include <GL/glx.h>

#ifndef XR_USE_GRAPHICS_API_OPENGL
#define XR_USE_GRAPHICS_API_OPENGL
#endif
#ifndef XR_USE_PLATFORM_XLIB
#define XR_USE_PLATFORM_XLIB
#endif
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace std;
using namespace ambf;

/// Set from volumetric_drilling (stylus / footpedal). Also see ambf_drilling_cam_clutch().
extern "C" void ambf_openxr_set_cam_clutch(int pressed);
/// Place the WiVRn scene at the current HMD look direction (LOCAL space).
extern "C" void ambf_openxr_recenter_to_head(void);
/// Place the scene down-and-forward of the current HMD (lap / keyboard pose).
extern "C" void ambf_openxr_recenter_lap(void);
/// While held (1), scene pose follows the HMD so you can drag it into place.
extern "C" void ambf_openxr_set_place_drag(int on);

class afOpenXREyepiece : public afObjectPlugin {
public:
    afOpenXREyepiece();
    virtual int init(const afBaseObjectPtr a_afObjectPtr, const afBaseObjectAttribsPtr a_objectAttribs) override;
    virtual void graphicsUpdate() override;
    virtual void physicsUpdate(double dt) override;
    virtual void reset() override;
    virtual bool close() override;

protected:
    bool readCamClutch() const;
    bool initOpenXR();
    /// Start the WiVRn connection on a background thread; AMBF keeps drawing meanwhile.
    void startConnect();
    /// True once connected (m_instance set). A refused connection is retried.
    bool pollConnect();
    void shutdownOpenXR();
    bool createSwapchains();
    bool beginXRFrame();
    void endXRFrame();
    bool renderEyeToSwapchain(int eyeIndex, const cTransform& viewLocal);
    /// OpenXR LOCAL pose → AMBF (same -90/-90 axis map as OpenVR).
    cTransform xrPoseToAmbf(const XrPosef& pose) const;
    void applyHmdClutchToMainCamera(const XrSpaceLocation& hmdLoc);
    void updateScenePlacement(const XrSpaceLocation& hmdLoc);
    /// Horizontal pixel shift for front-layer HUD so both eyes fuse at hudDepth.
    double hudStereoHalfShiftPx(uint32_t eyeWidth, const XrFovf& fov) const;
    static void shiftFrontLayerX(cWorld* frontLayer, double dxPixels);
    static cTransform defaultLapRelativeToHead();
    static cTransform cTransformFromXrPose(const XrPosef& pose);
    static XrPosef xrPoseFromCTransform(const cTransform& t);

    afCameraPtr m_camera = nullptr;
    afCameraPtr m_mainCamera = nullptr;
    cTransform m_dollyLocal;
    cTransform m_xrToAmbf;
    cTransform m_xrToAmbfInv;

    bool m_requireClutch = true;
    bool m_clutchPrev = false;
    cTransform m_mainAtClutchPress;
    cTransform m_hmdAtClutchPress;
    bool m_haveHmdRef = false;
    bool m_xrInitAttempted = false;
    bool m_xrReady = false;

    // xrCreateInstance can wait a long time for WiVRn. It runs on its own thread
    // and hands the instance over here; shared so the thread outlives close().
    struct XrConnect {
        std::mutex lock;
        int state = 0;          // 0 connecting, 1 connected, 2 failed
        bool abandoned = false; // close() ran first: the thread destroys the instance
        XrInstance instance = XR_NULL_HANDLE;
        XrSystemId system = XR_NULL_SYSTEM_ID;
    };
    std::shared_ptr<XrConnect> m_connect;
    double m_connectStartSec = 0.0;
    double m_connectNoteSec = 0.0;
    double m_retryAtSec = 0.0;  // after WiVRn refuses, try again from this time
    int m_connectTries = 0;
    bool m_warnedClutchUntracked = false;

    // Scene placement in OpenXR LOCAL (Y-up, -Z forward). Default = placeable lap panel.
    bool m_layerInLocalSpace = true;
    cTransform m_scenePoseXr;
    bool m_scenePoseInitialized = false;
    bool m_placeDragPrev = false;

    // OpenXR
    XrInstance m_instance = XR_NULL_HANDLE;
    XrSystemId m_systemId = XR_NULL_SYSTEM_ID;
    XrSession m_session = XR_NULL_HANDLE;
    XrSessionState m_sessionState = XR_SESSION_STATE_UNKNOWN;
    XrSpace m_viewSpace = XR_NULL_HANDLE;
    XrSpace m_localSpace = XR_NULL_HANDLE;
    XrFrameState m_frameState{};
    bool m_sessionRunning = false;
    bool m_frameStarted = false;

    struct Swapchain {
        XrSwapchain handle = XR_NULL_HANDLE;
        int64_t format = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<XrSwapchainImageOpenGLKHR> images;
    };
    Swapchain m_swapchains[2];
    XrViewConfigurationView m_configViews[2]{};
    XrView m_views[2]{};

    cFrameBufferPtr m_eyeFbo[2];
    cTransform m_proj[2];
    /// Apparent depth of fused HUD / pause menu (meters). Override with AMBF_OPENXR_HUD_DEPTH.
    /// ~2 m keeps left/right shifts small enough to fuse; 0.55 m was too much disparity.
    double m_hudDepthM = 3.0;

    // ---- Eyes from the FIVRS RTX window --------------------------------------
    // The RTX window path-traces a left and right microscope view and writes
    // them to /dev/shm/fivrs_rtx_eyes. While new frames arrive, those images
    // are shown as a head-placed eyepiece instead of AMBF's own rendering.
    // AMBF_OPENXR_RTX=0 turns this off. AMBF_OPENXR_RTX=only never shows AMBF:
    // black until the first RTX frame, then the latest RTX frame even if one is late.
    bool rtxPoll();
    void rtxUnmap();
    void reportStats();
    bool rtxSubmitEye(int eyeIndex);
    void rtxEyepieceFov(int eyeIndex, XrFovf& fov) const;
    bool m_rtxAllowed = true;
    bool m_rtxOnly = false;
    bool m_rtxActive = false;
    bool m_blankFrame = false;
    unsigned long m_rtxInode = 0;
    double m_rtxLastStatSec = -1.0e9;
    /// Wall-clock start; an eye file older than this is left over from an earlier session.
    long m_startWall = 0;
    bool m_rtxAnnounced = false;
    int m_rtxFd = -1;
    void* m_rtxMap = nullptr;
    size_t m_rtxMapSize = 0;
    uint64_t m_rtxSeq = 0;
    uint32_t m_rtxW = 0;
    uint32_t m_rtxH = 0;
    double m_rtxLastNewSec = -1.0e9;
    std::vector<uint8_t> m_rtxPixels;  ///< only if the pixel buffers cannot be mapped
    GLuint m_rtxPbo[2] = {0, 0};       ///< alternating upload buffers for the RTX eyes
    size_t m_rtxPboBytes = 0;
    int m_rtxPboNext = 0;
    bool m_rtxPboFailed = false;
    GLuint m_rtxTex[2] = {0, 0};
    uint32_t m_rtxTexW = 0;
    uint32_t m_rtxTexH = 0;
    XrRect2Di m_rtxRect[2]{};
    /// Horizontal width of the eyepiece image in the headset (degrees).
    double m_rtxEyepieceDeg = 65.0;
    /// Distance (m) at which the two eyepiece images fuse.
    double m_rtxFuseM = 1.5;

    /// Headset frame-rate counter, printed every few seconds (see reportStats).
    double m_statStartSec = -1.0;
    int m_statXrFrames = 0;       ///< headset frames submitted
    int m_statRtxNew = 0;         ///< new RTX frames uploaded and shown
    int m_statRtxSkipped = 0;     ///< RTX frames written but replaced before the headset took one
    double m_statWaitMs = 0.0;    ///< xrWaitFrame: waiting for the headset's next frame
    double m_statCopyMs = 0.0;    ///< shared memory -> buffer copy, new RTX frames only
    double m_statUploadMs = 0.0;  ///< buffer -> GL textures, new RTX frames only
    double m_statWorkMs = 0.0;    ///< rest of graphicsUpdate on headset frames
};

AF_REGISTER_OBJECT_PLUGIN(afOpenXREyepiece)
