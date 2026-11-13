//==============================================================================
/*
    AMBF OpenXR eyepiece — VIEW-space stereo projection of main_camera.
    Intended runtime: WiVRn (Linux → Meta Quest).
    Cam-clutch gates HMD→main_camera.
*/
//==============================================================================

#include "ambf_openxr_eyepiece.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;

namespace {
std::atomic<int> g_externalCamClutch{0};
std::atomic<int> g_recenterToHead{0};
std::atomic<int> g_recenterLap{0};
std::atomic<int> g_placeDrag{0};
typedef int (*DrillCamClutchFn)(void);
DrillCamClutchFn g_drillCamClutchFn = nullptr;

const char* xrResultString(XrResult r){
    switch (r){
        case XR_SUCCESS: return "XR_SUCCESS";
        case XR_ERROR_FORM_FACTOR_UNAVAILABLE: return "XR_ERROR_FORM_FACTOR_UNAVAILABLE";
        case XR_ERROR_RUNTIME_FAILURE: return "XR_ERROR_RUNTIME_FAILURE";
        case XR_ERROR_INSTANCE_LOST: return "XR_ERROR_INSTANCE_LOST";
        case XR_ERROR_SESSION_LOST: return "XR_ERROR_SESSION_LOST";
        case XR_ERROR_GRAPHICS_DEVICE_INVALID: return "XR_ERROR_GRAPHICS_DEVICE_INVALID";
        default: return "XR_ERROR";
    }
}

bool xrCheck(XrResult r, const char* what){
    if (XR_SUCCEEDED(r)){
        return true;
    }
    std::cerr << "ERROR! OpenXR " << what << ": " << xrResultString(r) << " (" << (int)r << ")" << std::endl;
    return false;
}

void tryResolveDrillCamClutch(){
    if (g_drillCamClutchFn){
        return;
    }
    g_drillCamClutchFn = reinterpret_cast<DrillCamClutchFn>(
        dlsym(RTLD_DEFAULT, "ambf_drilling_cam_clutch"));
    if (g_drillCamClutchFn){
        return;
    }
    const string home = getenv("HOME") ? getenv("HOME") : "";
    const string absSo = home + "/ros_ambf_ws/install/volumetric_plugin/lib/libvolumetric_drilling.so";
    for (const char* path : { "libvolumetric_drilling.so", absSo.c_str() }){
        void* handle = dlopen(path, RTLD_NOLOAD | RTLD_LAZY);
        if (!handle){
            handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
        }
        if (!handle){
            continue;
        }
        g_drillCamClutchFn = reinterpret_cast<DrillCamClutchFn>(dlsym(handle, "ambf_drilling_cam_clutch"));
        if (g_drillCamClutchFn){
            std::cout << "INFO! OpenXR eyepiece linked ambf_drilling_cam_clutch via " << path << std::endl;
            return;
        }
    }
}

cTransform xrPoseToCTransform(const XrPosef& pose){
    cTransform t;
    cVector3d p(pose.position.x, pose.position.y, pose.position.z);
    cQuaternion q;
    q.w = pose.orientation.w;
    q.x = pose.orientation.x;
    q.y = pose.orientation.y;
    q.z = pose.orientation.z;
    cMatrix3d R;
    q.toRotMat(R);
    t.set(p, R);
    return t;
}

XrPosef identityPose(){
    XrPosef p{};
    p.orientation.w = 1.f;
    return p;
}

bool poseOrientationValid(const XrPosef& pose){
    const float w = pose.orientation.w;
    const float x = pose.orientation.x;
    const float y = pose.orientation.y;
    const float z = pose.orientation.z;
    return (w * w + x * x + y * y + z * z) > 1e-6f;
}

XrFovf defaultFov(){
    // ~90° HMD-ish fallback if locateViews fails
    XrFovf fov{};
    fov.angleLeft = -0.785398f;
    fov.angleRight = 0.785398f;
    fov.angleUp = 0.785398f;
    fov.angleDown = -0.785398f;
    return fov;
}

// Shared memory written by the FIVRS RTX window (fivrs_rtx_view.py, EyeShare).
// 64-byte header, then the left and right RGBA8 images, bottom row first.
const char* kRtxEyesPath = "/dev/shm/fivrs_rtx_eyes";
const uint32_t kRtxMagic = 0x59455646u;  // "FVEY"
const size_t kRtxHeader = 64;
// Fall back to AMBF's own rendering when no new RTX frame arrives for this long.
const double kRtxStaleSec = 1.0;

double steadySeconds(){
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

uint64_t loadSeq(const void* base){
    return __atomic_load_n(reinterpret_cast<const uint64_t*>(static_cast<const char*>(base) + 8), __ATOMIC_ACQUIRE);
}

bool fovLooksValid(const XrFovf& fov){
    return std::fabs(fov.angleLeft) + std::fabs(fov.angleRight) > 1e-4f &&
           std::fabs(fov.angleUp) + std::fabs(fov.angleDown) > 1e-4f;
}
} // namespace

extern "C" void ambf_openxr_set_cam_clutch(int pressed){
    g_externalCamClutch.store(pressed ? 1 : 0);
}

extern "C" void ambf_openxr_recenter_to_head(void){
    g_recenterToHead.store(1);
}

extern "C" void ambf_openxr_recenter_lap(void){
    g_recenterLap.store(1);
}

extern "C" void ambf_openxr_set_place_drag(int on){
    g_placeDrag.store(on ? 1 : 0);
}

afOpenXREyepiece::afOpenXREyepiece(){
    // Match ambf_openvr_plugin: OpenXR/OpenVR (Y-up, -Z forward) → AMBF camera axes.
    cMatrix3d rotOffset;
    rotOffset.setExtrinsicEulerRotationDeg(-90., -90., 0., C_EULER_ORDER_XYZ);
    m_xrToAmbf.setLocalRot(rotOffset);
    m_xrToAmbfInv = m_xrToAmbf;
    m_xrToAmbfInv.invert();
    m_scenePoseXr.identity();
}

cTransform afOpenXREyepiece::xrPoseToAmbf(const XrPosef& pose) const{
    cTransform t = xrPoseToCTransform(pose);
    return m_xrToAmbfInv * t * m_xrToAmbf;
}

cTransform afOpenXREyepiece::cTransformFromXrPose(const XrPosef& pose){
    return xrPoseToCTransform(pose);
}

XrPosef afOpenXREyepiece::xrPoseFromCTransform(const cTransform& t){
    XrPosef p = identityPose();
    const cVector3d pos = t.getLocalPos();
    p.position.x = (float)pos.x();
    p.position.y = (float)pos.y();
    p.position.z = (float)pos.z();
    cQuaternion q;
    q.fromRotMat(t.getLocalRot());
    // chai may not normalize; OpenXR requires a unit quat.
    const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n > 1e-9){
        p.orientation.w = (float)(q.w / n);
        p.orientation.x = (float)(q.x / n);
        p.orientation.y = (float)(q.y / n);
        p.orientation.z = (float)(q.z / n);
    }
    return p;
}

cTransform afOpenXREyepiece::defaultLapRelativeToHead(){
    // In head/VIEW frame (OpenXR): -Z forward, +Y up. "Keyboard in lap" =
    // a bit forward and down, pitched down ~35°.
    cTransform rel;
    rel.setLocalPos(cVector3d(0.0, -0.28, -0.42));
    cMatrix3d R;
    R.setExtrinsicEulerRotationDeg(-35.0, 0.0, 0.0, C_EULER_ORDER_XYZ);
    rel.setLocalRot(R);
    return rel;
}

double afOpenXREyepiece::hudStereoHalfShiftPx(uint32_t eyeWidth, const XrFovf& fov) const{
    // Parallel-camera stereo: disparity ≈ ipd * f / depth. Half goes to each eye.
    double ipd = std::abs((double)m_views[1].pose.position.x - (double)m_views[0].pose.position.x);
    if (ipd < 0.04 || ipd > 0.09){
        ipd = 0.063;
    }
    const double tanSpan = std::tan((double)fov.angleRight) - std::tan((double)fov.angleLeft);
    if (eyeWidth < 8 || tanSpan < 1e-4 || m_hudDepthM < 1e-3){
        return 0.0;
    }
    const double focalPx = (double)eyeWidth / tanSpan;
    double half = 0.5 * ipd * focalPx / m_hudDepthM;
    // Comfort clamp: >~1% of eye width per eye tends to diplopia for UI.
    const double maxHalf = 0.01 * (double)eyeWidth;
    if (half > maxHalf){
        half = maxHalf;
    }
    return half;
}

void afOpenXREyepiece::shiftFrontLayerX(cWorld* frontLayer, double dxPixels){
    if (!frontLayer || std::fabs(dxPixels) < 1e-6){
        return;
    }
    const int n = frontLayer->getNumChildren();
    for (int i = 0; i < n; ++i){
        cGenericObject* child = frontLayer->getChild(i);
        if (!child){
            continue;
        }
        cVector3d p = child->getLocalPos();
        p(0) += dxPixels;
        child->setLocalPos(p);
    }
}

int afOpenXREyepiece::init(const afBaseObjectPtr a_afObjectPtr, const afBaseObjectAttribsPtr){
    m_camera = (afCameraPtr)a_afObjectPtr;
    if (afBaseObjectPtr parent = m_camera->getParentObject()){
        m_mainCamera = dynamic_cast<afCamera*>(parent);
    }
    if (!m_mainCamera && m_camera->m_afWorld){
        m_mainCamera = m_camera->m_afWorld->getCamera("main_camera");
    }

    m_dollyLocal = m_camera->getLocalTransform();

    const char* clutchEnv = std::getenv("AMBF_OPENXR_REQUIRE_CLUTCH");
    if (clutchEnv && (std::string(clutchEnv) == "0" || std::string(clutchEnv) == "false")){
        m_requireClutch = false;
    }

    // Default: placeable LOCAL panel (lap). AMBF_OPENXR_LAYER_SPACE=view = old head-glued eyepiece.
    const char* layerEnv = std::getenv("AMBF_OPENXR_LAYER_SPACE");
    if (layerEnv && (std::string(layerEnv) == "view" || std::string(layerEnv) == "VIEW")){
        m_layerInLocalSpace = false;
    }
    else{
        m_layerInLocalSpace = true;
    }

    if (const char* rtxEnv = std::getenv("AMBF_OPENXR_RTX")){
        const std::string v(rtxEnv);
        m_rtxAllowed = !(v == "0" || v == "false");
        m_rtxOnly = (v == "only");
    }
    if (const char* chaiEnv = std::getenv("AMBF_OPENXR_CHAI_EYES")){
        const std::string v(chaiEnv);
        m_chaiEyes = (v == "1" || v == "true");
    }
    if (m_chaiEyes){
        m_rtxAllowed = false;  // the eyepiece shows AMBF's microscope eyes, never the RTX window
        m_rtxOnly = false;
        if (const char* sizeEnv = std::getenv("AMBF_OPENXR_CHAI_SIZE")){
            unsigned w = 0, h = 0;
            if (std::sscanf(sizeEnv, "%ux%u", &w, &h) == 2 && w >= 64 && h >= 64 && w <= 4096 && h <= 4096){
                m_chaiW = w;
                m_chaiH = h;
            }
        }
        std::cout << "INFO! OpenXR eyepiece: CHAI3D microscope eyes " << m_chaiW << "x" << m_chaiH
                  << ", base " << m_chaiBaseUnits << " units, study text from /dev/shm/fivrs_hud_overlay" << std::endl;
    }
    m_startWall = (long)std::time(nullptr);
    if (const char* degEnv = std::getenv("AMBF_OPENXR_RTX_EYEPIECE_DEG")){
        const double d = std::atof(degEnv);
        if (d >= 10.0 && d <= 100.0){
            m_rtxEyepieceDeg = d;
        }
    }
    if (const char* fuseEnv = std::getenv("AMBF_OPENXR_RTX_FUSE_M")){
        const double d = std::atof(fuseEnv);
        if (d >= 0.3 && d <= 20.0){
            m_rtxFuseM = d;
        }
    }
    std::cout << "INFO! OpenXR RTX eyes "
              << (m_rtxOnly ? "only (AMBF is never shown)" : (m_rtxAllowed ? "on when the RTX window sends them" : "off"))
              << " (" << kRtxEyesPath << ", eyepiece " << m_rtxEyepieceDeg << " deg, fuse at "
              << m_rtxFuseM << " m)" << std::endl;

    if (const char* hudDepthEnv = std::getenv("AMBF_OPENXR_HUD_DEPTH")){
        const double d = std::atof(hudDepthEnv);
        if (d > 0.05 && d < 10.0){
            m_hudDepthM = d;
        }
    }

    std::cout << "INFO! OpenXR eyepiece dolly (local pos): " << m_dollyLocal.getLocalPos().str() << std::endl;
    std::cout << "INFO! OpenXR eyepiece main_camera: " << (m_mainCamera ? "OK" : "NULL") << std::endl;
    std::cout << "INFO! OpenXR eyepiece clutch "
              << (m_requireClutch ? "REQUIRED (stylus btn 1 / hold Alt)" : "always-on HMD→main_camera")
              << std::endl;
    std::cout << "INFO! OpenXR scene placement: "
              << (m_layerInLocalSpace
                      ? "LOCAL (F10 drag, F11 lap, F12 look-at)"
                      : "VIEW head-locked (AMBF_OPENXR_LAYER_SPACE=view)")
              << std::endl;
    std::cout << "INFO! OpenXR eyepiece will create the WiVRn session on the first graphics frame" << std::endl;

    // Do NOT call OpenXR here: invisible cameras / early ADF load often have a bad
    // GLX drawable and XGetWindowAttributes aborts the whole sim (nothing comes up).
    return 1;
}

bool afOpenXREyepiece::readCamClutch() const{
    if (g_externalCamClutch.load() != 0){
        return true;
    }
    tryResolveDrillCamClutch();
    return g_drillCamClutchFn && g_drillCamClutchFn() != 0;
}

void afOpenXREyepiece::applyHmdClutchToMainCamera(const XrSpaceLocation& hmdLoc){
    if (!m_mainCamera){
        if (afBaseObjectPtr parent = m_camera->getParentObject()){
            m_mainCamera = dynamic_cast<afCamera*>(parent);
        }
        if (!m_mainCamera && m_camera->m_afWorld){
            m_mainCamera = m_camera->m_afWorld->getCamera("main_camera");
        }
    }
    if (!m_mainCamera){
        return;
    }

    const bool clutch = m_requireClutch ? readCamClutch() : true;
    const bool orientOk = (hmdLoc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0
                          && poseOrientationValid(hmdLoc.pose);
    const bool posOk = (hmdLoc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;

    if (clutch && !orientOk){
        if (!m_warnedClutchUntracked){
            std::cerr << "WARNING! OpenXR cam-clutch held but HMD orientation not tracked yet"
                      << std::endl;
            m_warnedClutchUntracked = true;
        }
        m_clutchPrev = clutch;
        return;
    }

    if (clutch && orientOk){
        cTransform H = xrPoseToAmbf(hmdLoc.pose);
        if (!posOk){
            // Orientation-only: ignore translation so relative yaw/pitch still works.
            H.setLocalPos(cVector3d(0.0, 0.0, 0.0));
        }
        if (!m_clutchPrev || !m_haveHmdRef){
            m_mainAtClutchPress = m_mainCamera->getLocalTransform();
            m_hmdAtClutchPress = H;
            m_haveHmdRef = true;
            m_warnedClutchUntracked = false;
            std::cout << "INFO! OpenXR cam-clutch PRESSED — HMD moving main_camera"
                      << (posOk ? " (6DoF)" : " (orientation-only)") << std::endl;
        }
        cTransform H0inv = m_hmdAtClutchPress;
        H0inv.invert();
        cTransform D = m_dollyLocal;
        cTransform Dinv = D;
        Dinv.invert();
        // Same composition as OpenVR: eyepiece under main follows relative HMD delta.
        m_mainCamera->setLocalTransform(m_mainAtClutchPress * D * H0inv * H * Dinv);
    }
    else if (!clutch && m_clutchPrev){
        m_haveHmdRef = false;
        std::cout << "INFO! OpenXR cam-clutch RELEASED — main_camera locked" << std::endl;
    }
    m_clutchPrev = clutch;
}

void afOpenXREyepiece::updateScenePlacement(const XrSpaceLocation& hmdLoc){
    if (!m_layerInLocalSpace){
        return;
    }

    const bool orientOk = (hmdLoc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0
                          && poseOrientationValid(hmdLoc.pose);
    const bool posOk = (hmdLoc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;

    // Fallback absolute lap in LOCAL (seated origin) until HMD tracking is valid.
    if (!m_scenePoseInitialized){
        cTransform absLap;
        absLap.setLocalPos(cVector3d(0.0, -0.32, -0.50));
        cMatrix3d R;
        R.setExtrinsicEulerRotationDeg(-35.0, 0.0, 0.0, C_EULER_ORDER_XYZ);
        absLap.setLocalRot(R);
        m_scenePoseXr = absLap;
        m_scenePoseInitialized = true;
        std::cout << "INFO! OpenXR scene seeded at seated lap pose (LOCAL)" << std::endl;
    }

    if (!orientOk){
        // Still allow clearing one-shot requests so they do not stick forever.
        g_recenterToHead.store(0);
        g_recenterLap.store(0);
        return;
    }

    cTransform H = cTransformFromXrPose(hmdLoc.pose);
    if (!posOk){
        H.setLocalPos(m_scenePoseXr.getLocalPos());
    }

    const bool wantHead = g_recenterToHead.exchange(0) != 0;
    const bool wantLap = g_recenterLap.exchange(0) != 0;
    const bool dragging = g_placeDrag.load() != 0;

    if (wantHead){
        m_scenePoseXr = H;
        std::cout << "INFO! OpenXR scene placed at look direction" << std::endl;
    }
    else if (wantLap){
        m_scenePoseXr = H * defaultLapRelativeToHead();
        std::cout << "INFO! OpenXR scene placed in lap (down + forward of head)" << std::endl;
    }
    else if (dragging){
        m_scenePoseXr = H * defaultLapRelativeToHead();
        if (!m_placeDragPrev){
            std::cout << "INFO! OpenXR scene PLACE DRAG — move head, release F10 to lock" << std::endl;
        }
    }
    else if (m_placeDragPrev){
        std::cout << "INFO! OpenXR scene PLACE LOCKED" << std::endl;
    }
    m_placeDragPrev = dragging;
}

namespace {
double monoSec(){
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// Instance, system and the OpenGL requirements query: everything before
/// xrCreateSession that does not need AMBF's GL context. WiVRn can hold
/// xrCreateInstance for a long time, so this runs off the graphics thread.
bool connectInstance(XrInstance& instance, XrSystemId& system){
    const char* exts[] = {
        XR_KHR_OPENGL_ENABLE_EXTENSION_NAME,
    };

    XrInstanceCreateInfo ici{XR_TYPE_INSTANCE_CREATE_INFO};
    std::strncpy(ici.applicationInfo.applicationName,
                 "AMBF OpenXR Eyepiece",
                 XR_MAX_APPLICATION_NAME_SIZE - 1);
    ici.applicationInfo.applicationVersion = 1;
    std::strncpy(ici.applicationInfo.engineName, "AMBF", XR_MAX_ENGINE_NAME_SIZE - 1);
    ici.applicationInfo.engineVersion = 1;
    ici.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    ici.enabledExtensionCount = 1;
    ici.enabledExtensionNames = exts;

    if (!xrCheck(xrCreateInstance(&ici, &instance), "xrCreateInstance")){
        instance = XR_NULL_HANDLE;
        return false;
    }

    XrSystemGetInfo sgi{XR_TYPE_SYSTEM_GET_INFO};
    sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    bool ok = xrCheck(xrGetSystem(instance, &sgi, &system), "xrGetSystem");

    // Required before xrCreateSession.
    PFN_xrGetOpenGLGraphicsRequirementsKHR pfnReq = nullptr;
    if (ok){
        xrGetInstanceProcAddr(instance, "xrGetOpenGLGraphicsRequirementsKHR",
                              reinterpret_cast<PFN_xrVoidFunction*>(&pfnReq));
        if (!pfnReq){
            std::cerr << "ERROR! xrGetOpenGLGraphicsRequirementsKHR missing" << std::endl;
            ok = false;
        }
    }
    if (ok){
        XrGraphicsRequirementsOpenGLKHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR};
        ok = xrCheck(pfnReq(instance, system, &req), "xrGetOpenGLGraphicsRequirementsKHR");
    }
    if (!ok){
        xrDestroyInstance(instance);
        instance = XR_NULL_HANDLE;
    }
    return ok;
}
}  // namespace

void afOpenXREyepiece::startConnect(){
    auto shared = std::make_shared<XrConnect>();
    m_connect = shared;
    if (m_connectTries++ == 0){
        m_connectStartSec = monoSec();
        std::cout << "INFO! OpenXR eyepiece connecting to WiVRn (AMBF keeps running meanwhile)" << std::endl;
    }
    m_connectNoteSec = monoSec();
    std::thread([shared](){
        XrInstance instance = XR_NULL_HANDLE;
        XrSystemId system = XR_NULL_SYSTEM_ID;
        const bool ok = connectInstance(instance, system);
        std::lock_guard<std::mutex> guard(shared->lock);
        if (shared->abandoned){
            if (instance != XR_NULL_HANDLE){
                xrDestroyInstance(instance);
            }
            return;
        }
        shared->instance = instance;
        shared->system = system;
        shared->state = ok ? 1 : 2;
    }).detach();
}

bool afOpenXREyepiece::pollConnect(){
    if (!m_connect){
        return true;
    }
    int state;
    {
        std::lock_guard<std::mutex> guard(m_connect->lock);
        state = m_connect->state;
        if (state == 1){
            m_instance = m_connect->instance;
            m_systemId = m_connect->system;
            m_connect->instance = XR_NULL_HANDLE;
        }
    }
    if (state == 0){
        const double now = monoSec();
        if (now - m_connectNoteSec >= 10.0){
            m_connectNoteSec = now;
            std::cout << "INFO! OpenXR eyepiece still waiting for WiVRn ("
                      << (int)(now - m_connectStartSec) << " s)" << std::endl;
        }
        return false;
    }
    const double now = monoSec();
    const double waited = now - m_connectStartSec;
    m_connect.reset();
    if (state == 2){
        // WiVRn may be restarting (headset reconnecting): ask again shortly.
        m_retryAtSec = now + 3.0;
        std::cout << "INFO! OpenXR eyepiece: WiVRn refused the connection after " << (int)waited
                  << " s; retrying in 3 s" << std::endl;
        return false;
    }
    if (waited >= 5.0){
        std::cout << "INFO! OpenXR eyepiece: WiVRn let AMBF in after " << (int)waited << " s" << std::endl;
    }
    return true;
}

/// Session, spaces and swapchains, on the graphics thread (needs AMBF's GL
/// context). m_instance and m_systemId come from connectInstance.
bool afOpenXREyepiece::initOpenXR(){
    if (m_instance == XR_NULL_HANDLE){
        return false;
    }

    Display* dpy = glXGetCurrentDisplay();
    GLXContext ctx = glXGetCurrentContext();
    GLXDrawable drawable = glXGetCurrentDrawable();
    if (!dpy || !ctx || !drawable){
        std::cerr << "ERROR! No current GLX context — OpenXR eyepiece needs AMBF GL context" << std::endl;
        return false;
    }

    // Never query the drawable as an X Window (invisible cameras / FBO paths
    // can yield BadWindow and kill AMBF). Use the default visual id instead.
    Visual* defVis = DefaultVisual(dpy, DefaultScreen(dpy));
    const uint32_t visualId = defVis ? (uint32_t)XVisualIDFromVisual(defVis) : 0;

    GLXFBConfig fbConfig = nullptr;
    int nFb = 0;
    static int fbAttribs[] = {
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_DOUBLEBUFFER, True,
        GLX_RED_SIZE, 8,
        GLX_GREEN_SIZE, 8,
        GLX_BLUE_SIZE, 8,
        GLX_ALPHA_SIZE, 8,
        None
    };
    GLXFBConfig* fbConfigs = glXChooseFBConfig(dpy, DefaultScreen(dpy), fbAttribs, &nFb);
    if (fbConfigs && nFb > 0){
        fbConfig = fbConfigs[0];
        XFree(fbConfigs);
    }

    XrGraphicsBindingOpenGLXlibKHR glBinding{XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR};
    glBinding.xDisplay = dpy;
    glBinding.visualid = visualId;
    glBinding.glxFBConfig = fbConfig;
    glBinding.glxDrawable = drawable;
    glBinding.glxContext = ctx;

    XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
    sci.next = &glBinding;
    sci.systemId = m_systemId;
    if (!xrCheck(xrCreateSession(m_instance, &sci, &m_session), "xrCreateSession")){
        return false;
    }

    XrReferenceSpaceCreateInfo vsci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    vsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    vsci.poseInReferenceSpace = identityPose();
    if (!xrCheck(xrCreateReferenceSpace(m_session, &vsci, &m_viewSpace), "xrCreateReferenceSpace VIEW")){
        return false;
    }

    XrReferenceSpaceCreateInfo lsci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    lsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    lsci.poseInReferenceSpace = identityPose();
    if (!xrCheck(xrCreateReferenceSpace(m_session, &lsci, &m_localSpace), "xrCreateReferenceSpace LOCAL")){
        return false;
    }

    if (!createSwapchains()){
        return false;
    }

    // Allocate AMBF FBOs matching swapchain size
    for (int i = 0; i < 2; ++i){
        m_eyeFbo[i] = cFrameBuffer::create();
        m_eyeFbo[i]->setup(m_camera->getInternalCamera(),
                           m_swapchains[i].width, m_swapchains[i].height,
                           true, true, GL_RGBA);
    }

    // Panel HUD placement uses afCamera width/height; keep them matched to the
    // eye FBO so the pause menu centers correctly in the HMD.
    if (m_camera){
        m_camera->m_width = m_swapchains[0].width;
        m_camera->m_height = m_swapchains[0].height;
    }

    std::cout << "INFO! OpenXR eyepiece session created ("
              << m_swapchains[0].width << "x" << m_swapchains[0].height << " per eye)" << std::endl;
    return true;
}

bool afOpenXREyepiece::createSwapchains(){
    uint32_t viewCount = 0;
    xrEnumerateViewConfigurationViews(m_instance, m_systemId,
                                      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                      0, &viewCount, nullptr);
    if (viewCount < 2){
        std::cerr << "ERROR! OpenXR stereo view count < 2" << std::endl;
        return false;
    }
    viewCount = 2;
    for (int i = 0; i < 2; ++i){
        m_configViews[i] = XrViewConfigurationView{XR_TYPE_VIEW_CONFIGURATION_VIEW};
        m_views[i] = XrView{XR_TYPE_VIEW};
    }
    if (!xrCheck(xrEnumerateViewConfigurationViews(m_instance, m_systemId,
                                                   XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                                   viewCount, &viewCount, m_configViews),
                 "xrEnumerateViewConfigurationViews")){
        return false;
    }

    int64_t format = GL_RGBA8;
    {
        uint32_t n = 0;
        xrEnumerateSwapchainFormats(m_session, 0, &n, nullptr);
        std::vector<int64_t> formats(n);
        xrEnumerateSwapchainFormats(m_session, n, &n, formats.data());
        for (int64_t f : formats){
            if (f == GL_RGBA8 || f == 0x8058 /* GL_RGBA8 */){
                format = f;
                break;
            }
        }
        if (!formats.empty()){
            format = formats[0];
        }
        // The RTX eyes arrive sRGB-encoded, as shown on the desktop. In a linear
        // swapchain the compositor would encode them again and wash them out.
        const int64_t kSrgb8Alpha8 = 0x8C43;  // GL_SRGB8_ALPHA8
        if (m_rtxOnly || m_chaiEyes){
            // CHAI's eyes are display-encoded too (what the desktop window shows).
            for (int64_t f : formats){
                if (f == kSrgb8Alpha8){
                    format = f;
                    break;
                }
            }
        }
        std::cout << "INFO! OpenXR swapchain format 0x" << std::hex << format << std::dec
                  << (format == kSrgb8Alpha8 ? " (sRGB)" : " (linear)") << ", runtime offers";
        for (int64_t f : formats){
            std::cout << " 0x" << std::hex << f << std::dec;
        }
        std::cout << std::endl;
    }

    for (int eye = 0; eye < 2; ++eye){
        auto& sc = m_swapchains[eye];
        sc.width = m_configViews[eye].recommendedImageRectWidth;
        sc.height = m_configViews[eye].recommendedImageRectHeight;
        sc.format = format;

        XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
        ci.format = format;
        ci.sampleCount = 1;
        ci.width = sc.width;
        ci.height = sc.height;
        ci.faceCount = 1;
        ci.arraySize = 1;
        ci.mipCount = 1;
        if (!xrCheck(xrCreateSwapchain(m_session, &ci, &sc.handle), "xrCreateSwapchain")){
            return false;
        }

        uint32_t imgCount = 0;
        xrEnumerateSwapchainImages(sc.handle, 0, &imgCount, nullptr);
        sc.images.resize(imgCount, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR});
        if (!xrCheck(xrEnumerateSwapchainImages(sc.handle, imgCount, &imgCount,
                                                reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data())),
                     "xrEnumerateSwapchainImages")){
            return false;
        }
    }
    return true;
}

void afOpenXREyepiece::shutdownOpenXR(){
    for (int i = 0; i < 2; ++i){
        if (m_swapchains[i].handle != XR_NULL_HANDLE){
            xrDestroySwapchain(m_swapchains[i].handle);
            m_swapchains[i].handle = XR_NULL_HANDLE;
        }
    }
    if (m_viewSpace != XR_NULL_HANDLE){
        xrDestroySpace(m_viewSpace);
        m_viewSpace = XR_NULL_HANDLE;
    }
    if (m_localSpace != XR_NULL_HANDLE){
        xrDestroySpace(m_localSpace);
        m_localSpace = XR_NULL_HANDLE;
    }
    if (m_session != XR_NULL_HANDLE){
        xrDestroySession(m_session);
        m_session = XR_NULL_HANDLE;
    }
    if (m_instance != XR_NULL_HANDLE){
        xrDestroyInstance(m_instance);
        m_instance = XR_NULL_HANDLE;
    }
    m_sessionRunning = false;
}

void afOpenXREyepiece::graphicsUpdate(){
    if (!m_xrInitAttempted){
        if (!m_connect){
            if (monoSec() < m_retryAtSec){
                return;
            }
            startConnect();
        }
        if (!pollConnect()){
            return;  // still connecting: AMBF carries on, the headset stays black
        }
        m_xrInitAttempted = true;
        m_xrReady = initOpenXR();
        if (!m_xrReady){
            std::cerr << "ERROR! OpenXR eyepiece failed to start. "
                      << "Is WiVRn running with Quest 'Connection ready'? "
                      << "Check XR_RUNTIME_JSON=" << (getenv("XR_RUNTIME_JSON") ? getenv("XR_RUNTIME_JSON") : "(unset)")
                      << std::endl;
            return;
        }
    }
    if (!m_xrReady || m_instance == XR_NULL_HANDLE){
        return;
    }

    // Poll events
    XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(m_instance, &ev) == XR_SUCCESS){
        if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED){
            auto* s = reinterpret_cast<XrEventDataSessionStateChanged*>(&ev);
            m_sessionState = s->state;
            if (s->state == XR_SESSION_STATE_READY){
                XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
                bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                xrBeginSession(m_session, &bi);
                m_sessionRunning = true;
                // Prefer lap relative to the live head once tracking is up.
                g_recenterLap.store(1);
                std::cout << "INFO! OpenXR eyepiece session READY / begun" << std::endl;
            }
            else if (s->state == XR_SESSION_STATE_STOPPING){
                xrEndSession(m_session);
                m_sessionRunning = false;
            }
            else if (s->state == XR_SESSION_STATE_EXITING || s->state == XR_SESSION_STATE_LOSS_PENDING){
                m_sessionRunning = false;
            }
        }
        ev = XrEventDataBuffer{XR_TYPE_EVENT_DATA_BUFFER};
    }

    if (!m_sessionRunning){
        return;
    }

    if (!m_mainCamera && m_camera->m_afWorld){
        m_mainCamera = m_camera->m_afWorld->getCamera("main_camera");
    }

    reportStats();
    const double frameStart = steadySeconds();
    const double waitBefore = m_statWaitMs;
    if (!beginXRFrame()){
        return;
    }
    m_statXrFrames++;
    // Work on this headset frame, without the wait for the headset (counted separately).
    struct WorkTimer {
        afOpenXREyepiece* self; double start; double waitBefore;
        ~WorkTimer(){
            self->m_statWorkMs += (steadySeconds() - start) * 1e3 - (self->m_statWaitMs - waitBefore);
        }
    } workTimer{this, frameStart, waitBefore};

    // Cam-clutch: VIEW pose in LOCAL → AMBF axes → main_camera only while held.
    XrSpaceLocation hmdLoc{XR_TYPE_SPACE_LOCATION};
    if (m_localSpace != XR_NULL_HANDLE && m_viewSpace != XR_NULL_HANDLE &&
        m_frameState.predictedDisplayTime){
        xrLocateSpace(m_viewSpace, m_localSpace, m_frameState.predictedDisplayTime, &hmdLoc);
    }
    applyHmdClutchToMainCamera(hmdLoc);
    updateScenePlacement(hmdLoc);

    if (m_chaiEyes){
        hudPoll();
        if (chaiRenderEyes()){
            for (int eye = 0; eye < 2; ++eye){
                rtxSubmitEye(eye, m_chaiFbo[eye]->m_imageBuffer->getTextureId());
            }
        }
        else{
            m_blankFrame = true;
        }
        endXRFrame();
        m_blankFrame = false;
        return;
    }
    if (rtxPoll()){
        // The RTX window's path-traced eyes replace AMBF's own rendering.
        for (int eye = 0; eye < 2; ++eye){
            rtxSubmitEye(eye);
        }
        endXRFrame();
        return;
    }
    if (m_rtxOnly){
        // RTX only: nothing to show until the RTX window's first frame. Black, not AMBF.
        m_blankFrame = true;
        endXRFrame();
        m_blankFrame = false;
        return;
    }

    // Content always from ADF dolly under main_camera; layer pose places that stereo in the room.
    cTransform T_view = m_dollyLocal;
    cTransform backupProj = m_camera->getInternalCamera()->m_projectionMatrix;
    bool backupCustom = m_camera->getInternalCamera()->m_useCustomProjectionMatrix;
    cWorld* hudFront = m_camera->getInternalCamera()->m_frontLayer;

    for (int eye = 0; eye < 2; ++eye){
        // Stereo offset from OpenXR view pose in VIEW space (IPD), applied in local camera frame.
        cTransform eyeLocal = xrPoseToCTransform(m_views[eye].pose);
        cTransform viewMat = T_view * eyeLocal;

        // Build projection from OpenXR FOV
        const XrFovf& fov = m_views[eye].fov;
        float nearZ = (float)m_camera->getInternalCamera()->getNearClippingPlane();
        float farZ = (float)m_camera->getInternalCamera()->getFarClippingPlane();
        float l = tanf(fov.angleLeft) * nearZ;
        float r = tanf(fov.angleRight) * nearZ;
        float b = tanf(fov.angleDown) * nearZ;
        float t = tanf(fov.angleUp) * nearZ;
        double A = (r + l) / (r - l);
        double B = (t + b) / (t - b);
        double C = -(farZ + nearZ) / (farZ - nearZ);
        double D = -(2.0 * farZ * nearZ) / (farZ - nearZ);
        double nrl = (2.0 * nearZ) / (r - l);
        double ntb = (2.0 * nearZ) / (t - b);
        cTransform P;
        P.identity();
        double m[4][4] = {
            {nrl, 0, A, 0},
            {0, ntb, B, 0},
            {0, 0, C, D},
            {0, 0, -1, 0}
        };
        P.set(m[0][0], m[1][0], m[2][0], m[3][0],
              m[0][1], m[1][1], m[2][1], m[3][1],
              m[0][2], m[1][2], m[2][2], m[3][2],
              m[0][3], m[1][3], m[2][3], m[3][3]);
        m_proj[eye] = P;

        // Stereo HUD: same labels in both eyes with opposite X shifts so they fuse
        // at m_hudDepthM (not identical zero-disparity copies, not left-eye-only).
        const double halfShift = hudStereoHalfShiftPx(m_swapchains[eye].width, fov);
        const double eyeShift = (eye == 0) ? halfShift : -halfShift;
        shiftFrontLayerX(hudFront, eyeShift);

        m_camera->getInternalCamera()->m_frontLayer = hudFront;
        m_camera->getInternalCamera()->m_useCustomProjectionMatrix = true;
        m_camera->getInternalCamera()->m_projectionMatrix = m_proj[eye];
        m_camera->setLocalTransform(viewMat);
        m_eyeFbo[eye]->renderView();

        shiftFrontLayerX(hudFront, -eyeShift);
        renderEyeToSwapchain(eye, viewMat);
    }

    m_camera->getInternalCamera()->m_frontLayer = hudFront;
    m_camera->getInternalCamera()->m_projectionMatrix = backupProj;
    m_camera->getInternalCamera()->m_useCustomProjectionMatrix = backupCustom;
    m_camera->setLocalTransform(m_dollyLocal);

    endXRFrame();
}

bool afOpenXREyepiece::beginXRFrame(){
    m_frameStarted = false;
    m_frameState = XrFrameState{XR_TYPE_FRAME_STATE};
    XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};
    const double waitStart = steadySeconds();
    const XrResult waited = xrWaitFrame(m_session, &wi, &m_frameState);
    m_statWaitMs += (steadySeconds() - waitStart) * 1e3;
    if (!xrCheck(waited, "xrWaitFrame")){
        return false;
    }
    XrFrameBeginInfo bi{XR_TYPE_FRAME_BEGIN_INFO};
    if (!xrCheck(xrBeginFrame(m_session, &bi), "xrBeginFrame")){
        return false;
    }
    m_frameStarted = true;

    // Always start from a valid VIEW-space stereo layout. WiVRn often returns
    // zero quats when locating views in VIEW space; invalid poses make
    // xrEndFrame fail and the Quest stays on "waiting for an application".
    for (int i = 0; i < 2; ++i){
        m_views[i] = XrView{XR_TYPE_VIEW};
        m_views[i].pose = identityPose();
        m_views[i].fov = defaultFov();
    }
    // Mild IPD (±32 mm) so stereo is not identical if locate fails.
    m_views[0].pose.position.x = -0.032f;
    m_views[1].pose.position.x = 0.032f;

    if (!m_frameState.shouldRender){
        return true;
    }

    XrViewLocateInfo vli{XR_TYPE_VIEW_LOCATE_INFO};
    vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    vli.displayTime = m_frameState.predictedDisplayTime;
    // Locate in LOCAL (reliable FOV/IPD), then convert to VIEW-relative poses
    // so the composition layer can stay head-locked (layer.space = VIEW).
    vli.space = m_localSpace;
    XrViewState vs{XR_TYPE_VIEW_STATE};
    XrView located[2] = {XrView{XR_TYPE_VIEW}, XrView{XR_TYPE_VIEW}};
    uint32_t vc = 2;
    const XrResult lr = xrLocateViews(m_session, &vli, &vs, vc, &vc, located);
    const bool orientOk = XR_SUCCEEDED(lr) &&
                          (vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT);
    const bool posOk = XR_SUCCEEDED(lr) &&
                       (vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT);
    if (orientOk && posOk && vc >= 2 &&
        poseOrientationValid(located[0].pose) && poseOrientationValid(located[1].pose)){
        // Midpoint between eyes ≈ head; eye poses relative to that = VIEW layout.
        const float mx = 0.5f * (located[0].pose.position.x + located[1].pose.position.x);
        const float my = 0.5f * (located[0].pose.position.y + located[1].pose.position.y);
        const float mz = 0.5f * (located[0].pose.position.z + located[1].pose.position.z);
        for (int i = 0; i < 2; ++i){
            m_views[i].pose = identityPose();
            m_views[i].pose.position.x = located[i].pose.position.x - mx;
            m_views[i].pose.position.y = located[i].pose.position.y - my;
            m_views[i].pose.position.z = located[i].pose.position.z - mz;
            m_views[i].fov = fovLooksValid(located[i].fov) ? located[i].fov : defaultFov();
        }
    }
    return true;
}

bool afOpenXREyepiece::renderEyeToSwapchain(int eyeIndex, const cTransform&){
    if (!m_frameState.shouldRender){
        return true;
    }
    auto& sc = m_swapchains[eyeIndex];
    XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    uint32_t imageIndex = 0;
    if (!xrCheck(xrAcquireSwapchainImage(sc.handle, &ai, &imageIndex), "xrAcquireSwapchainImage")){
        return false;
    }
    XrSwapchainImageWaitInfo wai{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wai.timeout = XR_INFINITE_DURATION;
    if (!xrCheck(xrWaitSwapchainImage(sc.handle, &wai), "xrWaitSwapchainImage")){
        return false;
    }

    GLuint dst = sc.images[imageIndex].image;
    GLuint src = m_eyeFbo[eyeIndex]->m_imageBuffer->getTextureId();

    // Blit FBO color texture → swapchain texture via FBO blit
    GLint prevFbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);

    GLuint readFbo = 0, drawFbo = 0;
    glGenFramebuffers(1, &readFbo);
    glGenFramebuffers(1, &drawFbo);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src, 0);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFbo);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst, 0);

    glBlitFramebuffer(0, 0, sc.width, sc.height,
                      0, 0, sc.width, sc.height,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
    glDeleteFramebuffers(1, &readFbo);
    glDeleteFramebuffers(1, &drawFbo);

    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(sc.handle, &ri);
    return true;
}

void afOpenXREyepiece::endXRFrame(){
    if (!m_frameStarted){
        return;
    }

    XrCompositionLayerProjectionView projViews[2]{};
    XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    layer.layerFlags = 0;
    // LOCAL = placeable seated panel; VIEW = classic head-glued eyepiece.
    layer.space = m_layerInLocalSpace ? m_localSpace : m_viewSpace;
    layer.viewCount = 2;
    layer.views = projViews;

    for (int eye = 0; eye < 2; ++eye){
        projViews[eye] = XrCompositionLayerProjectionView{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
        XrPosef eyePose = poseOrientationValid(m_views[eye].pose) ? m_views[eye].pose : identityPose();
        if (m_layerInLocalSpace && m_scenePoseInitialized){
            // Scene center in LOCAL × per-eye IPD offset (VIEW-relative).
            const cTransform worldEye = m_scenePoseXr * cTransformFromXrPose(eyePose);
            eyePose = xrPoseFromCTransform(worldEye);
            if (!poseOrientationValid(eyePose)){
                eyePose = xrPoseFromCTransform(m_scenePoseXr);
            }
        }
        projViews[eye].pose = eyePose;
        projViews[eye].fov = fovLooksValid(m_views[eye].fov) ? m_views[eye].fov : defaultFov();
        projViews[eye].subImage.swapchain = m_swapchains[eye].handle;
        projViews[eye].subImage.imageRect.offset = {0, 0};
        projViews[eye].subImage.imageRect.extent = {(int32_t)m_swapchains[eye].width,
                                                    (int32_t)m_swapchains[eye].height};
        if (m_rtxActive){
            // The RTX image covers only the eyepiece: the compositor draws it at
            // that angle and leaves the rest of the view black.
            rtxEyepieceFov(eye, projViews[eye].fov);
            projViews[eye].subImage.imageRect = m_rtxRect[eye];
        }
    }

    XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
    ei.displayTime = m_frameState.predictedDisplayTime;
    ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    const XrCompositionLayerBaseHeader* layers[1] = {
        reinterpret_cast<const XrCompositionLayerBaseHeader*>(&layer)
    };
    if (m_frameState.shouldRender && !m_blankFrame){
        ei.layerCount = 1;
        ei.layers = layers;
    }
    else{
        ei.layerCount = 0;
        ei.layers = nullptr;
    }
    xrEndFrame(m_session, &ei);
    m_frameStarted = false;
}

bool afOpenXREyepiece::rtxPoll(){
    if (!m_rtxAllowed){
        return false;
    }
    const double now = steadySeconds();
    if (m_rtxMap && now - m_rtxLastStatSec > 0.5){
        // The RTX window writes a new file each time it starts. Follow it.
        m_rtxLastStatSec = now;
        struct stat st{};
        if (::stat(kRtxEyesPath, &st) != 0 || (unsigned long)st.st_ino != m_rtxInode){
            rtxUnmap();
        }
    }
    if (!m_rtxMap){
        // Until a file appears, keep the last RTX frame in RTX-only mode, else nothing.
        m_rtxActive = m_rtxOnly && m_rtxTex[0] != 0;
        const int fd = ::open(kRtxEyesPath, O_RDONLY);
        if (fd < 0){
            return m_rtxActive;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0 || (size_t)st.st_size < kRtxHeader || (long)st.st_mtime < m_startWall - 1){
            // Too small, or left over from an earlier session.
            ::close(fd);
            return m_rtxActive;
        }
        void* map = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED){
            ::close(fd);
            return m_rtxActive;
        }
        m_rtxFd = fd;
        m_rtxMap = map;
        m_rtxMapSize = (size_t)st.st_size;
        m_rtxInode = (unsigned long)st.st_ino;
        m_rtxLastStatSec = now;
        // A new file starts a new sequence.
        m_rtxSeq = 0;
    }
    const char* base = static_cast<const char*>(m_rtxMap);
    uint32_t magic = 0, width = 0, height = 0;
    std::memcpy(&magic, base, 4);
    std::memcpy(&width, base + 16, 4);
    std::memcpy(&height, base + 20, 4);
    const size_t eyeBytes = (size_t)width * height * 4;
    if (magic != kRtxMagic || width == 0 || height == 0 || kRtxHeader + 2 * eyeBytes > m_rtxMapSize){
        // The RTX window restarted with another size: map the file again next frame.
        rtxUnmap();
        m_rtxActive = m_rtxOnly && m_rtxTex[0] != 0;
        return m_rtxActive;
    }
    const uint64_t seq = loadSeq(base);
    if ((seq & 1u) == 0 && seq != m_rtxSeq){
        // Even = complete. Copy, then check nothing was written meanwhile.
        const size_t bytes = 2 * eyeBytes;
        if (m_rtxTexW != width || m_rtxTexH != height || m_rtxTex[0] == 0){
            if (m_rtxTex[0] != 0){
                glDeleteTextures(2, m_rtxTex);
            }
            glGenTextures(2, m_rtxTex);
            GLint pboAtCreate = 0;
            glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &pboAtCreate);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            GLint texAtCreate = 0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &texAtCreate);
            for (int eye = 0; eye < 2; ++eye){
                glBindTexture(GL_TEXTURE_2D, m_rtxTex[eye]);
                // Same format as the headset images, so the copy is raw bytes.
                glTexImage2D(GL_TEXTURE_2D, 0, (GLint)m_swapchains[eye].format, (GLsizei)width, (GLsizei)height,
                             0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            }
            glBindTexture(GL_TEXTURE_2D, (GLuint)texAtCreate);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, (GLuint)pboAtCreate);
            m_rtxTexW = width;
            m_rtxTexH = height;
        }
        // CHAI's partial voxel-texture updates (from the first drill contact on)
        // leave the unpack row length and skips set. Upload with tightly packed
        // rows, then put AMBF's settings back.
        const GLenum keys[] = {GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_PIXELS,
                               GL_UNPACK_SKIP_ROWS, GL_UNPACK_IMAGE_HEIGHT, GL_UNPACK_SKIP_IMAGES};
        const GLint packed[] = {4, 0, 0, 0, 0, 0};
        GLint saved[6] = {4, 0, 0, 0, 0, 0};
        GLint savedPbo = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &savedPbo);
        GLint savedTex = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTex);

        // One copy, from shared memory straight into a GL pixel buffer; the texture
        // upload from it then runs on the GPU without stalling AMBF's thread. Two
        // buffers alternate so a frame never waits on the previous frame's upload.
        const uint8_t* pixels = nullptr;  // client-memory fallback
        bool complete = false;
        const double copyStart = steadySeconds();
        if (!m_rtxPboFailed){
            if (m_rtxPbo[0] == 0 || m_rtxPboBytes != bytes){
                if (m_rtxPbo[0] == 0){
                    glGenBuffers(2, m_rtxPbo);
                }
                for (int k = 0; k < 2; ++k){
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_rtxPbo[k]);
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)bytes, nullptr, GL_STREAM_DRAW);
                }
                m_rtxPboBytes = bytes;
            }
            const int k = m_rtxPboNext;
            m_rtxPboNext ^= 1;
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_rtxPbo[k]);
            void* dst = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, (GLsizeiptr)bytes,
                                         GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
            if (dst){
                std::memcpy(dst, base + kRtxHeader, bytes);
                const bool unmapped = glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER) == GL_TRUE;
                complete = unmapped && loadSeq(base) == seq;
            }
            else{
                m_rtxPboFailed = true;
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                std::cerr << "WARNING! OpenXR eyepiece: pixel buffer map failed; uploading from memory." << std::endl;
            }
        }
        if (m_rtxPboFailed){
            m_rtxPixels.resize(bytes);
            std::memcpy(m_rtxPixels.data(), base + kRtxHeader, bytes);
            complete = loadSeq(base) == seq;
            pixels = m_rtxPixels.data();
        }
        m_statCopyMs += (steadySeconds() - copyStart) * 1e3;

        if (complete){
            const double uploadStart = steadySeconds();
            m_statRtxNew++;
            if (m_rtxSeq != 0 && seq > m_rtxSeq + 2){
                m_statRtxSkipped += (int)((seq - m_rtxSeq) / 2 - 1);
            }
            m_rtxSeq = seq;
            m_rtxW = width;
            m_rtxH = height;
            m_rtxLastNewSec = now;
            for (int k = 0; k < 6; ++k){
                glGetIntegerv(keys[k], &saved[k]);
                glPixelStorei(keys[k], packed[k]);
            }
            for (int eye = 0; eye < 2; ++eye){
                glBindTexture(GL_TEXTURE_2D, m_rtxTex[eye]);
                // With a pixel buffer bound the last argument is an offset into it.
                const void* source = pixels ? (const void*)(pixels + eye * eyeBytes)
                                            : (const void*)(uintptr_t)(eye * eyeBytes);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)width, (GLsizei)height, GL_RGBA, GL_UNSIGNED_BYTE,
                                source);
            }
            for (int k = 0; k < 6; ++k){
                glPixelStorei(keys[k], saved[k]);
            }
            m_statUploadMs += (steadySeconds() - uploadStart) * 1e3;
        }
        glBindTexture(GL_TEXTURE_2D, (GLuint)savedTex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, (GLuint)savedPbo);
    }
    // RTX only keeps the last frame through a late one; otherwise fall back to AMBF.
    const bool active = m_rtxTex[0] != 0 && (m_rtxOnly || (now - m_rtxLastNewSec) <= kRtxStaleSec);
    if (active != m_rtxActive || !m_rtxAnnounced){
        std::cout << "INFO! OpenXR eyepiece shows " << (active ? "the RTX window eyes" : "AMBF rendering")
                  << " (" << m_rtxW << "x" << m_rtxH << ")" << std::endl;
        m_rtxAnnounced = true;
    }
    m_rtxActive = active;
    return active;
}

void afOpenXREyepiece::reportStats(){
    // Every 5 s: how often the headset gets a new RTX image, and where the time goes.
    const double now = steadySeconds();
    if (m_statStartSec < 0.0){
        m_statStartSec = now;
        return;
    }
    const double span = now - m_statStartSec;
    if (span < 5.0){
        return;
    }
    const int frames = std::max(m_statXrFrames, 1);
    const int fresh = std::max(m_statRtxNew, 1);
    char line[400];
    std::snprintf(line, sizeof(line),
                  "INFO! OpenXR eyepiece: headset %.1f fps, new RTX images %.1f/s (%d replaced before shown), "
                  "per headset frame wait %.1f ms + work %.1f ms, per RTX image copy %.1f ms + upload %.1f ms",
                  m_statXrFrames / span, m_statRtxNew / span, m_statRtxSkipped,
                  m_statWaitMs / frames, m_statWorkMs / frames,
                  m_statRtxNew ? m_statCopyMs / fresh : 0.0, m_statRtxNew ? m_statUploadMs / fresh : 0.0);
    std::cout << line << std::endl;
    m_statStartSec = now;
    m_statXrFrames = m_statRtxNew = m_statRtxSkipped = 0;
    m_statWaitMs = m_statCopyMs = m_statUploadMs = m_statWorkMs = 0.0;
}

void afOpenXREyepiece::rtxUnmap(){
    if (m_rtxMap){
        munmap(m_rtxMap, m_rtxMapSize);
        m_rtxMap = nullptr;
    }
    if (m_rtxFd >= 0){
        ::close(m_rtxFd);
        m_rtxFd = -1;
    }
    m_rtxMapSize = 0;
    m_rtxInode = 0;
}

void afOpenXREyepiece::rtxEyepieceFov(int eyeIndex, XrFovf& fov) const{
    // Eyepiece width in the headset, and the height from the image aspect.
    const double half = 0.5 * m_rtxEyepieceDeg * M_PI / 180.0;
    const double aspect = (m_rtxW > 0) ? (double)m_rtxH / (double)m_rtxW : 1.0;
    const double halfUp = std::atan(std::tan(half) * aspect);
    // Turn each eye's image a little toward the nose so the two fuse at m_rtxFuseM.
    double ipd = std::fabs((double)m_views[1].pose.position.x - (double)m_views[0].pose.position.x);
    if (ipd < 0.04 || ipd > 0.09){
        ipd = 0.063;
    }
    const double inward = std::atan(0.5 * ipd / m_rtxFuseM);
    const double shift = (eyeIndex == 0) ? inward : -inward;
    fov.angleLeft = (float)(-half + shift);
    fov.angleRight = (float)(half + shift);
    fov.angleUp = (float)halfUp;
    fov.angleDown = (float)(-halfUp);
}

bool afOpenXREyepiece::rtxSubmitEye(int eyeIndex, GLuint source){
    if (!m_frameState.shouldRender){
        return true;
    }
    auto& sc = m_swapchains[eyeIndex];
    XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    uint32_t imageIndex = 0;
    if (!xrCheck(xrAcquireSwapchainImage(sc.handle, &ai, &imageIndex), "xrAcquireSwapchainImage")){
        return false;
    }
    XrSwapchainImageWaitInfo wai{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wai.timeout = XR_INFINITE_DURATION;
    if (!xrCheck(xrWaitSwapchainImage(sc.handle, &wai), "xrWaitSwapchainImage")){
        return false;
    }
    // Fit the RTX image inside the headset image, keeping its aspect.
    const double scale = std::min(1.0, std::min((double)sc.width / m_rtxW, (double)sc.height / m_rtxH));
    const int32_t w = (int32_t)std::lround(m_rtxW * scale);
    const int32_t h = (int32_t)std::lround(m_rtxH * scale);
    m_rtxRect[eyeIndex].offset = {0, 0};
    m_rtxRect[eyeIndex].extent = {w, h};

    GLint prevFbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
    const GLboolean srgbWasOn = glIsEnabled(GL_FRAMEBUFFER_SRGB);
    glDisable(GL_FRAMEBUFFER_SRGB);
    GLuint readFbo = 0, drawFbo = 0;
    glGenFramebuffers(1, &readFbo);
    glGenFramebuffers(1, &drawFbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           source != 0 ? source : m_rtxTex[eyeIndex], 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFbo);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sc.images[imageIndex].image, 0);
    glBlitFramebuffer(0, 0, (GLint)m_rtxW, (GLint)m_rtxH, 0, 0, w, h, GL_COLOR_BUFFER_BIT,
                      (w == (int32_t)m_rtxW && h == (int32_t)m_rtxH) ? GL_NEAREST : GL_LINEAR);
    if (m_chaiEyes && m_hudVisible && m_hudTex != 0){
        // The study text, drawn into the headset image over AMBF's eye.
        glBindFramebuffer(GL_FRAMEBUFFER, drawFbo);
        hudDraw(w, h);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
    glDeleteFramebuffers(1, &readFbo);
    glDeleteFramebuffers(1, &drawFbo);
    if (srgbWasOn){
        glEnable(GL_FRAMEBUFFER_SRGB);
    }

    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(sc.handle, &ri);
    return true;
}

namespace {
// /dev/shm/fivrs_hud_overlay, written by scripts/rtx_view/hud_overlay.py. Little endian.
//   0 u32 magic "FVHU"  4 u32 version  8 u64 seq (odd while writing)
//  16 u32 width  20 u32 height  24 u32 flags (bit 0: something to draw)  28 pad
//  64 RGBA8 image, straight alpha, bottom row first. Same pixels for both eyes.
const char* kHudPath = "/dev/shm/fivrs_hud_overlay";
const uint32_t kHudMagic = 0x55485646u;  // "FVHU"
const size_t kHudHeader = 64;
}  // namespace

bool afOpenXREyepiece::chaiRenderEyes(){
    if (!m_mainCamera || !m_camera || !m_frameState.shouldRender){
        return m_chaiFbo[0] != nullptr;
    }
    if (!m_chaiFbo[0]){
        for (int i = 0; i < 2; ++i){
            m_chaiFbo[i] = cFrameBuffer::create();
            m_chaiFbo[i]->setup(m_camera->getInternalCamera(), m_chaiW, m_chaiH, true, true, GL_RGBA);
        }
        m_chaiFront = new cWorld();
    }
    cCamera* scope = m_mainCamera->getInternalCamera();
    // Main camera's lens, so the microscope zoom (field of view) carries over.
    const double fovY = scope->getFieldViewAngleRad();
    const double nearZ = scope->getNearClippingPlane();
    const double farZ = scope->getFarClippingPlane();
    const double aspect = (double)m_chaiW / (double)m_chaiH;
    const double top = std::tan(0.5 * fovY) * nearZ;
    const double right = top * aspect;
    cTransform P;
    P.identity();
    const double C = -(farZ + nearZ) / (farZ - nearZ);
    const double D = -(2.0 * farZ * nearZ) / (farZ - nearZ);
    const double m[4][4] = {{nearZ / right, 0, 0, 0}, {0, nearZ / top, 0, 0}, {0, 0, C, D}, {0, 0, -1, 0}};
    P.set(m[0][0], m[1][0], m[2][0], m[3][0],
          m[0][1], m[1][1], m[2][1], m[3][1],
          m[0][2], m[1][2], m[2][2], m[3][2],
          m[0][3], m[1][3], m[2][3], m[3][3]);
    // Both eyes aim at the bone, as rtx_eyes does: focus = scope to volume centre.
    double focus = 5.0;
    if (m_camera->m_afWorld){
        afVolumeVec volumes = m_camera->m_afWorld->getVolumes();
        if (!volumes.empty() && volumes[0]){
            focus = std::max((volumes[0]->getLocalPos() - m_mainCamera->getGlobalPos()).length(), 0.5);
        }
    }

    cTransform backupLocal = m_camera->getLocalTransform();
    cTransform backupProj = m_camera->getInternalCamera()->m_projectionMatrix;
    bool backupCustom = m_camera->getInternalCamera()->m_useCustomProjectionMatrix;
    cWorld* backupFront = m_camera->getInternalCamera()->m_frontLayer;
    for (int eye = 0; eye < 2; ++eye){
        // Chai cameras look along local -X with +Y to the right. This camera's parent
        // is main_camera, so its local frame is the scope's.
        const double side = (eye == 0) ? -0.5 : 0.5;
        const double y = side * m_chaiBaseUnits;
        cMatrix3d R;
        R.setAxisAngleRotationRad(cVector3d(0, 0, 1), std::atan2(y, focus));
        cTransform eyeLocal;
        eyeLocal.set(cVector3d(0, y, 0), R);
        m_camera->getInternalCamera()->m_frontLayer = m_chaiFront;
        m_camera->getInternalCamera()->m_useCustomProjectionMatrix = true;
        m_camera->getInternalCamera()->m_projectionMatrix = P;
        m_camera->setLocalTransform(eyeLocal);
        m_chaiFbo[eye]->renderView();
    }
    m_camera->getInternalCamera()->m_frontLayer = backupFront;
    m_camera->getInternalCamera()->m_projectionMatrix = backupProj;
    m_camera->getInternalCamera()->m_useCustomProjectionMatrix = backupCustom;
    m_camera->setLocalTransform(backupLocal);

    // The eyepiece placement code works from the "RTX" image size.
    m_rtxW = m_chaiW;
    m_rtxH = m_chaiH;
    m_rtxActive = true;
    if (!m_chaiAnnounced){
        std::cout << "INFO! OpenXR eyepiece shows the CHAI3D microscope eyes (" << m_chaiW << "x" << m_chaiH
                  << ", fov " << fovY << " rad, focus " << focus << ")" << std::endl;
        m_chaiAnnounced = true;
    }
    return true;
}

bool afOpenXREyepiece::hudPoll(){
    const double now = steadySeconds();
    if (m_hudMap && now - m_hudLastStatSec > 0.5){
        m_hudLastStatSec = now;
        struct stat st{};
        if (::stat(kHudPath, &st) != 0 || (unsigned long)st.st_ino != m_hudInode){
            hudUnmap();
        }
    }
    if (!m_hudMap){
        if (now - m_hudLastStatSec < 0.5){
            return m_hudVisible;
        }
        m_hudLastStatSec = now;
        m_hudVisible = false;
        const int fd = ::open(kHudPath, O_RDONLY);
        if (fd < 0){
            return false;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0 || (size_t)st.st_size < kHudHeader){
            ::close(fd);
            return false;
        }
        void* map = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED){
            ::close(fd);
            return false;
        }
        m_hudFd = fd;
        m_hudMap = map;
        m_hudMapSize = (size_t)st.st_size;
        m_hudInode = (unsigned long)st.st_ino;
        m_hudSeq = 0;
    }
    const char* base = static_cast<const char*>(m_hudMap);
    uint32_t magic = 0, width = 0, height = 0, flags = 0;
    std::memcpy(&magic, base, 4);
    std::memcpy(&width, base + 16, 4);
    std::memcpy(&height, base + 20, 4);
    const uint64_t seq = loadSeq(base);
    const size_t bytes = (size_t)width * height * 4;
    if (magic != kHudMagic || width == 0 || height == 0 || kHudHeader + bytes > m_hudMapSize){
        hudUnmap();
        m_hudVisible = false;
        return false;
    }
    if ((seq & 1u) != 0 || seq == m_hudSeq){
        return m_hudVisible;
    }
    std::memcpy(&flags, base + 24, 4);
    m_hudPixels.resize(bytes);
    std::memcpy(m_hudPixels.data(), base + kHudHeader, bytes);
    if (loadSeq(base) != seq){
        return m_hudVisible;  // written meanwhile: take the next one
    }
    m_hudSeq = seq;
    m_hudVisible = (flags & 1u) != 0;
    if (!m_hudVisible){
        return false;
    }
    GLint savedTex = 0, savedPbo = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTex);
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &savedPbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    if (m_hudTex == 0 || m_hudTexW != width || m_hudTexH != height){
        if (m_hudTex != 0){
            glDeleteTextures(1, &m_hudTex);
        }
        glGenTextures(1, &m_hudTex);
        glBindTexture(GL_TEXTURE_2D, m_hudTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)width, (GLsizei)height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        m_hudTexW = width;
        m_hudTexH = height;
    }
    // Same unpack care as the RTX upload: CHAI leaves row length / skips set.
    const GLenum keys[] = {GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_PIXELS,
                           GL_UNPACK_SKIP_ROWS, GL_UNPACK_IMAGE_HEIGHT, GL_UNPACK_SKIP_IMAGES};
    const GLint packed[] = {4, 0, 0, 0, 0, 0};
    GLint saved[6] = {4, 0, 0, 0, 0, 0};
    for (int k = 0; k < 6; ++k){
        glGetIntegerv(keys[k], &saved[k]);
        glPixelStorei(keys[k], packed[k]);
    }
    glBindTexture(GL_TEXTURE_2D, m_hudTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)width, (GLsizei)height, GL_RGBA, GL_UNSIGNED_BYTE,
                    m_hudPixels.data());
    for (int k = 0; k < 6; ++k){
        glPixelStorei(keys[k], saved[k]);
    }
    glBindTexture(GL_TEXTURE_2D, (GLuint)savedTex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, (GLuint)savedPbo);
    return true;
}

void afOpenXREyepiece::hudUnmap(){
    if (m_hudMap){
        munmap(m_hudMap, m_hudMapSize);
        m_hudMap = nullptr;
    }
    if (m_hudFd >= 0){
        ::close(m_hudFd);
        m_hudFd = -1;
    }
    m_hudMapSize = 0;
    m_hudInode = 0;
}

void afOpenXREyepiece::hudDraw(int32_t width, int32_t height){
    // Fixed-function quad over the bound framebuffer; AMBF's GL state is put back after.
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    GLint program = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    GLint activeTex = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTex);
    glActiveTexture(GL_TEXTURE0);
    GLint savedTex = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTex);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glUseProgram(0);
    glViewport(0, 0, width, height);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_hudTex);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glColor4f(1.f, 1.f, 1.f, 1.f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.f, 0.f); glVertex2f(-1.f, -1.f);
    glTexCoord2f(1.f, 0.f); glVertex2f(1.f, -1.f);
    glTexCoord2f(1.f, 1.f); glVertex2f(1.f, 1.f);
    glTexCoord2f(0.f, 1.f); glVertex2f(-1.f, 1.f);
    glEnd();
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopAttrib();
    glBindTexture(GL_TEXTURE_2D, (GLuint)savedTex);
    glActiveTexture((GLenum)activeTex);
    glUseProgram((GLuint)program);
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
}

void afOpenXREyepiece::physicsUpdate(double){}
void afOpenXREyepiece::reset(){}

bool afOpenXREyepiece::close(){
    if (m_connect){
        // Still connecting: the thread cleans up whatever it gets.
        std::lock_guard<std::mutex> guard(m_connect->lock);
        m_connect->abandoned = true;
        if (m_connect->instance != XR_NULL_HANDLE){
            xrDestroyInstance(m_connect->instance);
            m_connect->instance = XR_NULL_HANDLE;
        }
    }
    rtxUnmap();
    hudUnmap();
    shutdownOpenXR();
    return true;
}
