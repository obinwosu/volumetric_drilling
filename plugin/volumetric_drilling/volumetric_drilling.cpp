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

    \author    <amunawar@wpi.edu>
    \author    Adnan Munawar

    \author    <pkunjam1@jhu.edu>
    \author    Punit Kunjam
*/
//==============================================================================

#include "volumetric_drilling.h"
#include <algorithm>
#include <boost/program_options.hpp>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <dlfcn.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
std::atomic<int> g_drillingCamClutch{0};
afVolmetricDrillingPlugin* g_drillingPlugin = nullptr;

double rtxWallClock()
{
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
}

// Polled by ambf_openvr_plugin when setter linkage fails (RTLD_LOCAL plugins).
extern "C" int ambf_drilling_cam_clutch(void){
    return g_drillingCamClutch.load();
}

#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>

using namespace std;
namespace fs = std::filesystem;


//------------------------------------------------------------------------------
// DECLARED FUNCTIONS
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
namespace p_opt = boost::program_options;


afVolmetricDrillingPlugin::afVolmetricDrillingPlugin(){
    g_drillingPlugin = this;
}

int afVolmetricDrillingPlugin::init(int argc, char **argv, const afWorldPtr a_afWorld){
    p_opt::options_description cmd_opts("drilling_simulator Command Line Options");
    cmd_opts.add_options()
            ("info", "Show Info")
            ("nt", p_opt::value<int>()->default_value(8), "Number Tool Cursors to Load. Default 8")
            ("ds", p_opt::value<float>()->default_value(0.026), "Offset between shaft tool cursors. Default 0.026")
            ("vm", p_opt::value<string>()->default_value("00ShinyWhite.jpg"), "Volume's Matcap Filename (Should be placed in the ./resources/matcap/ folder)")
            ("dm", p_opt::value<string>()->default_value("dark_metal_brushed.jpg"), "Drill's Matcap Filename (Should be placed in ./resources/matcap/ folder)")
            ("fp", p_opt::value<string>()->default_value("/dev/input/js0"), "Footpedal joystick input file description E.g. /dev/input/js0)")
            ("mute", p_opt::value<bool>()->default_value(false), "Mute")
            ("gcdr", p_opt::value<double>()->default_value(30.0), "Gaze Calibration Marker Motion Duration")
            ("sb", p_opt::value<int>()->default_value(6), "Starting burr size in mm (1, 2, 4, or 6). Default 6")
            ("anatomy_adf", p_opt::value<string>()->default_value(""), "Path to anatomy ADF (for laterality)")
            ("session_user", p_opt::value<string>()->default_value(""), "Study session user / participant name")
            ("session_anatomy", p_opt::value<string>()->default_value(""), "Study session anatomy display name")
            ("startup_view", p_opt::value<string>()->default_value(""), "Canonical view id at launch (default|home|<id>)")
            ("rtx_view", p_opt::value<int>()->default_value(0), "Publish status for the optional RTX bone view");

    p_opt::variables_map var_map;
    p_opt::store(p_opt::command_line_parser(argc, argv).options(cmd_opts).allow_unregistered().run(), var_map);
    p_opt::notify(var_map);

    m_sessionUser = var_map["session_user"].as<string>();
    m_sessionAnatomy = var_map["session_anatomy"].as<string>();
    m_startupViewId = var_map["startup_view"].as<string>();
    m_rtxView = var_map["rtx_view"].as<int>() != 0;
    for (char& c : m_startupViewId){
        c = static_cast<char>(::tolower(c));
    }

    // Prefer explicit --anatomy_adf; fall back to AMBF -a if present in argv.
    string anatomyAdfPath = var_map["anatomy_adf"].as<string>();
    if (anatomyAdfPath.empty()){
        anatomyAdfPath = findAnatomyAdfPathFromArgs(argc, argv);
    }
    m_anatomyLaterality = readLateralityFromAdf(anatomyAdfPath);
    if (m_sessionAnatomy.empty() && !anatomyAdfPath.empty()){
        // Fallback: basename without .yaml
        string base = anatomyAdfPath;
        size_t slash = base.find_last_of("/\\");
        if (slash != string::npos){
            base = base.substr(slash + 1);
        }
        size_t dot = base.rfind(".yaml");
        if (dot != string::npos){
            base = base.substr(0, dot);
        }
        m_sessionAnatomy = base;
    }
    if (!m_anatomyLaterality.empty()){
        cerr << "INFO! Anatomy laterality=" << m_anatomyLaterality
             << " from " << anatomyAdfPath << endl;
    }
    else if (!anatomyAdfPath.empty()){
        cerr << "INFO! No laterality in anatomy ADF " << anatomyAdfPath << endl;
    }

    if(var_map.count("info")){
        std::cout<< cmd_opts << std::endl;
        return -1;
    }

    string file_path = __FILE__;
    string current_filepath = file_path.substr(0, file_path.rfind("/"));

    string volume_matcap = var_map["vm"].as<string>();
    string footpedal_fd = var_map["fp"].as<string>();

    m_zeroColor = cColorb(0x00, 0x00, 0x00, 0x00);

    m_boneColor = cColorb(255, 249, 219, 255);

    m_storedColor = cColorb(0x00, 0x00, 0x00, 0x00);

    m_worldPtr = a_afWorld;

    // Get first camera
    m_mainCamera = findAndAppendCamera("main_camera");
    m_cameraL = findAndAppendCamera("cameraL");
    m_cameraR = findAndAppendCamera("cameraR");
    m_stereoCamera = findAndAppendCamera("stereoLR");
    m_openvrCamera = findAndAppendCamera("openvr_cam");
    m_openxrEyepieceCamera = findAndAppendCamera("openxr_eyepiece");

    if (m_cameras.size() == 0){
        cerr << "ERROR! NO CAMERAS FOUND. " << endl;
        return -1;
    }

    if (!m_mainCamera){
        cerr << "INFO! FAILED TO LOAD main_camera, taking the first camera from world " << endl;
        m_mainCamera = m_worldPtr->getCameras()[0];
    }

    if (m_openvrCamera || m_openxrEyepieceCamera){
        // HMD plugins often load after simulator init — resolve clutch exports lazily.
        m_drillManager.m_vrHmdOwnsCamClutch = true;
        if (m_openxrEyepieceCamera){
            cerr << "INFO! OpenXR eyepiece present — cam-clutch gates HMD→main_camera" << endl;
        }
        if (m_openvrCamera){
            cerr << "INFO! OpenVR camera present — cam-clutch gates HMD→main_camera" << endl;
        }
    }

    if (m_stereoCamera){
        makeVRWindowFullscreen(m_stereoCamera);
    }

    m_panelManager.addCamera(m_mainCamera);
    if (m_stereoCamera){
        m_stereoCamera->getInternalCamera()->m_stereoOffsetW = 0.1;
        m_panelManager.addCamera(m_stereoCamera);
    }
    // HMD eyepiece cameras need panels too — pause menu / notices render into their FBOs.
    if (m_openvrCamera){
        m_panelManager.addCamera(m_openvrCamera);
    }
    if (m_openxrEyepieceCamera){
        m_panelManager.addCamera(m_openxrEyepieceCamera);
    }

    if (!m_drillManager.init(a_afWorld, &m_panelManager, var_map)){
        return -1;
    }

    m_volumeObject = m_worldPtr->getVolume("mastoidectomy_volume");
    if (!m_volumeObject){
        cerr << "ERROR! FAILED TO FIND VOLUME NAMED " << "mastoidectomy_volume" << endl;
        return -1;
    }
    else{
        m_voxelObj = m_volumeObject->getInternalVolume();

        m_maxVolCorner = m_voxelObj->m_maxCorner;
        m_minVolCorner = m_voxelObj->m_minCorner;
        m_maxTexCoord = m_voxelObj->m_maxTextureCoord;
        m_minTexCoord = m_voxelObj->m_minTextureCoord;

        m_textureCoordScale(0) = (m_maxTexCoord.x() - m_minTexCoord.x()) / (m_maxVolCorner.x() - m_minVolCorner.x());
        m_textureCoordScale(1) = (m_maxTexCoord.y() - m_minTexCoord.y()) / (m_maxVolCorner.y() - m_minVolCorner.y());
        m_textureCoordScale(2) = (m_maxTexCoord.z() - m_minTexCoord.z()) / (m_maxVolCorner.z() - m_minVolCorner.z());
    }

    // read the scale factor between the physical workspace of the haptic
    // device and the virtual workspace defined for the tool
    double workspaceScaleFactor = m_drillManager.m_toolCursorList[0]->getWorkspaceScaleFactor();

    // stiffness properties
    double maxStiffness = m_drillManager.m_hapticDevice->getSpecifications().m_maxLinearStiffness / workspaceScaleFactor;

    // Set voxels surface contact properties
    m_voxelObj->m_material->setStiffness(3.0*maxStiffness); // increasing haptic feedback (2.0 --> 3.0)
    m_voxelObj->m_material->setDamping(0.0);
    m_voxelObj->m_material->setDynamicFriction(0.0);
    m_voxelObj->setUseMaterial(true);

    initializeLabels();
    applyStereoCenteredHud();

    // Publish Volume Info
    cTransform volPose = m_volumeObject->getLocalTransform();
    cVector3d dims = m_volumeObject->getDimensions();
    cVector3d count = m_volumeObject->getVoxelCount();
    m_drillManager.m_drillingPub->setVolumeInfo(volPose, dims, count);
    m_drillManager.m_drillingPub->publishVolumeInfo(m_worldPtr->getCurrentTimeStamp());

    string volumeMatcapFilepath = current_filepath + "/../../resources/matcap/" + volume_matcap;
    cTexture2dPtr volMatCap = cTexture2d::create();
    if(volMatCap->loadFromFile(volumeMatcapFilepath)){
        m_volumeObject->getInternalVolume()->m_aoTexture = volMatCap;
        m_volumeObject->getInternalVolume()->m_aoTexture->setTextureUnit(GL_TEXTURE5);
        cerr << "SUCCESFULLY LOADED VOLUME'S MATCAP TEXTURE" << endl;
    }
    else{
        cerr << "FAILED TO LOAD VOLUME'S MATCAP TEXTURE" << endl;
    }



    // Dull gray room behind the microscope beam (same idea as the surreal sim).
    auto addRoomBackdrop = [this](afCameraPtr cam){
        if (!cam){
            return;
        }
        cBackground* background = new cBackground();
        cam->getBackLayer()->addChild(background);
        m_roomBackdrops.push_back(background);
    };
    addRoomBackdrop(m_mainCamera);
    addRoomBackdrop(m_stereoCamera);
    addRoomBackdrop(m_openvrCamera);
    addRoomBackdrop(m_openxrEyepieceCamera);
    applyRoomGray();
    // Hide legacy checkered floor / bright planes if an older world ADF still loads them.
    if (afRigidBodyPtr floor = m_worldPtr->getRigidBody("Plane", true)){
        if (floor->getVisualObject()){
            floor->getVisualObject()->setShowEnabled(false);
        }
    }

    if (m_footpedal.init(footpedal_fd) != -1){
        cerr << "INFO! FOUND FOOTPEDAL INTERFACE " << footpedal_fd << endl;
        cerr << "INFO! Pedal map (Ikkegol a/b/c): a=style, b=hold-to-run drill, c=size" << endl;
    }
    else{
        cerr << "WARNING! NO FOOTPEDAL /dev/input DEVICE OPEN "
             << "(using keyboard a/b/c, Space hold-drill, + key 1 cam-clutch)\n";
    }
    cerr << "INFO! Ikkegol pedals: a=diamond/cutting, b or Space=hold drill, c=size; key 1=cam clutch" << endl;

    m_gazeMarkerController.init(m_worldPtr, &m_panelManager);
    cerr << "INFO! GAZE MARKER INITIALIZED \n";
    m_timeSinceLastCalibration = 0.0;

    m_originalCameraTransform = m_mainCamera->getLocalTransform();
    m_volumeLookAt = m_volumeObject->getLocalTransform().getLocalPos();
    m_cameraHomeUp = m_mainCamera->getUpVector();
    loadCanonicalViews();
    applyStartupView();
    cFontPtr font = NEW_CFONTCALIBRI36();
    m_calibrationWarningLabel = new cLabel(font);
    m_calibrationWarningLabel->m_fontColor.setBlack();
    m_calibrationWarningLabel->setCornerRadius(10, 10, 10, 10);
    m_calibrationWarningLabel->setColor(cColorf(1., 1., 0.2));
    m_calibrationWarningLabel->setTransparencyLevel(0.8);
    m_calibrationWarningLabel->setShowPanel(true);
    m_calibrationWarningLabel->setShowEnabled(false);
    m_panelManager.addPanel(m_calibrationWarningLabel, 0.5, 0.5, 
                        PanelReferenceOrigin::CENTER, PanelReferenceType::NORMALIZED);
    m_panelManager.setVisible(m_calibrationWarningLabel, false);
    m_isCountdownActive = false;
    m_countdownTimer = 0.0;
    return 1;
}

void afVolmetricDrillingPlugin::publishDrillVisualState()
{
    namespace fs = std::filesystem;
    const fs::path dir = "/tmp/fivrs-rtx";
    std::error_code mkdirError;
    fs::create_directories(dir, mkdirError);
    const char* style = (m_drillManager.m_burrStyle == BurrStyle::DIAMOND) ? "diamond" : "cutting";
    int sizeMm = 0;
    if (m_drillManager.m_activeDrill){
        sizeMm = static_cast<int>(std::lround(
            m_drillManager.m_activeDrill->m_size / m_drillManager.m_units_mmToSim));
    }
    cVector3d tip(0.0, 0.0, 0.0);
    cQuaternion tipRot;
    tipRot.x = 0.0;
    tipRot.y = 0.0;
    tipRot.z = 0.0;
    tipRot.w = 1.0;
    if (m_drillManager.m_activeDrill && m_drillManager.m_activeDrill->m_rigidBody){
        cTransform pose = m_drillManager.m_activeDrill->m_rigidBody->getLocalTransform();
        tip = pose.getLocalPos();
        tipRot.fromRotMat(pose.getLocalRot());
    }
    // q and -q are the same rotation. Flipping the hemisphere each frame
    // makes the RTX shaft whip back and forth when the pose is interpolated.
    static cQuaternion previousQuat;
    static bool havePreviousQuat = false;
    if (havePreviousQuat){
        const double dot = tipRot.x * previousQuat.x + tipRot.y * previousQuat.y
            + tipRot.z * previousQuat.z + tipRot.w * previousQuat.w;
        if (dot < 0.0){
            tipRot.x = -tipRot.x;
            tipRot.y = -tipRot.y;
            tipRot.z = -tipRot.z;
            tipRot.w = -tipRot.w;
        }
    }
    previousQuat = tipRot;
    havePreviousQuat = true;
    const fs::path finalPath = dir / "drill_state.json";
    const fs::path tempPath = dir / "drill_state.json.tmp";
    {
        std::ofstream out(tempPath, std::ios::trunc);
        if (!out){
            return;
        }
        out << std::setprecision(8)
            << "{\"on\":" << (m_drillManager.m_isOn ? "true" : "false")
            << ",\"style\":\"" << style << "\""
            << ",\"size_mm\":" << sizeMm
            << ",\"px\":" << tip.x() << ",\"py\":" << tip.y() << ",\"pz\":" << tip.z()
            << ",\"qx\":" << tipRot.x << ",\"qy\":" << tipRot.y
            << ",\"qz\":" << tipRot.z << ",\"qw\":" << tipRot.w << "}\n";
    }
    std::error_code renameError;
    fs::rename(tempPath, finalPath, renameError);
}

void afVolmetricDrillingPlugin::publishRtxStatus(double now)
{
    namespace fs = std::filesystem;
    const fs::path dir = "/tmp/fivrs-rtx";
    std::error_code mkdirError;
    fs::create_directories(dir, mkdirError);
    const uint64_t removed = (m_drillManager.m_drillingPub != nullptr)
        ? m_drillManager.m_drillingPub->removedCount()
        : 0;
    {
        std::ofstream out(dir / "status.json", std::ios::trunc);
        if (out){
            out << "{\n"
                << "  \"graphics_hz\": " << (m_worldPtr ? m_worldPtr->getGraphicsFrequency() : 0) << ",\n"
                << "  \"physics_hz\": " << (m_worldPtr ? m_worldPtr->getPhysicsFrequency() : 0) << ",\n"
                << "  \"removed\": " << removed << ",\n"
                << "  \"wall_time\": " << std::setprecision(17) << now << "\n"
                << "}\n";
        }
    }

    const fs::path request = dir / "resync.request";
    if (!fs::exists(request)){
        return;
    }
    if (dumpOccupancy((dir / "occupancy.bin").string())){
        fs::remove(request);
    }
}

bool afVolmetricDrillingPlugin::loadRtxTrial(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    uint32_t magic = 0;
    uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in || magic != 0x51455346u || count > 10000u){
        return false;
    }
    m_rtxTrial.clear();
    m_rtxTrial.reserve(count);
    for (uint32_t batchIndex = 0; batchIndex < count; ++batchIndex){
        uint32_t idLen = 0;
        in.read(reinterpret_cast<char*>(&idLen), sizeof(idLen));
        if (!in || idLen > 128u){
            return false;
        }
        RtxTrialBatch batch;
        batch.id.assign(idLen, '\0');
        in.read(batch.id.data(), idLen);
        in.read(reinterpret_cast<char*>(&batch.release_s), sizeof(batch.release_s));
        uint32_t voxels = 0;
        in.read(reinterpret_cast<char*>(&voxels), sizeof(voxels));
        if (!in || voxels > 2000000u){
            return false;
        }
        const std::string marker = "|drop";
        if (batch.id.size() > marker.size() && batch.id.compare(batch.id.size() - marker.size(), marker.size(), marker) == 0){
            batch.drop = true;
            batch.id.resize(batch.id.size() - marker.size());
        }
        batch.indices.resize(static_cast<size_t>(voxels) * 3u);
        if (!batch.indices.empty()){
            in.read(reinterpret_cast<char*>(batch.indices.data()), batch.indices.size() * sizeof(int));
        }
        if (!in){
            return false;
        }
        m_rtxTrial.push_back(std::move(batch));
    }
    return true;
}

bool afVolmetricDrillingPlugin::dumpOccupancy(const std::string& path)
{
    if (m_voxelObj == nullptr || m_voxelObj->m_texture == nullptr){
        return false;
    }
    cMultiImagePtr image = dynamic_pointer_cast<cMultiImage>(m_voxelObj->m_texture->m_image);
    if (image == nullptr || image->getArray() == nullptr){
        return false;
    }
    const int width = image->getWidth();
    const int height = image->getHeight();
    const int depth = static_cast<int>(image->getImageCount());
    const int bytesPerSlice = static_cast<int>(image->getSizeInBytes());
    const int voxels = width * height;
    const int bpp = (voxels > 0) ? bytesPerSlice / voxels : 0;
    std::ofstream dump(path, std::ios::binary | std::ios::trunc);
    if (!dump){
        return false;
    }
    const int32_t header[4] = {width, height, depth, bpp};
    dump.write(reinterpret_cast<const char*>(header), sizeof(header));
    dump.write(reinterpret_cast<const char*>(image->getArray()),
               static_cast<std::streamsize>(bytesPerSlice) * depth);
    const bool wrote = static_cast<bool>(dump);
    dump.close();
    cerr << "INFO! RTX occupancy dump " << width << "x" << height << "x" << depth << endl;
    return wrote;
}

void afVolmetricDrillingPlugin::serviceRtxRestore()
{
    if (!m_rtxView || m_volumeObject == nullptr){
        return;
    }
    namespace fs = std::filesystem;
    const fs::path dir = "/tmp/fivrs-rtx";
    if (m_rtxRestoreWait > 0){
        m_rtxRestoreWait -= 1;
        if (m_rtxRestoreWait == 0){
            std::error_code mkdirError;
            fs::create_directories(dir, mkdirError);
            if (dumpOccupancy((dir / "occupancy.bin").string())){
                std::ofstream done(dir / "restore.done", std::ios::trunc);
                done << std::setprecision(17) << rtxWallClock() << "\n";
                cerr << "INFO! RTX trial volume restored" << endl;
            }
        }
    }
    if (!fs::exists(dir / "restore.request")){
        return;
    }
    fs::remove(dir / "restore.request");
    fs::remove(dir / "delete_sequence.bin");
    fs::remove(dir / "catchup.request");
    fs::remove(dir / "catchup.bin");
    m_rtxTrialLoaded = false;
    m_rtxTrial.clear();
    m_rtxPending.clear();
    m_rtxHistory.clear();
    if (m_drillManager.m_drillingPub != nullptr){
        m_drillManager.m_drillingPub->resetRemovedCount();
    }
    m_flagMarkVolumeForUpdate = false;
    m_volumeUpdate.setEmpty();
    // updateSceneObjects runs before this graphics update, so the texture
    // copy lands at the start of the next frame. The dump waits for that.
    m_volumeObject->reset();
    m_rtxRestoreWait = 1;
}

void afVolmetricDrillingPlugin::serviceRtxTrial()
{
    if (!m_rtxView || m_rtxRestoreWait > 0 || m_voxelObj == nullptr || m_voxelObj->m_texture == nullptr){
        return;
    }
    if (m_drillManager.m_drillingPub == nullptr){
        return;
    }
    namespace fs = std::filesystem;
    const fs::path path = "/tmp/fivrs-rtx/delete_sequence.bin";
    if (!m_rtxTrialLoaded){
        m_rtxTrialScan = (m_rtxTrialScan + 1) % 16;
        if (m_rtxTrialScan != 0 || !fs::exists(path)){
            return;
        }
        if (!loadRtxTrial(path.string())){
            fs::remove(path);
            cerr << "WARNING! RTX trial file was not readable" << endl;
            return;
        }
        m_rtxTrialNext = 0;
        m_rtxTrialStart = rtxWallClock();
        m_rtxTrialLoaded = true;
        cerr << "INFO! RTX trial uses the live graphics-frame batch" << endl;
    }
    if (m_rtxTrialNext >= m_rtxTrial.size()){
        fs::remove(path);
        m_rtxTrialLoaded = false;
        m_rtxTrial.clear();
        return;
    }
    const double now = rtxWallClock();
    const RtxTrialBatch& batch = m_rtxTrial[m_rtxTrialNext];
    if (now + 1e-4 < m_rtxTrialStart + batch.release_s){
        return;
    }
    cMultiImagePtr image = dynamic_pointer_cast<cMultiImage>(m_voxelObj->m_texture->m_image);
    if (image == nullptr){
        return;
    }
    const int width = image->getWidth();
    const int height = image->getHeight();
    const int depth = static_cast<int>(image->getImageCount());
    int cleared = 0;
    m_mutexVoxel.acquire();
    for (size_t i = 0; i + 2 < batch.indices.size(); i += 3){
        const int x = batch.indices[i];
        const int y = batch.indices[i + 1];
        const int z = batch.indices[i + 2];
        if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth){
            continue;
        }
        cColorb colorb;
        image->getVoxelColor(static_cast<uint>(x), static_cast<uint>(y), static_cast<uint>(z), colorb);
        if (colorb == m_zeroColor){
            continue;
        }
        image->setVoxelColor(static_cast<uint>(x), static_cast<uint>(y), static_cast<uint>(z), m_zeroColor);
        m_volumeUpdate.enclose(cVector3d(x, y, z));
        enqueueRemoved(x, y, z, batch.id, batch.drop);
        cleared += 1;
    }
    m_mutexVoxel.release();
    if (cleared > 0){
        m_flagMarkVolumeForUpdate = true;
    }
    cerr << "INFO! RTX trial " << batch.id << " cleared " << cleared
         << (batch.drop ? " (withheld from publish)" : "") << endl;
    m_rtxTrialNext += 1;
}

void afVolmetricDrillingPlugin::enqueueRemoved(int x, int y, int z, const std::string& trialId, bool silent)
{
    RtxPendingVoxel voxel;
    voxel.trialId = trialId;
    voxel.wallClear = rtxWallClock();
    voxel.x = x;
    voxel.y = y;
    voxel.z = z;
    voxel.silent = silent;
    m_rtxPending.push_back(voxel);
}

void afVolmetricDrillingPlugin::writeTrialLine(const std::string& id, int cleared, uint64_t batchId, double wallClear, double wallPublish, double clearMs, bool dropped)
{
    std::ofstream log("/tmp/fivrs-rtx/ambf_timing.jsonl", std::ios::app);
    log << std::setprecision(17)
        << "{\"id\":\"" << id << "\",\"n\":" << cleared
        << ",\"serial\":" << batchId
        << ",\"message_serial\":" << batchId
        << ",\"wall_clear\":" << wallClear
        << ",\"wall_publish\":" << (dropped ? 0.0 : wallPublish)
        << ",\"clear_ms\":" << clearMs
        << ",\"dropped\":" << (dropped ? "true" : "false")
        << ",\"mode\":\"live\"}\n";
}

void afVolmetricDrillingPlugin::serviceRtxFlush()
{
    if (!m_rtxView || m_drillManager.m_drillingPub == nullptr){
        return;
    }
    m_mutexVoxel.acquire();
    if (m_rtxPending.empty()){
        m_mutexVoxel.release();
        return;
    }
    std::vector<RtxPendingVoxel> pending;
    pending.swap(m_rtxPending);
    m_mutexVoxel.release();

    auto emit = [&](bool silent){
        std::vector<RtxPendingVoxel> group;
        for (const RtxPendingVoxel& voxel : pending){
            if (voxel.silent == silent){
                group.push_back(voxel);
            }
        }
        if (group.empty()){
            return;
        }
        const uint64_t batchId = m_rtxNextBatch++;
        RtxBatch batch;
        batch.id = batchId;
        batch.published = !silent;
        batch.indices.reserve(group.size() * 3u);
        m_drillManager.m_drillingPub->clearVoxelMsg();
        for (const RtxPendingVoxel& voxel : group){
            batch.indices.push_back(voxel.x);
            batch.indices.push_back(voxel.y);
            batch.indices.push_back(voxel.z);
            if (!silent){
                cVector3d index(voxel.x, voxel.y, voxel.z);
                cColorf color(0.0f, 0.0f, 0.0f);
                m_drillManager.m_drillingPub->appendToVoxelMsg(index, color);
            }
        }
        m_mutexVoxel.acquire();
        m_rtxHistory.push_back(batch);
        if (m_rtxHistory.size() > 8192u){
            m_rtxHistory.erase(m_rtxHistory.begin(), m_rtxHistory.begin() + static_cast<std::ptrdiff_t>(m_rtxHistory.size() - 8192u));
        }
        m_mutexVoxel.release();
        const double published = rtxWallClock();
        if (!silent){
            m_drillManager.m_drillingPub->publishVoxelBatch(m_worldPtr->getCurrentTimeStamp(), batchId);
            m_flagMarkVolumeForUpdate = true;
        }
        size_t cursor = 0;
        while (cursor < group.size()){
            const std::string id = group[cursor].trialId.empty() ? (silent ? "withheld" : "drill") : group[cursor].trialId;
            const double wallClear = group[cursor].wallClear;
            int count = 0;
            while (cursor < group.size()){
                const std::string nextId = group[cursor].trialId.empty() ? (silent ? "withheld" : "drill") : group[cursor].trialId;
                if (nextId != id){
                    break;
                }
                count += 1;
                cursor += 1;
            }
            writeTrialLine(id, count, batchId, wallClear, published, 0.0, silent);
        }
    };
    emit(false);
    emit(true);
}

void afVolmetricDrillingPlugin::serviceRtxCatchup()
{
    namespace fs = std::filesystem;
    const fs::path request = "/tmp/fivrs-rtx/catchup.request";
    if (!fs::exists(request)){
        return;
    }
    std::ifstream in(request);
    uint64_t first = 0;
    uint64_t last = 0;
    in >> first >> last;
    if (!in || last < first){
        fs::remove(request);
        return;
    }
    std::vector<RtxBatch> found;
    bool missing = false;
    if (last > first + 8192u){
        missing = true;
        last = first + 8192u;
    }
    m_mutexVoxel.acquire();
    for (uint64_t id = first; id <= last; ++id){
        const auto batch = std::find_if(m_rtxHistory.begin(), m_rtxHistory.end(), [id](const RtxBatch& item){ return item.id == id; });
        if (batch == m_rtxHistory.end()){
            missing = true;
        } else {
            found.push_back(*batch);
        }
    }
    m_mutexVoxel.release();
    const fs::path temporary = "/tmp/fivrs-rtx/catchup.tmp";
    const fs::path bin = "/tmp/fivrs-rtx/catchup.bin";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    const uint32_t magic = 0x48544346u;
    const uint32_t missingFlag = missing ? 1u : 0u;
    const uint32_t batchCount = static_cast<uint32_t>(found.size());
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&missingFlag), sizeof(missingFlag));
    out.write(reinterpret_cast<const char*>(&batchCount), sizeof(batchCount));
    for (const RtxBatch& batch : found){
        const uint64_t id = batch.id;
        const uint32_t count = static_cast<uint32_t>(batch.indices.size() / 3u);
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        if (count > 0){
            out.write(reinterpret_cast<const char*>(batch.indices.data()), batch.indices.size() * sizeof(int));
        }
    }
    out.close();
    fs::remove(bin);
    fs::rename(temporary, bin);
    fs::remove(request);
    cerr << "INFO! RTX catch-up batches " << first << "-" << last
         << " found " << found.size() << (missing ? " (history miss)" : "") << endl;
}


void afVolmetricDrillingPlugin::graphicsUpdate(){
    serviceRtxRestore();
    serviceRtxCatchup();
    serviceRtxFlush();
    serviceRtxCamera();

    // update region of voxels to be updated
    if (m_flagMarkVolumeForUpdate)
    {
        m_mutexVoxel.acquire();
        cVector3d min = m_volumeUpdate.m_min;
        cVector3d max = m_volumeUpdate.m_max;
        m_volumeUpdate.setEmpty();
        m_mutexVoxel.release();
        ((cTexture3d*)m_voxelObj->m_texture.get())->markForPartialUpdate(min, max);
        m_flagMarkVolumeForUpdate = false;
    }
    m_volumeObject->getShaderProgram()->setUniformi("uMatcapMap", C_TU_AO);
    m_volumeObject->getShaderProgram()->setUniformi("shadowMap", C_TU_SHADOWMAP);
    m_volumeObject->getShaderProgram()->setUniformf("uAlphaThreshold", 0.85);
    m_volumeObject->getShaderProgram()->setUniformi("uEnableDVR", 1);
    applyVolumeLightingUniforms();

    static double last_time = 0.0;
    double now = m_worldPtr->getWallTime();
    double dt = now - last_time;
    last_time = now;

    if (m_rtxView){
        if (!m_rtxWindowSized && m_mainCamera && m_mainCamera->m_window){
            glfwSetWindowSize(m_mainCamera->m_window, 1280, 720);
            m_rtxWindowSized = true;
        }
        if (dt > 0.0 && dt < 0.5){
            m_rtxStatusTimer += dt;
        }
        if (m_rtxStatusTimer >= 0.5){
            m_rtxStatusTimer = 0.0;
            publishRtxStatus(now);
        }
    }

    // While paused (VR HUD up), do not advance study timers or capture video frames.
    // Keep last_time current so unpausing does not apply a large catch-up dt.
    if (!m_simPaused){
        if (m_isRecording) {
            // m_timeSinceLastCalibration += dt; // Commenting this out only ensures one calibration sequence

            // Check if 320 seconds passed and not in countdown
            if (m_timeSinceLastCalibration >= 100 && !m_isCountdownActive) {
                cerr << "INFO! Starting calibration countdown" << endl;
                m_isCountdownActive = true;
                m_countdownTimer = 3.0; // 3-second countdown
                m_calibrationWarningLabel->setShowEnabled(true);
                m_calibrationWarningLabel->setShowPanel(true); // Show background
                m_panelManager.setVisible(m_calibrationWarningLabel, true);

            }
            m_videoRecordingController.update(m_worldPtr->getCurrentTimeStamp());
        }

        // Handle active countdown
        if (m_isCountdownActive) {
            m_countdownTimer -= dt;
            int remaining = static_cast<int>(ceil(m_countdownTimer));

            if (remaining > 0) {
                string text = "Calibration starting in " + to_string(remaining) + "...\n";
                text += "Returning to default view";
                m_panelManager.setText(m_calibrationWarningLabel, text);
            }
            else {
                // Countdown complete
                m_calibrationWarningLabel->setShowEnabled(false);
                m_calibrationWarningLabel->setShowPanel(false); // Show background
                m_panelManager.setVisible(m_calibrationWarningLabel, false);
                m_isCountdownActive = false;
                // m_timeSinceLastCalibration = 0.0; // Commenting this out only ensures one calibration sequence

                // Reset camera to original view
                m_mainCamera->setLocalTransform(m_originalCameraTransform);

                // Restart calibration sequence
                cerr << "INFO! Restarting gaze calibration" << endl;
                m_gazeMarkerController.restart();
            }
        }
        m_gazeMarkerController.update(dt);
    }
    if (m_noticeTimer > 0.0){
        m_noticeTimer -= dt;
        if (m_noticeTimer <= 0.0){
            m_panelManager.setVisible(m_noticeLabel, false);
        }
    }
    if (m_vrHudVisible){
        refreshVrHudText();
    }
    updateButtons();
    m_panelManager.update();
    if (m_rtxView){
        publishDrillVisualState();
    }
}

void afVolmetricDrillingPlugin::physicsUpdate(double dt){

    m_worldPtr->getChaiWorld()->computeGlobalPositions(true);

    // HUD pause: freeze drill pose (no haptic velocity) and skip voxel removal / forces.
    if (m_simPaused){
        if (m_drillManager.m_toolCursorList.size() > 0 && m_drillManager.m_toolCursorList[0]){
            m_drillManager.m_toolCursorList[0]->setDeviceLocalForce(0.0, 0.0, 0.0);
        }
        if (m_drillManager.m_audioSource && m_drillManager.m_audioState == AudioState::PLAYING){
            m_drillManager.m_audioSource->stop();
            m_drillManager.m_audioState = AudioState::STOPPED;
        }
        return;
    }

    serviceRtxTrial();
    m_drillManager.update(dt);

    bool anyCursorInContact = false;
    if (m_voxelObj){
        for (int i = 0; i < (int)m_drillManager.m_toolCursorList.size(); i++){
            if (m_drillManager.m_toolCursorList[i] &&
                m_drillManager.m_toolCursorList[i]->isInContact(m_voxelObj)){
                anyCursorInContact = true;
                break;
            }
        }
    }
    m_drillManager.tickVolumeContactSuppression(anyCursorInContact);
    const bool skipContact = m_drillManager.isVolumeContactSuppressed();

    // Sample the wall before this step deletes it. Clearing the contacted voxels
    // first made the restoring force fall to zero on the same step, so the
    // drill felt like it punched through and caught again.
    for(int i = 0 ; i < m_drillManager.m_toolCursorList.size() ; i++){
        m_drillManager.m_toolCursorList[i]->computeInteractionForces();
    }

    if (!skipContact && m_drillManager.m_toolCursorList[0]->isInContact(m_voxelObj) && m_drillManager.m_targetToolCursorIdx == 0 /*&& (userSwitches == 2)*/)
    {

        // retrieve contact event
        cCollisionEvent* contact = m_drillManager.m_toolCursorList[0]->m_hapticPoint->getCollisionEvent(0);

        cVector3d orig(contact->m_voxelIndexX, contact->m_voxelIndexY, contact->m_voxelIndexZ); // This is the closest voxel index to the drill tip

        m_voxelObj->m_texture->m_image->getVoxelColor(uint(orig.x()), uint(orig.y()), uint(orig.z()), m_storedColor);

        if (m_storedColor != m_zeroColor){
            if (m_drillManager.m_isOn){
                int removalCount = cMin(m_drillManager.m_activeDrill->getVoxelRemovalThreshold(), (int)contact->m_events.size());
                m_drillManager.m_drillingPub->clearVoxelMsg();
                m_mutexVoxel.acquire();
                for (int cIdx = 0 ; cIdx < removalCount ; cIdx++){
                    cVector3d ct(contact->m_events[cIdx].m_voxelIndexX, contact->m_events[cIdx].m_voxelIndexY, contact->m_events[cIdx].m_voxelIndexZ);
                    cColorb colorb;
                    m_voxelObj->m_texture->m_image->getVoxelColor(uint(ct.x()), uint(ct.y()), uint(ct.z()), colorb);
                    cColorf colorf = colorb.getColorf();
                    if (m_rtxView){
                        enqueueRemoved(static_cast<int>(ct.x()), static_cast<int>(ct.y()), static_cast<int>(ct.z()), "", false);
                    } else {
                        m_drillManager.m_drillingPub->appendToVoxelMsg(ct, colorf);
                    }
                    m_voxelObj->m_texture->m_image->setVoxelColor(uint(ct.x()), uint(ct.y()), uint(ct.z()), m_zeroColor);
                    m_volumeUpdate.enclose(cVector3d(uint(ct.x()), uint(ct.y()), uint(ct.z())));
                }
                m_mutexVoxel.release();

                // RTX publishes the accumulated voxels once per graphics frame.
                if (!m_rtxView){
                    m_drillManager.m_drillingPub->publishVoxelMsg(m_worldPtr->getCurrentTimeStamp());
                }
            }

            //Publisher for voxels removed
            if(m_storedColor != m_boneColor)
            {
                m_panelManager.setVisible(m_warningLabel, true);
            }
        }

        // mark voxel for update

        m_flagMarkVolumeForUpdate = true;
    }
    // remove warning panel
    else
    {
        m_panelManager.setVisible(m_warningLabel, false);
    }
    // Tip and shaft forces are kept together. Sending only the cursor with the
    // larger proxy gap dropped the tip force whenever the shaft barely won.
    cVector3d force(0.0, 0.0, 0.0);
    if (!skipContact){
        cVector3d local(0.0, 0.0, 0.0);
        for(int i = 0 ; i < m_drillManager.m_toolCursorList.size() ; i++){
            local.add(m_drillManager.m_toolCursorList[i]->getDeviceLocalForce());
        }
        force = cTranspose(m_mainCamera->getLocalRot()) * local;
        const double kContactForce = 0.05;
        const double kHoldSeconds = 0.012;
        if (force.length() >= kContactForce){
            m_heldHapticForce = force;
            m_hapticHoldSeconds = kHoldSeconds;
        }
        else if (m_hapticHoldSeconds > 0.0){
            force = m_heldHapticForce * (m_hapticHoldSeconds / kHoldSeconds);
            m_hapticHoldSeconds = cMax(0.0, m_hapticHoldSeconds - dt);
        }
        if (m_drillManager.m_isOn){
            force += (cVector3d(1.0, 1.0, 1.0) * m_waveGenerator.generate(dt));
        }
        double maxF = m_drillManager.m_hapticDevice->getSpecifications().m_maxLinearForce;
        if (maxF > 0.5 && force.length() > maxF){
            force.mul(maxF / force.length());
        }
    }
    else{
        m_hapticHoldSeconds = 0.0;
    }

    m_drillManager.m_toolCursorList[0]->setDeviceLocalForce(force);
    m_drillManager.m_drillingPub->publishForceFeedback(force, force, m_worldPtr->getCurrentTimeStamp());
    double maxF = m_drillManager.m_hapticDevice->getSpecifications().m_maxLinearForce;
    double force_mag = cClamp(force.length(), 0.0, maxF);
    m_drillManager.setAudioPitch(3.0 - force_mag / maxF);

    if (m_flagStart)
    {
        if (force.length() != 0.0)
        {
            m_counter = 0;
        }
        else
        {
            m_counter++;
            if (m_counter > 10)
                m_flagStart = false;
        }
    }
    else
    {
        if (force.length() > 10.0)
        {
            m_flagStart = true;
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // FINALIZE
    /////////////////////////////////////////////////////////////////////////

    // send forces to haptic device
    if (m_drillManager.getOverrideControl() == false){
        m_drillManager.m_toolCursorList[0]->applyToDevice();
    }

}



void afVolmetricDrillingPlugin::sliceVolume(int axisIdx, double delta)
{
    string axis_str = "";
    if (axisIdx == 0){
        axis_str = "X";
    }
    else if (axisIdx == 1){
        axis_str = "Y";
    }
    else if (axisIdx == 2){
        axis_str = "Z";
    }
    else{
        cerr << "ERROR! Volume axis index should be either 0, 1 or 2" << endl;
        return;
    }

    string delta_dir_str = "";
    if (delta > 0){
        delta_dir_str = "Increase";
    }
    else{
        delta_dir_str = "Decrease";
    }

    double value = cClamp((m_voxelObj->m_maxCorner(axisIdx) + delta), 0.01, m_maxVolCorner(axisIdx));
    m_voxelObj->m_maxCorner(axisIdx) = value;
    m_voxelObj->m_minCorner(axisIdx) = -value;
    m_voxelObj->m_maxTextureCoord(axisIdx) = 0.5 + value * m_textureCoordScale(axisIdx);
    m_voxelObj->m_minTextureCoord(axisIdx) = 0.5 - value * m_textureCoordScale(axisIdx);

    cerr << "> " << delta_dir_str << " Volume size along " << axis_str << " axis.                            \r";
}


void afVolmetricDrillingPlugin::makeVRWindowFullscreen(afCameraPtr vrCam, int monitor_number)
{
    int count;
    GLFWmonitor** monitors = glfwGetMonitors(&count);

    if (monitor_number >= 0){
        if (monitor_number >= count){
            cerr << "WARNING! SPECIFIED MONITOR NUMBER " << monitor_number <<
                    " FOR VR IS GREATER THAN NUMBER OF DISPLAYS. IGNORING" << endl;
            return;
        }
        vrCam->m_monitor = monitors[monitor_number];
    }

    for (int cnt = 0 ; cnt < count ; cnt++){
        string monitor_name = glfwGetMonitorName(monitors[cnt]);
        cerr << "\t Monitor Number: " << cnt << " | Name: " << monitor_name << endl;
    }


    const GLFWvidmode* mode = glfwGetVideoMode(vrCam->m_monitor);
    int w = 0.9 * mode->width;
    int h = 0.9 * mode->height;
    int x = 0.7 * (mode->width - w);
    int y = 0.7 * (mode->height - h);
    int xpos, ypos;
    glfwGetMonitorPos(vrCam->m_monitor, &xpos, &ypos);
    x += xpos; y += ypos;
    glfwSetWindowPos(vrCam->m_window, x, y);
    glfwSetWindowSize(vrCam->m_window, w, h);
    vrCam->m_width = w;
    vrCam->m_height = h;
    glfwSwapInterval(0);
    cerr << "\t\t Setting the Window on the VR Monitor to fullscreen \n" ;
}

void afVolmetricDrillingPlugin::updateButtons(){
    m_footpedal.poll();

    // Ikkegol footpedal workaround: left/middle/right emit keyboard a/b/c.
    // Poll via X11 so this works even when /dev/input is not readable.
    const bool xStyleHeld = isLetterKeyHeld(GLFW_KEY_A);
    const bool xDrillHeld = isLetterKeyHeld(GLFW_KEY_B);
    const bool xSizeHeld  = isLetterKeyHeld(GLFW_KEY_C);

    const bool styleEdge = (!m_xFootStylePrev && xStyleHeld);
    const bool sizeEdge  = (!m_xFootSizePrev && xSizeHeld);
    m_xFootStylePrev = xStyleHeld;
    m_xFootSizePrev = xSizeHeld;

    if (m_footpedal.isChangeBurrStylePressed() || styleEdge){
        m_drillManager.toggleBurrStyle();
    }

    if (m_footpedal.isChangeBurrSizePressed() || sizeEdge){
        m_drillManager.cycleDrillTypes();
    }

    // Hold-to-run drill: device pedal, keyboard 'b', or Space (RTX window included).
    m_drillManager.m_isOn = m_footpedal.isDrillOn() || xDrillHeld || isDrillHoldRequested();
    m_drillManager.updateBurrLabel();

    m_drillManager.m_camClutch = false;
    m_drillManager.m_deviceClutch = false;
    m_drillManager.m_camOrbitBurr = false;

    if (m_footpedal.isAvailable()){
        m_drillManager.m_camClutch = m_footpedal.isCamClutchPressed();
        m_drillManager.m_deviceClutch = m_footpedal.isDeviceClutchPressed();

    }

    // SayoDevice macro key programmed as "1": hold = camera clutch.
    if (isDigit1Held()){
        m_drillManager.m_camClutch = true;
    }

    // Hold Left Alt = cam-clutch when HMD is present (works without Geomagic).
    if ((m_openxrEyepieceCamera || m_openvrCamera) && isAltHeld()){
        m_drillManager.m_camClutch = true;
    }

    if (m_drillManager.m_hapticDevice->isDeviceAvailable()){
        bool btn1 = false; // stylus button 1 (cam-clutch switch index 1)
        bool btn2 = false; // stylus button 2 (device-clutch switch index 0)
        m_drillManager.m_hapticDevice->getUserSwitch(1, btn1);
        m_drillManager.m_hapticDevice->getUserSwitch(0, btn2);

        const bool ctrlHeld = isCtrlHeld();
        const bool shiftHeld = isShiftHeld();

        if (ctrlHeld){
            // Ctrl + button 2 = zoom/dolly in; Ctrl + button 1 = zoom/dolly out.
            // Hold for continuous adjust; do not treat as clutch / double-click.
            const double zoomStep = 0.08;
            if (btn2){
                dollyCamera(zoomStep);
            }
            if (btn1){
                dollyCamera(-zoomStep);
            }
            m_btn1Prev = btn1;
            m_btn2Prev = btn2;
        }
        else{
            // Double-click button 1 toggles pause. Required for VR (no keyboard).
            // Use glfw time, not AMBF wall clock — wall clock is rewound on resume.
            const double doubleClickSec = 0.45;
            double now = glfwGetTime();

            if (shiftHeld){
                if (m_vrHudVisible){
                    // Paused: Shift + button 2 applies the highlighted canonical view.
                    if (btn2 && !m_btn2Prev){
                        applyHighlightedPauseView();
                        refreshVrHudText();
                    }
                }
                else{
                    // Shift + button 2: orbit around the burr.
                    m_drillManager.m_camOrbitBurr |= btn2;
                }
                m_btn1Prev = btn1;
                m_btn2Prev = btn2;
                m_btn1LastClickTime = -1.0;
                m_btn2LastClickTime = -1.0;
            }
            else{
                if (btn1 && !m_btn1Prev){
                    if (m_btn1LastClickTime >= 0.0 && (now - m_btn1LastClickTime) <= doubleClickSec){
                        setVrHudVisible(!m_vrHudVisible);
                        m_btn1LastClickTime = -1.0;
                    }
                    else{
                        m_btn1LastClickTime = now;
                    }
                }
                m_btn1Prev = btn1;

                if (m_vrHudVisible){
                    // Paused: button 2 tap cycles the view list (headset, no keyboard).
                    if (btn2 && !m_btn2Prev){
                        cyclePauseViewHighlight(1);
                        refreshVrHudText();
                    }
                    m_btn2Prev = btn2;
                    m_btn2LastClickTime = -1.0;
                }
                else{
                    // Double-tap button 2: snap drill burr into main_camera view.
                    bool btn2DoubleTapped = false;
                    if (btn2 && !m_btn2Prev){
                        if (m_btn2LastClickTime >= 0.0 && (now - m_btn2LastClickTime) <= doubleClickSec){
                            requestDrillReposition();
                            m_btn2LastClickTime = -1.0;
                            btn2DoubleTapped = true;
                        }
                        else{
                            m_btn2LastClickTime = now;
                        }
                    }
                    m_btn2Prev = btn2;
                    m_drillManager.m_camClutch |= btn1;
                    // Hold btn2 = device clutch; skip the double-tap edge so it does not
                    // briefly clutch while recentering.
                    if (btn2 && !btn2DoubleTapped){
                        m_drillManager.m_deviceClutch |= btn2;
                    }
                }
            }
        }
    }

    g_drillingCamClutch.store(m_drillManager.m_camClutch ? 1 : 0);

    auto linkClutch = [&](HmdCamClutchFn& fn, const char* symbol, const char* soName, const string& absSo){
        if (fn){
            return;
        }
        fn = reinterpret_cast<HmdCamClutchFn>(dlsym(RTLD_DEFAULT, symbol));
        if (fn){
            cerr << "INFO! Linked " << symbol << endl;
            return;
        }
        const char* paths[] = { soName, absSo.c_str() };
        for (const char* path : paths){
            void* handle = dlopen(path, RTLD_NOLOAD | RTLD_LAZY);
            if (!handle){
                handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
            }
            if (!handle){
                continue;
            }
            fn = reinterpret_cast<HmdCamClutchFn>(dlsym(handle, symbol));
            if (fn){
                cerr << "INFO! Linked " << symbol << " via " << path << endl;
                return;
            }
        }
    };

    const string home = getenv("HOME") ? getenv("HOME") : "";
    if (m_openvrCamera){
        linkClutch(m_openvrSetCamClutch, "ambf_openvr_set_cam_clutch", "libambf_openvr_camera.so",
                   home + "/ros_ambf_ws/src/ambf_openvr_plugin/build/libambf_openvr_camera.so");
        if (m_openvrSetCamClutch){
            m_openvrSetCamClutch(m_drillManager.m_camClutch ? 1 : 0);
        }
    }
    if (m_openxrEyepieceCamera){
        linkClutch(m_openxrSetCamClutch, "ambf_openxr_set_cam_clutch", "libambf_openxr_eyepiece.so",
                   home + "/ros_ambf_ws/src/ambf_openxr_eyepiece/build/libambf_openxr_eyepiece.so");
        if (m_openxrSetCamClutch){
            m_openxrSetCamClutch(m_drillManager.m_camClutch ? 1 : 0);
        }
        ensureOpenxrPlacementFns();
    }
}

void afVolmetricDrillingPlugin::ensureOpenxrPlacementFns(){
    if (m_openxrRecenterToHead && m_openxrRecenterLap && m_openxrSetPlaceDrag){
        return;
    }
    const string home = getenv("HOME") ? getenv("HOME") : "";
    const string absSo = home + "/ros_ambf_ws/src/ambf_openxr_eyepiece/build/libambf_openxr_eyepiece.so";
    auto resolve = [&](const char* symbol) -> void* {
        void* sym = dlsym(RTLD_DEFAULT, symbol);
        if (sym){
            return sym;
        }
        for (const char* path : { "libambf_openxr_eyepiece.so", absSo.c_str() }){
            void* handle = dlopen(path, RTLD_NOLOAD | RTLD_LAZY);
            if (!handle){
                handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
            }
            if (!handle){
                continue;
            }
            sym = dlsym(handle, symbol);
            if (sym){
                cerr << "INFO! Linked " << symbol << " via " << path << endl;
                return sym;
            }
        }
        return nullptr;
    };
    if (!m_openxrRecenterToHead){
        m_openxrRecenterToHead = reinterpret_cast<OpenxrVoidFn>(resolve("ambf_openxr_recenter_to_head"));
    }
    if (!m_openxrRecenterLap){
        m_openxrRecenterLap = reinterpret_cast<OpenxrVoidFn>(resolve("ambf_openxr_recenter_lap"));
    }
    if (!m_openxrSetPlaceDrag){
        m_openxrSetPlaceDrag = reinterpret_cast<OpenxrIntFn>(resolve("ambf_openxr_set_place_drag"));
    }
}

bool afVolmetricDrillingPlugin::isCtrlHeld() const
{
    // Global X11 keymap: works when focus is on study_gui or another window.
    Display* dpy = glfwGetX11Display();
    if (dpy){
        char keys[32];
        XQueryKeymap(dpy, keys);
        auto keyDown = [&](KeySym sym) -> bool {
            KeyCode code = XKeysymToKeycode(dpy, sym);
            if (code == 0){
                return false;
            }
            return (keys[code / 8] & (1 << (code % 8))) != 0;
        };
        if (keyDown(XK_Control_L) || keyDown(XK_Control_R)){
            return true;
        }
    }

    // Fallback: any AMBF camera window that currently reports Ctrl pressed.
    auto winCtrl = [](GLFWwindow* win) -> bool {
        if (!win){
            return false;
        }
        return (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            || (glfwGetKey(win, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    };
    if (m_mainCamera && winCtrl(m_mainCamera->m_window)){
        return true;
    }
    if (m_stereoCamera && winCtrl(m_stereoCamera->m_window)){
        return true;
    }
    for (const auto& kv : m_cameras){
        if (kv.second && winCtrl(kv.second->m_window)){
            return true;
        }
    }
    return false;
}

bool afVolmetricDrillingPlugin::isShiftHeld() const
{
    Display* dpy = glfwGetX11Display();
    if (dpy){
        char keys[32];
        XQueryKeymap(dpy, keys);
        auto keyDown = [&](KeySym sym) -> bool {
            KeyCode code = XKeysymToKeycode(dpy, sym);
            if (code == 0){
                return false;
            }
            return (keys[code / 8] & (1 << (code % 8))) != 0;
        };
        if (keyDown(XK_Shift_L) || keyDown(XK_Shift_R)){
            return true;
        }
    }

    auto winShift = [](GLFWwindow* win) -> bool {
        if (!win){
            return false;
        }
        return (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            || (glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    };
    if (m_mainCamera && winShift(m_mainCamera->m_window)){
        return true;
    }
    if (m_stereoCamera && winShift(m_stereoCamera->m_window)){
        return true;
    }
    for (const auto& kv : m_cameras){
        if (kv.second && winShift(kv.second->m_window)){
            return true;
        }
    }
    return false;
}

bool afVolmetricDrillingPlugin::isAltHeld() const
{
    Display* dpy = glfwGetX11Display();
    if (dpy){
        char keys[32];
        XQueryKeymap(dpy, keys);
        auto keyDown = [&](KeySym sym) -> bool {
            KeyCode code = XKeysymToKeycode(dpy, sym);
            if (code == 0){
                return false;
            }
            return (keys[code / 8] & (1 << (code % 8))) != 0;
        };
        if (keyDown(XK_Alt_L) || keyDown(XK_Alt_R)){
            return true;
        }
    }

    auto winAlt = [](GLFWwindow* win) -> bool {
        if (!win){
            return false;
        }
        return (glfwGetKey(win, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
            || (glfwGetKey(win, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS);
    };
    if (m_mainCamera && winAlt(m_mainCamera->m_window)){
        return true;
    }
    if (m_stereoCamera && winAlt(m_stereoCamera->m_window)){
        return true;
    }
    for (const auto& kv : m_cameras){
        if (kv.second && winAlt(kv.second->m_window)){
            return true;
        }
    }
    return false;
}

bool afVolmetricDrillingPlugin::isXKeycodeHeld(unsigned int keycode) const
{
    if (keycode == 0){
        return false;
    }
    Display* dpy = glfwGetX11Display();
    if (!dpy){
        return false;
    }
    char keys[32];
    XQueryKeymap(dpy, keys);
    return (keys[keycode / 8] & (1 << (keycode % 8))) != 0;
}

bool afVolmetricDrillingPlugin::isDigit1Held() const
{
    Display* dpy = glfwGetX11Display();
    if (dpy){
        char keys[32];
        XQueryKeymap(dpy, keys);
        auto keyDown = [&](KeySym sym) -> bool {
            KeyCode code = XKeysymToKeycode(dpy, sym);
            if (code == 0){
                return false;
            }
            return (keys[code / 8] & (1 << (code % 8))) != 0;
        };
        // Main "1" key and keypad 1 — SayoDevice macro programmed as digit 1.
        if (keyDown(XK_1) || keyDown(XK_KP_1)){
            return true;
        }
    }

    auto win1 = [](GLFWwindow* win) -> bool {
        if (!win){
            return false;
        }
        return (glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS)
            || (glfwGetKey(win, GLFW_KEY_KP_1) == GLFW_PRESS);
    };
    if (m_mainCamera && win1(m_mainCamera->m_window)){
        return true;
    }
    if (m_stereoCamera && win1(m_stereoCamera->m_window)){
        return true;
    }
    for (const auto& kv : m_cameras){
        if (kv.second && win1(kv.second->m_window)){
            return true;
        }
    }
    return false;
}

bool afVolmetricDrillingPlugin::isDrillHoldRequested() const
{
    Display* dpy = glfwGetX11Display();
    if (dpy){
        KeyCode code = XKeysymToKeycode(dpy, XK_space);
        if (isXKeycodeHeld(code)){
            return true;
        }
    }
    auto winSpace = [](GLFWwindow* win) -> bool {
        return win && (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS);
    };
    if (m_mainCamera && winSpace(m_mainCamera->m_window)){
        return true;
    }
    if (m_stereoCamera && winSpace(m_stereoCamera->m_window)){
        return true;
    }
    for (const auto& kv : m_cameras){
        if (kv.second && winSpace(kv.second->m_window)){
            return true;
        }
    }
    std::error_code existsError;
    return fs::exists("/tmp/fivrs-rtx/drill_hold", existsError);
}

bool afVolmetricDrillingPlugin::isLetterKeyHeld(int glfwKey) const
{
    KeySym sym = 0;
    switch (glfwKey){
        case GLFW_KEY_A: sym = XK_a; break;
        case GLFW_KEY_B: sym = XK_b; break;
        case GLFW_KEY_C: sym = XK_c; break;
        default: return false;
    }

    Display* dpy = glfwGetX11Display();
    if (dpy){
        KeyCode code = XKeysymToKeycode(dpy, sym);
        if (code != 0){
            char keys[32];
            XQueryKeymap(dpy, keys);
            if ((keys[code / 8] & (1 << (code % 8))) != 0){
                return true;
            }
        }
    }

    auto winKey = [&](GLFWwindow* win) -> bool {
        return win && (glfwGetKey(win, glfwKey) == GLFW_PRESS);
    };
    if (m_mainCamera && winKey(m_mainCamera->m_window)){
        return true;
    }
    if (m_stereoCamera && winKey(m_stereoCamera->m_window)){
        return true;
    }
    for (const auto& kv : m_cameras){
        if (kv.second && winKey(kv.second->m_window)){
            return true;
        }
    }
    return false;
}

void afVolmetricDrillingPlugin::requestDrillReposition()
{
    if (!m_volumeObject || !m_mainCamera){
        showNotice("Cannot reposition drill (no volume or camera)");
        return;
    }

    cTransform poseFrom = m_drillManager.m_T_d;
    cVector3d dims = m_volumeObject->getDimensions();
    if (!m_drillManager.snapToCameraView(m_volumeLookAt, dims, m_volumeObject->getLocalTransform())){
        showNotice("Cannot reposition drill");
        return;
    }

    if (m_drillManager.m_drillingPub){
        m_drillManager.m_drillingPub->publishDrillReposition(
            poseFrom, m_drillManager.m_T_d, m_worldPtr->getCurrentTimeStamp());
    }
    showNotice("Drill repositioned");
    cerr << "INFO! DRILL REPOSITIONED INTO VIEW" << endl;
}

afCameraPtr afVolmetricDrillingPlugin::findAndAppendCamera(string cam_name){
    afCameraPtr cam = m_worldPtr->getCamera(cam_name);
    if (cam){
        cerr << "INFO! GOT CAMERA: " << cam->getName() << endl;
        m_cameras[cam_name] = cam;
    }
    else{
        cerr << "WARNING! CAMERA NOT FOUND " << cam_name << endl;
    }
    return cam;
}

void afVolmetricDrillingPlugin::setCameraRoll(double clockwiseDeg)
{
    m_activeCameraRollDeg = clockwiseDeg;
    m_customViewActive = true;
    applyCameraRoll(clockwiseDeg);
    cerr << "INFO! CAMERA ROLL " << clockwiseDeg << " deg CW about view axis" << endl;
}

void afVolmetricDrillingPlugin::applyCameraRoll(double clockwiseDeg)
{
    if (!m_mainCamera){
        return;
    }

    cVector3d pos = m_mainCamera->getLocalPos();
    cVector3d axis = m_volumeLookAt - pos;
    if (axis.length() < 1e-6){
        return;
    }
    axis.normalize();

    // Home up projected onto the plane perpendicular to the view axis
    cVector3d refUp = m_cameraHomeUp - axis * (m_cameraHomeUp.dot(axis));
    if (refUp.length() < 1e-6){
        cVector3d tmp = (fabs(axis.z()) < 0.9) ? cVector3d(0, 0, 1) : cVector3d(0, 1, 0);
        refUp = cCross(axis, tmp);
    }
    refUp.normalize();

    // Clockwise as seen looking toward the volume (negative right-hand rotation about the look axis)
    cMatrix3d rot;
    rot.setAxisAngleRotationDeg(axis, -clockwiseDeg);
    cVector3d up = rot * refUp;

    m_mainCamera->setView(pos, m_volumeLookAt, up);
}

void afVolmetricDrillingPlugin::nudgeCameraRoll(double clockwiseDeg)
{
    if (!m_mainCamera){
        return;
    }
    cVector3d pos = m_mainCamera->getLocalPos();
    cVector3d look = m_mainCamera->getLookVector();
    if (look.length() < 1e-6){
        return;
    }
    look.normalize();
    cVector3d up = m_mainCamera->getUpVector();
    cMatrix3d rot;
    rot.setAxisAngleRotationDeg(look, -clockwiseDeg);
    cVector3d newUp = rot * up;
    if (newUp.length() < 1e-6){
        return;
    }
    newUp.normalize();
    m_mainCamera->setView(pos, m_mainCamera->getTargetPosLocal(), newUp);
    m_activeCameraRollDeg += clockwiseDeg;
    m_customViewActive = true;
}

void afVolmetricDrillingPlugin::nudgeCameraOrbitYaw(double rightDeg)
{
    if (!m_mainCamera){
        return;
    }
    cVector3d pos = m_mainCamera->getLocalPos();
    cVector3d pivot = m_mainCamera->getTargetPosLocal();
    if ((pivot - pos).length() < 1e-3){
        pivot = m_volumeLookAt;
    }
    cVector3d up = m_mainCamera->getUpVector();
    if (up.length() < 1e-6){
        return;
    }
    up.normalize();
    cVector3d offset = pos - pivot;
    double dist = offset.length();
    if (dist < 1e-4){
        return;
    }
    // Look-at is along up: no yaw circle.
    if (fabs(offset.dot(up) / dist) > 0.98){
        return;
    }
    cMatrix3d rot;
    rot.setAxisAngleRotationDeg(up, rightDeg);
    m_mainCamera->setView(pivot + rot * offset, pivot, up);
    m_customViewActive = true;
}

void afVolmetricDrillingPlugin::nudgeCameraOrbitPitch(double upDeg)
{
    if (!m_mainCamera){
        return;
    }
    cVector3d pos = m_mainCamera->getLocalPos();
    cVector3d pivot = m_mainCamera->getTargetPosLocal();
    if ((pivot - pos).length() < 1e-3){
        pivot = m_volumeLookAt;
    }
    cVector3d right = m_mainCamera->getRightVector();
    cVector3d up = m_mainCamera->getUpVector();
    if (right.length() < 1e-6 || up.length() < 1e-6){
        return;
    }
    right.normalize();
    up.normalize();
    cVector3d offset = pos - pivot;
    double dist = offset.length();
    if (dist < 1e-4){
        return;
    }
    // Avoid flipping through the poles (offset parallel to up).
    if (fabs(offset.dot(up) / dist) > 0.98){
        return;
    }
    cMatrix3d rot;
    rot.setAxisAngleRotationDeg(right, upDeg);
    cVector3d newUp = rot * up;
    if (newUp.length() < 1e-6){
        return;
    }
    newUp.normalize();
    m_mainCamera->setView(pivot + rot * offset, pivot, newUp);
    m_customViewActive = true;
}

void afVolmetricDrillingPlugin::undoAmbfWheelDolly(GLFWwindow* a_window, double y_pos)
{
    if (!m_mainCamera){
        return;
    }
    // Match ambf_simulator mouseScrollCallback: local-X dolly, Shift slows 10x.
    double speed_scale = 1.0;
    if (a_window && glfwGetKey(a_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
        speed_scale = 0.1;
    }
    cVector3d camVelAlongLook(speed_scale * 0.1 * (-y_pos), 0, 0);
    cVector3d pos = m_mainCamera->getLocalPos();
    cVector3d delta = m_mainCamera->getLocalRot() * camVelAlongLook;
    m_mainCamera->setLocalPos(pos - delta);
}

namespace {
string trimCopy(const string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == string::npos){
        return string();
    }
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

string unquoteCopy(const string& s)
{
    string t = trimCopy(s);
    if (t.size() >= 2 && ((t.front() == '"' && t.back() == '"') || (t.front() == '\'' && t.back() == '\''))){
        return t.substr(1, t.size() - 2);
    }
    return t;
}

vector<double> parseYamlArray(const string& s)
{
    vector<double> out;
    string inner = s;
    size_t lb = inner.find('[');
    size_t rb = inner.rfind(']');
    if (lb != string::npos && rb != string::npos && rb > lb){
        inner = inner.substr(lb + 1, rb - lb - 1);
    }
    for (char& c : inner){
        if (c == ','){
            c = ' ';
        }
    }
    istringstream ss(inner);
    double x;
    while (ss >> x){
        out.push_back(x);
    }
    return out;
}

bool parseYamlBool(const string& s)
{
    string t = unquoteCopy(s);
    for (char& c : t){
        c = static_cast<char>(::tolower(c));
    }
    return t == "true" || t == "1" || t == "yes";
}

string yamlQuote(const string& s)
{
    bool need = s.empty();
    for (char c : s){
        if (c == ':' || c == '#' || c == '"' || c == '\'' || c == ' ' || c == '\t'){
            need = true;
            break;
        }
    }
    if (!need){
        return s;
    }
    string out = "\"";
    for (char c : s){
        if (c == '"' || c == '\\'){
            out.push_back('\\');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}
} // namespace

string afVolmetricDrillingPlugin::canonicalViewsPath()
{
    fs::path p = fs::path(__FILE__).parent_path() / ".." / ".." / "ADF" / "world" / "canonical_views.yaml";
    return fs::weakly_canonical(p).string();
}

void afVolmetricDrillingPlugin::loadCanonicalViews()
{
    m_canonicalViews.clear();
    ifstream in(canonicalViewsPath());
    if (!in.is_open()){
        cerr << "INFO! No canonical views at " << canonicalViewsPath() << endl;
        return;
    }

    CanonicalView cur;
    bool have = false;
    auto flush = [&](){
        if (have && !cur.id.empty()){
            m_canonicalViews.push_back(cur);
        }
        cur = CanonicalView();
        have = false;
    };

    string line;
    while (getline(in, line)){
        string raw = trimCopy(line);
        if (raw.empty() || raw[0] == '#'){
            continue;
        }
        if (raw == "views:"){
            continue;
        }
        bool newItem = false;
        if (raw.size() >= 2 && raw[0] == '-' && (raw[1] == ' ' || raw[1] == '\t')){
            flush();
            have = true;
            newItem = true;
            raw = trimCopy(raw.substr(1));
        }
        size_t colon = raw.find(':');
        if (colon == string::npos){
            continue;
        }
        if (!have && !newItem){
            continue;
        }
        have = true;
        string key = trimCopy(raw.substr(0, colon));
        string val = trimCopy(raw.substr(colon + 1));
        if (key == "id"){
            cur.id = unquoteCopy(val);
        }
        else if (key == "label"){
            cur.label = unquoteCopy(val);
        }
        else if (key == "laterality"){
            string lat = unquoteCopy(val);
            for (char& c : lat){
                c = static_cast<char>(::tolower(c));
            }
            cur.laterality = lat;
        }
        else if (key == "valid"){
            cur.valid = parseYamlBool(val);
        }
        else if (key == "t"){
            vector<double> arr = parseYamlArray(val);
            if (arr.size() >= 3){
                cur.T_rel.setLocalPos(cVector3d(arr[0], arr[1], arr[2]));
            }
        }
        else if (key == "q"){
            vector<double> arr = parseYamlArray(val);
            if (arr.size() >= 4){
                cQuaternion quat(arr[3], arr[0], arr[1], arr[2]); // qw, qx, qy, qz
                cMatrix3d rot;
                quat.toRotMat(rot);
                cur.T_rel.setLocalRot(rot);
            }
        }
    }
    flush();
    cerr << "INFO! Loaded " << m_canonicalViews.size() << " canonical views from "
         << canonicalViewsPath() << endl;
}

void afVolmetricDrillingPlugin::saveCanonicalViews()
{
    ofstream out(canonicalViewsPath());
    if (!out.is_open()){
        showNotice("ERROR: could not write " + canonicalViewsPath(), 4.0);
        return;
    }
    out << "# T_world_camera = T_world_home * T_home_view\n";
    out << "# Volume is never moved. Quaternion is qx qy qz qw.\n";
    out << "# Add new labeled slots here (valid: false) and overwrite them from the pause menu.\n";
    out << "views:\n";
    for (const CanonicalView& v : m_canonicalViews){
        cQuaternion q;
        q.fromRotMat(v.T_rel.getLocalRot());
        cVector3d p = v.T_rel.getLocalPos();
        out << "  - id: " << v.id << "\n";
        out << "    label: " << yamlQuote(v.label.empty() ? v.id : v.label) << "\n";
        out << "    laterality: " << v.laterality << "\n";
        out << "    valid: " << (v.valid ? "true" : "false") << "\n";
        out << "    t: [" << p.x() << ", " << p.y() << ", " << p.z() << "]\n";
        out << "    q: [" << q.x << ", " << q.y << ", " << q.z << ", " << q.w << "]\n";
    }
    out.flush();
    if (!out.good()){
        showNotice("ERROR: failed writing " + canonicalViewsPath(), 4.0);
        return;
    }
    cerr << "INFO! Wrote file " << canonicalViewsPath() << endl;
}

afVolmetricDrillingPlugin::CanonicalView* afVolmetricDrillingPlugin::findCanonicalView(const string& id)
{
    for (CanonicalView& v : m_canonicalViews){
        if (v.id == id){
            return &v;
        }
    }
    return nullptr;
}

void afVolmetricDrillingPlugin::saveCurrentIntoView(const string& id)
{
    if (!m_mainCamera){
        return;
    }
    if (id.empty() || id == "home" || id == "default"){
        showNotice("Cannot save into Home / Default");
        return;
    }
    CanonicalView* v = findCanonicalView(id);
    if (!v){
        showNotice("Unknown view id: " + id);
        return;
    }
    cTransform T_home_inv = m_originalCameraTransform;
    T_home_inv.invert();
    v->T_rel = T_home_inv * m_mainCamera->getLocalTransform();
    v->valid = true;
    m_customViewActive = true;
    m_activeCameraRollDeg = 0.0;
    saveCanonicalViews();
    string name = v->label.empty() ? v->id : v->label;
    showNotice("Saved view: " + name + "\n" + canonicalViewsPath());
}

bool afVolmetricDrillingPlugin::applyNamedView(const string& id)
{
    if (!m_mainCamera){
        return false;
    }
    if (id == "home"){
        applyHomeView();
        return true;
    }
    CanonicalView* v = findCanonicalView(id);
    if (!v){
        showNotice("Unknown view id: " + id);
        return false;
    }
    if (!v->valid){
        showNotice("View \"" + (v->label.empty() ? v->id : v->label) + "\" is empty.\n"
                   "Frame the camera, pause, highlight it, Shift+Enter to save.");
        return false;
    }
    m_mainCamera->setLocalTransform(m_originalCameraTransform * v->T_rel);
    m_customViewActive = true;
    m_activeCameraRollDeg = 0.0;
    showNotice("Applied view: " + (v->label.empty() ? v->id : v->label));
    return true;
}

void afVolmetricDrillingPlugin::applyHomeView()
{
    if (!m_mainCamera){
        return;
    }
    m_mainCamera->setLocalTransform(m_originalCameraTransform);
    m_customViewActive = false;
    m_activeCameraRollDeg = 0.0;
    showNotice("Applied Home view");
}

void afVolmetricDrillingPlugin::applyStartupView()
{
    string id = m_startupViewId;
    if (id.empty() || id == "default"){
        applyLateralityStartupView();
        return;
    }
    if (id == "home"){
        applyHomeView();
        return;
    }
    if (!applyNamedView(id)){
        applyLateralityStartupView();
    }
}

vector<afVolmetricDrillingPlugin::PauseViewEntry> afVolmetricDrillingPlugin::buildPauseViewList() const
{
    vector<PauseViewEntry> items;
    for (const CanonicalView& v : m_canonicalViews){
        if (!m_anatomyLaterality.empty()
            && (v.laterality == "left" || v.laterality == "right")
            && v.laterality != m_anatomyLaterality){
            continue;
        }
        PauseViewEntry e;
        e.id = v.id;
        e.label = v.label.empty() ? v.id : v.label;
        e.valid = v.valid;
        e.isHome = false;
        items.push_back(e);
    }
    PauseViewEntry home;
    home.id = "home";
    home.label = "Home";
    home.valid = true;
    home.isHome = true;
    items.push_back(home);
    return items;
}

void afVolmetricDrillingPlugin::cyclePauseViewHighlight(int delta)
{
    int n = static_cast<int>(buildPauseViewList().size());
    if (n <= 0){
        m_pauseViewIndex = 0;
        return;
    }
    int idx = m_pauseViewIndex % n;
    if (idx < 0){
        idx += n;
    }
    m_pauseViewIndex = (idx + delta % n + n) % n;
}

void afVolmetricDrillingPlugin::applyHighlightedPauseView()
{
    vector<PauseViewEntry> items = buildPauseViewList();
    if (items.empty()){
        return;
    }
    if (m_pauseViewIndex < 0){
        m_pauseViewIndex = 0;
    }
    if (m_pauseViewIndex >= static_cast<int>(items.size())){
        m_pauseViewIndex = static_cast<int>(items.size()) - 1;
    }
    applyNamedView(items[m_pauseViewIndex].id);
}

void afVolmetricDrillingPlugin::saveHighlightedPauseView()
{
    vector<PauseViewEntry> items = buildPauseViewList();
    if (items.empty()){
        return;
    }
    if (m_pauseViewIndex < 0){
        m_pauseViewIndex = 0;
    }
    if (m_pauseViewIndex >= static_cast<int>(items.size())){
        m_pauseViewIndex = static_cast<int>(items.size()) - 1;
    }
    const PauseViewEntry& e = items[m_pauseViewIndex];
    if (e.isHome){
        showNotice("Cannot save into Home");
        return;
    }
    saveCurrentIntoView(e.id);
}

void afVolmetricDrillingPlugin::showNotice(const string& text, double durationSec)
{
    cerr << text << endl;
    if (!m_noticeLabel){
        return;
    }
    m_panelManager.setText(m_noticeLabel, text);
    // Desktop: upper-right. VR: center (set in applyStereoCenteredHud; keep it there).
    m_panelManager.setPosForCamera(m_noticeLabel, m_mainCamera, 0.985, 0.97,
                                   PanelReferenceOrigin::UPPER_RIGHT, PanelReferenceType::NORMALIZED);
    if (m_stereoCamera){
        m_panelManager.setPosForCamera(m_noticeLabel, m_stereoCamera, 0.5, 0.72,
                                       PanelReferenceOrigin::CENTER, PanelReferenceType::NORMALIZED);
    }
    if (m_openvrCamera){
        m_panelManager.setPosForCamera(m_noticeLabel, m_openvrCamera, 0.5, 0.72,
                                       PanelReferenceOrigin::CENTER, PanelReferenceType::NORMALIZED);
        m_panelManager.setFontScaleForCamera(m_noticeLabel, m_openvrCamera, 1.1);
    }
    if (m_openxrEyepieceCamera){
        m_panelManager.setPosForCamera(m_noticeLabel, m_openxrEyepieceCamera, 0.5, 0.72,
                                       PanelReferenceOrigin::CENTER, PanelReferenceType::NORMALIZED);
        m_panelManager.setFontScaleForCamera(m_noticeLabel, m_openxrEyepieceCamera, 1.1);
    }
    m_panelManager.setVisible(m_noticeLabel, true);
    m_noticeTimer = durationSec;
}

void afVolmetricDrillingPlugin::applyStereoCenteredHud()
{
    ensurePauseMenuLabel();

    if (m_stereoCamera){
        // Desktop corner labels stay on main_camera only; hide their stereo copies.
        m_panelManager.setVisibleForCamera(m_drillManager.m_sizeLabel, m_stereoCamera, false);
        m_panelManager.setVisibleForCamera(m_drillManager.m_controlModeLabel, m_stereoCamera, false);
        m_panelManager.setVisibleForCamera(m_volumeSmoothingLabel, m_stereoCamera, false);
        m_panelManager.setVisibleForCamera(m_cameraPresetLabel, m_stereoCamera, false);

        // Notices: larger + centered on VR (still event-driven, not part of the toggle HUD).
        m_panelManager.setPosForCamera(m_noticeLabel, m_stereoCamera, 0.5, 0.72,
                                       PanelReferenceOrigin::CENTER, PanelReferenceType::NORMALIZED);
        m_panelManager.setFontScaleForCamera(m_noticeLabel, m_stereoCamera, 1.1);
    }

    setVrHudVisible(false);
}

void afVolmetricDrillingPlugin::ensurePauseMenuLabel()
{
    if (m_vrHudLabel){
        return;
    }
    cFontPtr font = NEW_CFONTCALIBRI40();
    m_vrHudLabel = new cLabel(font);
    m_vrHudLabel->m_fontColor.setBlack();
    m_vrHudLabel->setFontScale(0.75);
    m_vrHudLabel->setColor(cColorf(1.0, 1.0, 1.0));
    m_vrHudLabel->setTransparencyLevel(0.88);
    m_vrHudLabel->setCornerRadius(12, 12, 12, 12);
    m_vrHudLabel->setMargins(20, 20, 20, 20);
    m_vrHudLabel->setShowPanel(true);
    refreshVrHudText();
    m_panelManager.addPanel(m_vrHudLabel, 0.5, 0.48, PanelReferenceOrigin::CENTER, PanelReferenceType::NORMALIZED);
    // Start hidden on every camera; setVrHudVisible toggles visibility.
    if (m_mainCamera){
        m_panelManager.setVisibleForCamera(m_vrHudLabel, m_mainCamera, false);
        m_panelManager.setFontScaleForCamera(m_vrHudLabel, m_mainCamera, 0.65);
    }
    if (m_stereoCamera){
        m_panelManager.setVisibleForCamera(m_vrHudLabel, m_stereoCamera, false);
        m_panelManager.setFontScaleForCamera(m_vrHudLabel, m_stereoCamera, 0.85);
    }
    if (m_openvrCamera){
        m_panelManager.setVisibleForCamera(m_vrHudLabel, m_openvrCamera, false);
        m_panelManager.setFontScaleForCamera(m_vrHudLabel, m_openvrCamera, 1.0);
        m_panelManager.setPosForCamera(m_vrHudLabel, m_openvrCamera, 0.5, 0.48,
                                       PanelReferenceOrigin::CENTER, PanelReferenceType::NORMALIZED);
    }
    if (m_openxrEyepieceCamera){
        m_panelManager.setVisibleForCamera(m_vrHudLabel, m_openxrEyepieceCamera, false);
        m_panelManager.setFontScaleForCamera(m_vrHudLabel, m_openxrEyepieceCamera, 1.0);
        m_panelManager.setPosForCamera(m_vrHudLabel, m_openxrEyepieceCamera, 0.5, 0.48,
                                       PanelReferenceOrigin::CENTER, PanelReferenceType::NORMALIZED);
    }
}

void afVolmetricDrillingPlugin::refreshVrHudText()
{
    if (!m_vrHudLabel){
        return;
    }

    string user = m_sessionUser.empty() ? "(none)" : m_sessionUser;
    string anatomy = m_sessionAnatomy.empty() ? "(unknown)" : m_sessionAnatomy;
    string laterality = m_anatomyLaterality.empty() ? "unknown" : m_anatomyLaterality;
    string drill = "unknown";
    if (m_drillManager.m_activeDrill){
        drill = m_drillManager.m_activeDrill->m_name;
    }
    else if (m_drillManager.m_sizeLabel){
        drill = m_drillManager.m_sizeLabel->getText();
    }

    string status = m_simPaused ? "PAUSED\n\n" : "PAUSE MENU\n\n";

    string session =
        "SESSION\n"
        "  User          " + user + "\n"
        "  Anatomy       " + anatomy + "\n"
        "  Side          " + laterality + "\n"
        "  Drill         " + drill + "\n";

    string viewsHeader = "\nVIEWS";
    if (!m_anatomyLaterality.empty()){
        string lat = m_anatomyLaterality;
        for (char& c : lat){
            c = static_cast<char>(::toupper(c));
        }
        viewsHeader += "  (" + lat + ")";
    }
    viewsHeader += "\n";

    vector<PauseViewEntry> items = buildPauseViewList();
    if (items.empty()){
        m_pauseViewIndex = 0;
    }
    else if (m_pauseViewIndex < 0){
        m_pauseViewIndex = 0;
    }
    else if (m_pauseViewIndex >= static_cast<int>(items.size())){
        m_pauseViewIndex = static_cast<int>(items.size()) - 1;
    }
    string viewList;
    for (int i = 0; i < static_cast<int>(items.size()); i++){
        const PauseViewEntry& e = items[i];
        string marker = (i == m_pauseViewIndex) ? "  > " : "    ";
        string state = e.isHome ? "" : (e.valid ? "  saved" : "  empty");
        viewList += marker + e.label + state + "\n";
    }
    string text = status + session + viewsHeader + viewList + "\n" + controlHelpText(true);
    m_vrHudLabel->setText(text);
    m_panelManager.setText(m_vrHudLabel, text);
}

string afVolmetricDrillingPlugin::controlHelpText(bool paused) const
{
    // Action on the left, binding on the right — same pattern as a game
    // pause / controls screen. Pause menu lists the keyboard debug recenter;
    // the always-on overlay does not.
    string s;
    if (paused){
        s +=
            "MENU\n"
            "  Select                 Up / Down  or  Stylus 2\n"
            "  Confirm                Enter  or  Shift + Stylus 2\n"
            "  Save to slot           Shift + Enter\n"
            "  Resume                 Double-tap Stylus 1  or  Ctrl + P\n"
            "  Recenter drill         Double-tap Stylus 2  or  F9\n"
            "\n";
    }
    s += "CONTROLS\n";
    if (!paused){
        s += "  Hide / show this panel H\n";
        s += "  Pause / Resume         Double-tap Stylus 1  or  Ctrl + P\n";
    }
    s +=
            "  Camera clutch          Hold 1  or  Stylus 1  or  Alt (HMD)\n"
            "  Drill (hold-to-run)    Middle pedal (b)  or  Space\n"
            "  Burr style / size      Left pedal (a) / Right pedal (c)\n"
            "  Place WiVRn scene      Hold F10 (drag) / F11 lap / F12 look\n"
            "  Device clutch          Hold Stylus 2\n"
            "  Look                   Arrow Keys\n"
            "  Roll                   Ctrl + Mouse Wheel\n"
            "  Zoom in / out          Ctrl + Stylus 2 / 1\n"
            "  Orbit around burr      Shift + Hold Stylus 2\n"
            "  Recenter drill         Double-tap Stylus 2  or  F9\n"
            "  Left / Right view      F3 / F4\n"
            "  Quick roll L / R       F1 / F2\n"
            "  Specular legacy/new    F5\n"
            "  Shininess / strength   [ ]   /   - =\n"
            "  Attenuation none/mild  F6        , . linear\n"
            "  Beam narrower / wider  F7 / F8\n"
            "  Beam softer / tighter  ; / '\n"
            "  Room darker / lighter  Shift + [ ]\n";
    return s;
}

void afVolmetricDrillingPlugin::setVrHudVisible(bool visible)
{
    m_vrHudVisible = visible;
    m_simPaused = visible;

    // Freeze AMBF Sim Time via pausePhysics. Wall Time keeps ticking during the
    // pause for the HUD, but on resume we rewind wall by the pause duration so
    // computeStepSize() does not catch Sim Time up through the paused interval
    // (which would put pause time back into simulated task time).
    if (m_worldPtr){
        if (m_simPaused){
            m_wallTimeAtPause = m_worldPtr->getWallTime();
            m_worldPtr->pausePhysics(true);
        }
        else{
            m_worldPtr->pausePhysics(false);
            if (m_wallTimeAtPause >= 0.0){
                double now = m_worldPtr->getWallTime();
                double pauseDur = now - m_wallTimeAtPause;
                if (pauseDur > 0.0){
                    m_worldPtr->m_wallClock.stop();
                    m_worldPtr->m_wallClock.reset(now - pauseDur);
                    m_worldPtr->m_wallClock.start(false);
                }
                m_wallTimeAtPause = -1.0;
            }
        }
    }

    ensurePauseMenuLabel();
    if (!m_vrHudLabel){
        cerr << "INFO! Simulator " << (m_simPaused ? "PAUSED" : "RESUMED")
             << " (pause menu label unavailable)" << endl;
        return;
    }

    if (visible){
        vector<PauseViewEntry> items = buildPauseViewList();
        if (items.empty()){
            m_pauseViewIndex = 0;
        }
        else if (m_pauseViewIndex < 0 || m_pauseViewIndex >= static_cast<int>(items.size())){
            m_pauseViewIndex = 0;
        }
        refreshVrHudText();
    }

    // Show on desktop AMBF window and VR headset when available.
    if (m_mainCamera){
        m_panelManager.setVisibleForCamera(m_vrHudLabel, m_mainCamera, visible);
    }
    if (m_stereoCamera){
        m_panelManager.setVisibleForCamera(m_vrHudLabel, m_stereoCamera, visible);
        // Keep individual stereo corner copies hidden; the combined panel is the menu.
        m_panelManager.setVisibleForCamera(m_drillManager.m_sizeLabel, m_stereoCamera, false);
        m_panelManager.setVisibleForCamera(m_drillManager.m_controlModeLabel, m_stereoCamera, false);
        m_panelManager.setVisibleForCamera(m_volumeSmoothingLabel, m_stereoCamera, false);
        m_panelManager.setVisibleForCamera(m_cameraPresetLabel, m_stereoCamera, false);
    }
    if (m_openvrCamera){
        m_panelManager.setVisibleForCamera(m_vrHudLabel, m_openvrCamera, visible);
    }
    if (m_openxrEyepieceCamera){
        m_panelManager.setVisibleForCamera(m_vrHudLabel, m_openxrEyepieceCamera, visible);
    }

    cerr << "INFO! Simulator " << (m_simPaused ? "PAUSED" : "RESUMED")
         << " sim=" << (m_worldPtr ? m_worldPtr->getSimulationTime() : -1.0)
         << " | Pause menu " << (visible ? "ON" : "OFF")
         << " (Ctrl+P or dbl-click stylus button 1)" << endl;
}

string afVolmetricDrillingPlugin::findAnatomyAdfPathFromArgs(int argc, char** argv) const
{
    for (int i = 0; i < argc - 1; i++){
        string arg = argv[i] ? argv[i] : "";
        if (arg == "-a" || arg == "--adf" || arg == "--anatomy"){
            return argv[i + 1] ? string(argv[i + 1]) : string();
        }
        // Combined form: -a/path/to.yaml is uncommon; skip.
    }
    return string();
}

string afVolmetricDrillingPlugin::readLateralityFromAdf(const string& adfPath) const
{
    if (adfPath.empty()){
        return string();
    }
    ifstream in(adfPath);
    if (!in.is_open()){
        cerr << "WARNING! Could not open anatomy ADF for laterality: " << adfPath << endl;
        return string();
    }
    string line;
    while (getline(in, line)){
        size_t first = line.find_first_not_of(" \t");
        if (first == string::npos || line[first] == '#'){
            continue;
        }
        // Match: laterality: left|right
        const string key = "laterality:";
        size_t pos = line.find(key);
        if (pos == string::npos){
            continue;
        }
        string value = line.substr(pos + key.size());
        size_t v0 = value.find_first_not_of(" \t");
        if (v0 == string::npos){
            continue;
        }
        size_t v1 = value.find_first_of(" \t#\r", v0);
        string lat = value.substr(v0, v1 == string::npos ? string::npos : v1 - v0);
        for (char& c : lat){
            c = static_cast<char>(::tolower(c));
        }
        if (lat == "left" || lat == "right"){
            return lat;
        }
    }
    return string();
}

void afVolmetricDrillingPlugin::applyLateralityStartupView()
{
    if (m_anatomyLaterality != "left" && m_anatomyLaterality != "right"){
        return;
    }
    const string id = m_anatomyLaterality;
    CanonicalView* v = findCanonicalView(id);
    if (v && v->valid){
        applyNamedView(id);
        return;
    }
    // Rough transtemporal presets (same as Fn+F1 / Fn+F2).
    const bool left = (m_anatomyLaterality == "left");
    setCameraRoll(left ? 45.0 : 315.0);
    showNotice(string("Applied rough ") + (left ? "LEFT" : "RIGHT")
               + " view (no saved view; laterality=" + m_anatomyLaterality + ")");
}

void afVolmetricDrillingPlugin::mousePosUpdate(GLFWwindow *a_window, double x_pos, double y_pos)
{
    if (!m_mainCamera || !m_customViewActive){
        return;
    }

    // Middle-mouse: AMBF arcball forces world-Z up; restore the current view's up.
    if (m_arcballMouseDown){
        m_mainCamera->setView(m_mainCamera->getLocalPos(),
                              m_mainCamera->getTargetPosLocal(),
                              m_preservedUp);
        return;
    }

    // Right-mouse: undo FPS snap; do not remap to a new action.
    if (m_rightMouseDown){
        m_mainCamera->setLocalRot(m_rightDragRot);
    }
}

void afVolmetricDrillingPlugin::applyRoomGray()
{
    const float g = static_cast<float>(m_roomGrayStep) / 20.0f;
    const cColorf room(g, g, g);
    if (m_worldPtr && m_worldPtr->getChaiWorld()){
        m_worldPtr->getChaiWorld()->setBackgroundColor(g, g, g);
    }
    for (cBackground* background : m_roomBackdrops){
        if (background){
            background->setCornerColors(room, room, room, room);
        }
    }
}

void afVolmetricDrillingPlugin::nudgeRoomGray(int deltaSteps)
{
    m_roomGrayStep = cClamp(m_roomGrayStep + deltaSteps, 0, 20);
    applyRoomGray();
    ostringstream msg;
    msg << "Room " << (m_roomGrayStep * 5) << "% gray";
    showNotice(msg.str(), 1.5);
}

void afVolmetricDrillingPlugin::applyVolumeLightingUniforms()
{
    if (!m_volumeObject){
        return;
    }
    cShaderProgramPtr shader = m_volumeObject->getShaderProgram();
    if (shader == nullptr){
        return;
    }

    shader->setUniformi("uAttenuationMode", m_attenuationMode);
    shader->setUniformf("uAttConstant", (float)m_attConstant);
    shader->setUniformf("uAttLinear", (float)m_attLinear);
    shader->setUniformf("uAttQuadratic", (float)m_attQuadratic);
    shader->setUniformf("uAttClampMin", (float)m_attClampMin);
    shader->setUniformf("uAttClampMax", (float)m_attClampMax);
    shader->setUniformi("uUseRevisedSpecular", m_revisedSpecular ? 1 : 0);
    shader->setUniformf("uSpecularStrength", (float)m_specularStrength);
    shader->setUniformf("uShininess", (float)m_shininess);
    shader->setUniformf("uSpotFocus", (float)m_spotFocus);

    if (m_worldPtr){
        afLightPtr light = m_worldPtr->getLight("light1", true);
        if (light){
            light->setCutOffAngle(m_spotCutoffRad);
        }
    }
}

void afVolmetricDrillingPlugin::dollyCamera(double distanceTowardVolume)
{
    if (!m_mainCamera){
        return;
    }

    cVector3d look = m_mainCamera->getLookVector();
    if (look.length() < 1e-6){
        return;
    }
    look.normalize();

    cVector3d pos = m_mainCamera->getLocalPos();
    cVector3d newPos = pos + look * distanceTowardVolume;

    // Keep a usable range vs the volume origin; do not re-aim the camera.
    const double minDist = 0.5;
    const double maxDist = 25.0;
    double distToVolume = (m_volumeLookAt - newPos).length();
    if (distToVolume < minDist || distToVolume > maxDist){
        return;
    }

    m_mainCamera->setLocalPos(newPos);
    m_customViewActive = true;
}

void afVolmetricDrillingPlugin::panCamera(double rightAmount, double upAmount)
{
    if (!m_mainCamera){
        return;
    }
    cVector3d right = m_mainCamera->getRightVector();
    cVector3d up = m_mainCamera->getUpVector();
    if (right.length() < 1e-6 || up.length() < 1e-6){
        return;
    }
    right.normalize();
    up.normalize();
    m_mainCamera->setLocalPos(m_mainCamera->getLocalPos() + right * rightAmount + up * upAmount);
    m_customViewActive = true;
}

bool afVolmetricDrillingPlugin::takeRtxCameraFile(double& yawDeg, double& pitchDeg,
                                                    double& panRight, double& panUp, double& dolly)
{
    // The drilling ROS node is never spun, so a Twist subscriber on it does
    // not receive. The RTX window writes running totals here instead.
    std::ifstream in("/tmp/fivrs-rtx/camera_nudge.txt");
    unsigned long long seq = 0;
    double yaw = 0.0;
    double pitch = 0.0;
    double right = 0.0;
    double up = 0.0;
    double along = 0.0;
    if (!(in >> seq >> yaw >> pitch >> right >> up >> along)){
        return false;
    }
    if (!std::isfinite(yaw) || !std::isfinite(pitch) || !std::isfinite(right)
        || !std::isfinite(up) || !std::isfinite(along)){
        return false;
    }
    if (seq == m_cameraNudgeSeq){
        return false;
    }
    if (seq < m_cameraNudgeSeq){
        m_cameraNudgeSeq = seq;
        m_cameraNudgeYaw = yaw;
        m_cameraNudgePitch = pitch;
        m_cameraNudgePanRight = right;
        m_cameraNudgePanUp = up;
        m_cameraNudgeDolly = along;
        return false;
    }
    yawDeg = yaw - m_cameraNudgeYaw;
    pitchDeg = pitch - m_cameraNudgePitch;
    panRight = right - m_cameraNudgePanRight;
    panUp = up - m_cameraNudgePanUp;
    dolly = along - m_cameraNudgeDolly;
    m_cameraNudgeSeq = seq;
    m_cameraNudgeYaw = yaw;
    m_cameraNudgePitch = pitch;
    m_cameraNudgePanRight = right;
    m_cameraNudgePanUp = up;
    m_cameraNudgeDolly = along;
    return fabs(yawDeg) > 1e-8 || fabs(pitchDeg) > 1e-8
        || fabs(panRight) > 1e-8 || fabs(panUp) > 1e-8 || fabs(dolly) > 1e-8;
}

void afVolmetricDrillingPlugin::serviceRtxCamera()
{
    if (!m_mainCamera){
        return;
    }
    double yaw = 0.0;
    double pitch = 0.0;
    double panRight = 0.0;
    double panUp = 0.0;
    double dolly = 0.0;
    if (!takeRtxCameraFile(yaw, pitch, panRight, panUp, dolly)){
        return;
    }
    static bool announced = false;
    if (!announced){
        announced = true;
        cerr << "INFO! RTX window is moving main_camera" << endl;
    }
    if (fabs(panRight) > 1e-8 || fabs(panUp) > 1e-8){
        panCamera(panRight, panUp);
    }
    if (fabs(dolly) > 1e-8){
        dollyCamera(dolly);
    }
    if (fabs(yaw) > 1e-6){
        nudgeCameraOrbitYaw(yaw);
    }
    if (fabs(pitch) > 1e-6){
        nudgeCameraOrbitPitch(pitch);
    }
}

void afVolmetricDrillingPlugin::initializeLabels()
{
    // create a font
    cFontPtr font = NEW_CFONTCALIBRI40();

    m_warningLabel = new cLabel(font);
    m_warningLabel->m_fontColor.setWhite();
    m_warningLabel->setText("WARNING! Critical Region Detected");
    m_warningLabel->setCornerRadius(5, 5, 5, 5);
    m_warningLabel->setShowPanel(true);
    m_warningLabel->setColor(cColorf(0.6, 0., 0., 1.0));
    m_warningLabel->setTransparencyLevel(0.6);

    m_panelManager.addPanel(m_warningLabel, 0.5, 0.5, PanelReferenceOrigin::CENTER, PanelReferenceType::NORMALIZED);

    m_volumeSmoothingLabel = new cLabel(font);
    m_volumeSmoothingLabel->m_fontColor.setRed();
    m_volumeSmoothingLabel->setFontScale(.5);
    m_volumeSmoothingLabel->setColor(cColorf(0.6, 0., 0., 1.0));
    m_volumeSmoothingLabel->setText("[ALT+S] Volume Smoothing: DISABLED");

    m_panelManager.addPanel(m_volumeSmoothingLabel, 20, 10, PanelReferenceOrigin::LOWER_LEFT, PanelReferenceType::PIXEL);

    m_cameraPresetLabel = new cLabel(font);
    m_cameraPresetLabel->m_fontColor.setBlack();
    m_cameraPresetLabel->setFontScale(.45);
    m_cameraPresetLabel->setColor(cColorf(1, 1, 1));
    m_cameraPresetLabel->setTransparencyLevel(0.8);
    m_cameraPresetLabel->setCornerRadius(8, 8, 8, 8);
    m_cameraPresetLabel->setShowPanel(true);
    m_cameraPresetLabel->setText(controlHelpText(false));
    m_panelManager.addPanel(m_cameraPresetLabel, 0.985, 0.03, PanelReferenceOrigin::LOWER_RIGHT, PanelReferenceType::NORMALIZED);

    m_noticeLabel = new cLabel(font);
    m_noticeLabel->m_fontColor.setGreen();
    m_noticeLabel->setFontScale(0.55);
    m_noticeLabel->setColor(cColorf(0.85, 1.0, 0.7));
    m_noticeLabel->setTransparencyLevel(0.9);
    m_noticeLabel->setCornerRadius(8, 8, 8, 8);
    m_noticeLabel->setShowPanel(true);
    m_noticeLabel->setText("");
    m_panelManager.addPanel(m_noticeLabel, 0.985, 0.97, PanelReferenceOrigin::UPPER_RIGHT, PanelReferenceType::NORMALIZED);
    m_panelManager.setVisible(m_noticeLabel, false);
}


void afVolmetricDrillingPlugin::keyboardUpdate(GLFWwindow *a_window, int a_key, int a_scancode, int a_action, int a_mods) {
    const bool ctrl = (a_mods & GLFW_MOD_CONTROL) != 0;
    const bool shift = (a_mods & GLFW_MOD_SHIFT) != 0;
    const bool alt = (a_mods & GLFW_MOD_ALT) != 0;

    // Ikkegol pedals emit bare a/b/c — owned by updateButtons(), not these bindings.
    if (!ctrl && !alt && !shift &&
        (a_key == GLFW_KEY_A || a_key == GLFW_KEY_B || a_key == GLFW_KEY_C)){
        return;
    }

    if (a_action == GLFW_PRESS && a_key == GLFW_KEY_F9 && !ctrl && !alt){
        requestDrillReposition();
        return;
    }

    // WiVRn / OpenXR eyepiece scene placement (LOCAL panel): drag + snap.
    if (m_openxrEyepieceCamera && !ctrl && !alt){
        ensureOpenxrPlacementFns();
        if (a_key == GLFW_KEY_F10 && m_openxrSetPlaceDrag){
            m_openxrSetPlaceDrag((a_action == GLFW_RELEASE) ? 0 : 1);
            return;
        }
        if (a_action == GLFW_PRESS && a_key == GLFW_KEY_F11 && m_openxrRecenterLap){
            m_openxrRecenterLap();
            showNotice("WiVRn scene → lap (down + forward)");
            return;
        }
        if (a_action == GLFW_PRESS && a_key == GLFW_KEY_F12 && m_openxrRecenterToHead){
            m_openxrRecenterToHead();
            showNotice("WiVRn scene → look direction");
            return;
        }
    }

    if (a_action == GLFW_PRESS && a_key == GLFW_KEY_H && !ctrl && !alt && !shift){
        m_helpVisible = !m_helpVisible;
        if (m_cameraPresetLabel){
            m_panelManager.setVisible(m_cameraPresetLabel, m_helpVisible);
        }
        showNotice(m_helpVisible ? "Controls shown" : "Controls hidden  (H)");
        return;
    }

    if (m_vrHudVisible && !ctrl && !alt){
        if (a_action == GLFW_PRESS &&
            (a_key == GLFW_KEY_UP || a_key == GLFW_KEY_DOWN ||
             a_key == GLFW_KEY_LEFT || a_key == GLFW_KEY_RIGHT)){
            int delta = (a_key == GLFW_KEY_UP || a_key == GLFW_KEY_LEFT) ? -1 : 1;
            cyclePauseViewHighlight(delta);
            refreshVrHudText();
            return;
        }
        if (a_action == GLFW_PRESS && (a_key == GLFW_KEY_ENTER || a_key == GLFW_KEY_KP_ENTER)){
            if (shift){
                saveHighlightedPauseView();
            }
            else{
                applyHighlightedPauseView();
            }
            refreshVrHudText();
            return;
        }
    }

    if (ctrl && !alt){
        if (a_action == GLFW_PRESS && a_key == GLFW_KEY_P){
            // Ctrl+P: pause / session menu on desktop + VR.
            setVrHudVisible(!m_vrHudVisible);
            return;
        }

        double rate = m_drillManager.m_drillRate;
        // controls linear motion of tool
        if (a_key == GLFW_KEY_W) {

            cVector3d dir = m_mainCamera->getUpVector() * rate;
            m_drillManager.incrementDevicePos(dir);
        }

        else if (a_key == GLFW_KEY_D) {

            cVector3d dir = m_mainCamera->getRightVector() * rate;
            m_drillManager.incrementDevicePos(dir);

        }

        else if (a_key == GLFW_KEY_S) {

            cVector3d dir = m_mainCamera->getUpVector() * rate;
            m_drillManager.incrementDevicePos(-dir);

        }

        else if (a_key == GLFW_KEY_A) {

            cVector3d dir = m_mainCamera->getRightVector() * rate;
            m_drillManager.incrementDevicePos(-dir);

        }

        else if (a_key == GLFW_KEY_K) {

            cVector3d dir = m_mainCamera->getLookVector() * rate;
            m_drillManager.incrementDevicePos(-dir);

        }

        else if (a_key == GLFW_KEY_I) {

            cVector3d dir = m_mainCamera->getLookVector() * rate;
            m_drillManager.incrementDevicePos(dir);
        }

        else if (a_key == GLFW_KEY_O) {

            m_drillManager.setOverrideControl(!m_drillManager.getOverrideControl());
        }

        else if (a_key == GLFW_KEY_C) {
            m_drillManager.m_showGoalProxySpheres = !m_drillManager.m_showGoalProxySpheres;
            for (int i = 0 ; i < m_drillManager.m_toolCursorList.size() ; i++){
                m_drillManager.m_toolCursorList[i]->m_hapticPoint->setShow(m_drillManager.m_showGoalProxySpheres, m_drillManager.m_showGoalProxySpheres);
            }
        }

        // option - polygonize model and save to file
        else if (a_key == GLFW_KEY_P) {
            cMultiMesh *surface = new cMultiMesh;
            m_voxelObj->polygonize(surface, 0.01, 0.01, 0.01);
            double SCALE = 0.1;
            double METERS_TO_MILLIMETERS = 1000.0;
            surface->scale(SCALE * METERS_TO_MILLIMETERS);
            surface->setUseVertexColors(true);
            surface->saveToFile("volume.obj");
            cout << "> Volume has been polygonized and saved to disk                            \r";
            delete surface;
        }

        // toggles size of drill burr/tip tool cursor
        else if (a_key == GLFW_KEY_N){
            cerr << "INFO! RESETTING THE VOLUME" << endl;
            m_volumeObject->reset();
        }

        else if (a_key == GLFW_KEY_G){
            m_gazeMarkerController.restart();
            m_timeSinceLastCalibration = 0;
            // commands for video_recording_controller
            // Fetch the recording path from the environment variable or a pre-defined location
            m_isRecording = !m_isRecording;
            if (m_isRecording) {
                // Prefer study GUI sidecar file next to this source tree, then env, then ~/recordings.
                std::string tempPath;
                if (const char* envTmp = std::getenv("VOLDRILL_RECORDING_PATH_FILE")) {
                    tempPath = envTmp;
                } else {
                    // plugin/volumetric_drilling -> repo/scripts/study_gui/Simulator_Recordings/tmp/...
                    fs::path sidecar = fs::path(__FILE__).parent_path().parent_path().parent_path()
                        / "scripts" / "study_gui" / "Simulator_Recordings" / "tmp" / "recording_path.txt";
                    tempPath = sidecar.string();
                }
                std::string recordingPath;
                std::ifstream inputFile(tempPath);
                if (inputFile.is_open()) {
                    std::getline(inputFile, recordingPath);
                    inputFile.close();
                    cerr << "Read recording path from file: " << recordingPath << std::endl;
                } else {
                    cerr << "Failed to open temp file: " << tempPath << std::endl;
                    if (const char* envRec = std::getenv("VOLDRILL_RECORDING_DIR")) {
                        recordingPath = envRec;
                    } else if (const char* home = std::getenv("HOME")) {
                        recordingPath = std::string(home) + "/Volumetric_Drilling_Userstudy/";
                    } else {
                        recordingPath = "/tmp/Volumetric_Drilling_Userstudy/";
                    }
                }
                // Ensure the directory exists
                if (!fs::exists(recordingPath)) {
                    if (!fs::create_directories(recordingPath)) {
                        cerr << "Failed to create recording directory: " << recordingPath << endl;
                        return;
                    }
                }
                m_videoRecordingController.init(m_worldPtr, recordingPath);
                // Set the save directory and proceed with video recording logic
                m_videoRecordingController.start_recording(m_worldPtr->getCurrentTimeStamp());
                // Visual or log confirmation for the user
                cout << "Video recording started at: " << recordingPath << endl;
            }
        }


        else if (a_key == GLFW_KEY_M) { // Toggle between hiding and showing the Gaze Marker (only if recording 1) Data + 2) Pupils)
            if (m_isRecording) {
                m_gazeMarkerController.toggleMarker();
                
                if (m_gazeMarkerController.isMarkerVisible()) {
                    cerr << "INFO! Gaze marker shown at camera center" << endl;
                } else {
                    cerr << "INFO! Gaze marker hidden" << endl;
                }
            } else {
                cerr << "WARNING! Cannot toggle gaze marker - recording is not active" << endl;
            }
        }
        
        else if (a_key == GLFW_KEY_Y){ // Stop recording key
            m_videoRecordingController.close();    
            m_isRecording = false;
        }

        else if (a_key == GLFW_KEY_E){
            static int enableShadow = 0;
            enableShadow = ! enableShadow;
            cerr << "INFO! TOGGLING SHADOW MAP " << enableShadow << endl;
            m_volumeObject->getShaderProgram()->setUniformi("uEnableShadow", enableShadow);
        }

        else if (a_key == GLFW_KEY_PAGE_UP){
            if(m_stereoCamera){
                m_stereoCamera->getInternalCamera()->m_stereoOffsetW += 0.05;
                cerr << "INFO! SETTING STEREO OFFSET W " <<
                        m_stereoCamera->getInternalCamera()->m_stereoOffsetW << endl;
            }
        }

        else if (a_key == GLFW_KEY_PAGE_DOWN){
            if(m_stereoCamera){
                m_stereoCamera->getInternalCamera()->m_stereoOffsetW -= 0.05;
                cerr << "INFO! SETTING STEREO OFFSET W " <<
                        m_stereoCamera->getInternalCamera()->m_stereoOffsetW << endl;
            }
        }

        else if (a_key == GLFW_KEY_T){
            int val = m_drillManager.m_activeDrill->getVoxelRemovalThreshold() + 1;
            m_drillManager.m_activeDrill->setVoxelRemvalThreshold(val);
            cerr << "INFO! REMOVAL THRESHOLD " << val << endl;
        }

        else if (a_key == GLFW_KEY_V){
            m_activeCameraRollDeg = 0.0;
            m_customViewActive = false;
            cerr << "INFO! CAMERA ROLL CLEARED" << endl;
        }

        // Reset the drill pose
        if (a_key == GLFW_KEY_R){
            cerr << "INFO! RESETTING THE DRILL" << endl;
            m_drillManager.reset();
        }
    }
    else if (alt && !ctrl){
        // Toggle Volume Smoothing
        if (a_key == GLFW_KEY_S){
            m_enableVolumeSmoothing = !m_enableVolumeSmoothing;
            cerr << "INFO! ENABLE VOLUME SMOOTHING: " << m_enableVolumeSmoothing << endl;
            m_volumeObject->getShaderProgram()->setUniformi("uSmoothVolume", m_enableVolumeSmoothing);
            m_volumeObject->getShaderProgram()->setUniformi("uSmoothingLevel", m_volumeSmoothingLevel);

        }
        else if(a_key == GLFW_KEY_UP){
            m_volumeSmoothingLevel = cClamp(m_volumeSmoothingLevel+1, 1, 10);
            cerr << "INFO! SETTING SMOOTHING LEVEL " << m_volumeSmoothingLevel << endl;
            m_volumeObject->getShaderProgram()->setUniformi("uSmoothingLevel", m_volumeSmoothingLevel);
        }
        else if(a_key == GLFW_KEY_DOWN){
            m_volumeSmoothingLevel = cClamp(m_volumeSmoothingLevel-1, 1, 10);
            cerr << "INFO! SETTING SMOOTHING LEVEL " << m_volumeSmoothingLevel << endl;
            m_volumeObject->getShaderProgram()->setUniformi("uSmoothingLevel", m_volumeSmoothingLevel);
        }

        else if (a_key == GLFW_KEY_T){
            int val = cMax(1, m_drillManager.m_activeDrill->getVoxelRemovalThreshold() - 1);
            m_drillManager.m_activeDrill->setVoxelRemvalThreshold(val);
            cerr << "INFO! REMOVAL THRESHOLD " << val << endl;
        }

        std::string text = "[ALT+S] Volume Smoothing: " + std::string(m_enableVolumeSmoothing ? "ENABLED" : "DISABLED");
        cColorf color;
        color.setRed();
        if (m_enableVolumeSmoothing){
            text += " ( LEVEL: " + to_string(m_volumeSmoothingLevel) + ")";
            color.setGreen();
        }
        m_panelManager.setText(m_volumeSmoothingLabel, text);
        m_panelManager.setFontColor(m_volumeSmoothingLabel, color);
    }
    else{

        // Roll main_camera about the camera-to-volume axis (volume stays fixed)
        if (a_action == GLFW_PRESS && a_key == GLFW_KEY_F1) {
            setCameraRoll(45);
        }
        else if (a_action == GLFW_PRESS && a_key == GLFW_KEY_F2) {
            setCameraRoll(315);
        }
        else if (a_action == GLFW_PRESS && a_key == GLFW_KEY_F3) {
            applyNamedView("left");
        }
        else if (a_action == GLFW_PRESS && a_key == GLFW_KEY_F4) {
            applyNamedView("right");
        }
        else if (a_action == GLFW_PRESS && a_key == GLFW_KEY_F5) {
            m_revisedSpecular = !m_revisedSpecular;
            if (m_revisedSpecular){
                ostringstream msg;
                msg << "Specular revised  strength " << fixed << setprecision(2)
                    << m_specularStrength << "  shininess " << setprecision(0) << m_shininess;
                showNotice(msg.str());
            }
            else{
                showNotice("Specular legacy  strength 1  shininess 64");
            }
        }
        else if (a_action == GLFW_PRESS && a_key == GLFW_KEY_F6) {
            m_attenuationMode = (m_attenuationMode == 0) ? 1 : 0;
            if (m_attenuationMode == 0){
                showNotice("Attenuation none  (brightness does not change with distance)");
            }
            else{
                ostringstream msg;
                msg << "Attenuation mild  linear " << fixed << setprecision(3) << m_attLinear
                    << "  clamp " << setprecision(2) << m_attClampMin << " to " << m_attClampMax;
                showNotice(msg.str());
            }
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && shift && a_key == GLFW_KEY_LEFT_BRACKET) {
            nudgeRoomGray(-1);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && shift && a_key == GLFW_KEY_RIGHT_BRACKET) {
            nudgeRoomGray(1);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_LEFT_BRACKET) {
            m_shininess = cClamp(m_shininess - 4.0, 1.0, 128.0);
            m_revisedSpecular = true;
            ostringstream msg;
            msg << "Shininess " << fixed << setprecision(0) << m_shininess
                << "  (higher = tighter hotspot)";
            showNotice(msg.str(), 1.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_RIGHT_BRACKET) {
            m_shininess = cClamp(m_shininess + 4.0, 1.0, 128.0);
            m_revisedSpecular = true;
            ostringstream msg;
            msg << "Shininess " << fixed << setprecision(0) << m_shininess
                << "  (higher = tighter hotspot)";
            showNotice(msg.str(), 1.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_MINUS) {
            m_specularStrength = cClamp(m_specularStrength - 0.05, 0.0, 2.0);
            m_revisedSpecular = true;
            ostringstream msg;
            msg << "Specular strength " << fixed << setprecision(2) << m_specularStrength;
            showNotice(msg.str(), 1.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_EQUAL) {
            m_specularStrength = cClamp(m_specularStrength + 0.05, 0.0, 2.0);
            m_revisedSpecular = true;
            ostringstream msg;
            msg << "Specular strength " << fixed << setprecision(2) << m_specularStrength;
            showNotice(msg.str(), 1.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_COMMA) {
            m_attLinear = cClamp(m_attLinear - 0.01, 0.0, 0.5);
            m_attenuationMode = 1;
            ostringstream msg;
            msg << "Attenuation linear " << fixed << setprecision(3) << m_attLinear
                << "  (higher = dimmer when farther)";
            showNotice(msg.str(), 1.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_PERIOD) {
            m_attLinear = cClamp(m_attLinear + 0.01, 0.0, 0.5);
            m_attenuationMode = 1;
            ostringstream msg;
            msg << "Attenuation linear " << fixed << setprecision(3) << m_attLinear
                << "  (higher = dimmer when farther)";
            showNotice(msg.str(), 1.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_F7) {
            m_spotCutoffRad = cClamp(m_spotCutoffRad - 0.01, 0.04, 0.60);
            ostringstream msg;
            msg << "Beam half-angle " << fixed << setprecision(1)
                << (m_spotCutoffRad * 180.0 / 3.1416) << " deg";
            showNotice(msg.str(), 1.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_F8) {
            m_spotCutoffRad = cClamp(m_spotCutoffRad + 0.01, 0.04, 0.60);
            ostringstream msg;
            msg << "Beam half-angle " << fixed << setprecision(1)
                << (m_spotCutoffRad * 180.0 / 3.1416) << " deg";
            showNotice(msg.str(), 1.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_SEMICOLON) {
            m_spotFocus = cClamp(m_spotFocus - 0.25, 0.25, 8.0);
            ostringstream msg;
            msg << "Beam focus " << fixed << setprecision(2) << m_spotFocus
                << "  (lower = broader inside the cone)";
            showNotice(msg.str(), 1.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_APOSTROPHE) {
            m_spotFocus = cClamp(m_spotFocus + 0.25, 0.25, 8.0);
            ostringstream msg;
            msg << "Beam focus " << fixed << setprecision(2) << m_spotFocus
                << "  (higher = tighter core)";
            showNotice(msg.str(), 1.5);
        }

        // Orbit around the current look-at. Hold for GLFW_REPEAT.
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_UP) {
            nudgeCameraOrbitPitch(2.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_DOWN) {
            nudgeCameraOrbitPitch(-2.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_RIGHT) {
            nudgeCameraOrbitYaw(2.5);
        }
        else if ((a_action == GLFW_PRESS || a_action == GLFW_REPEAT) && a_key == GLFW_KEY_LEFT) {
            nudgeCameraOrbitYaw(-2.5);
        }

        // option - reduce size along X axis
        else if (a_key == GLFW_KEY_4) {
            sliceVolume(0, -0.005);
        }

        // option - increase size along X axis
        else if (a_key == GLFW_KEY_5) {
            sliceVolume(0, 0.005);
        }

        // option - reduce size along Y axis
        else if (a_key == GLFW_KEY_6) {
            sliceVolume(1, -0.005);
        }

        // option - increase size along Y axis
        else if (a_key == GLFW_KEY_7) {
            sliceVolume(1, 0.005);
        }

        // option - reduce size along Z axis
        else if (a_key == GLFW_KEY_8) {
            sliceVolume(2, -0.005);
        }

        // option - increase size along Z axis
        else if (a_key == GLFW_KEY_9) {
            sliceVolume(2, 0.005);
        }
        // option - decrease quality of graphic rendering
        else if (a_key == GLFW_KEY_L) {
            double value = m_voxelObj->getQuality();
            m_voxelObj->setQuality(value - 0.01);
            cout << "> Quality set to " << cStr(m_voxelObj->getQuality(), 1) << "                            \r";
        }

        // option - increase quality of graphic rendering
        else if (a_key == GLFW_KEY_U) {
            double value = m_voxelObj->getQuality();
            m_voxelObj->setQuality(value + 0.01);
            cout << "> Quality set to " << cStr(m_voxelObj->getQuality(), 1) << "                            \r";
        }

        // option - toggle vertical mirroring
        else if (a_key == GLFW_KEY_ENTER) {
            m_renderingMode++;
            if (m_renderingMode > 7) {
                m_renderingMode = 0;
            }
            switch (m_renderingMode) {
            case 0:
                m_voxelObj->setRenderingModeBasic();
                std::cerr << "setRenderingModeBasic" << std::endl;
                break;
            case 1:
                m_voxelObj->setRenderingModeVoxelColors();
                std::cerr << "setRenderingModeVoxelColors" << std::endl;
                break;
            case 2:
                m_voxelObj->setRenderingModeVoxelColorMap();
                std::cerr << "setRenderingModeVoxelColorMap" << std::endl;
                break;
            case 3:
                m_voxelObj->setRenderingModeIsosurfaceColors();
                std::cerr << "setRenderingModeIsosurfaceColors" << std::endl;
                break;
            case 4:
                m_voxelObj->setRenderingModeIsosurfaceMaterial();
                std::cerr << "setRenderingModeIsosurfaceMaterial" << std::endl;
                break;
            case 5:
                m_voxelObj->setRenderingModeIsosurfaceColorMap();
                std::cerr << "setRenderingModeIsosurfaceColorMap" << std::endl;
                break;
            case 6:
                m_voxelObj->setRenderingModeDVRColorMap();
                std::cerr << "setRenderingModeDVRColorMap" << std::endl;
                break;
            case 7:
                m_voxelObj->setRenderingModeCustom();
                std::cerr << "setRenderingModeCustom" << std::endl;
                break;
            default:
                break;
            }
        } else if (a_key == GLFW_KEY_PAGE_UP) {
            m_opticalDensity += 0.1;
            m_voxelObj->setOpticalDensity(m_opticalDensity);
            cout << "> Optical Density set to " << cStr(m_opticalDensity, 1) << "                            \n";
        } else if (a_key == GLFW_KEY_PAGE_DOWN) {
            m_opticalDensity -= 0.1;
            m_voxelObj->setOpticalDensity(m_opticalDensity);
            cout << "> Optical Density set to " << cStr(m_opticalDensity, 1) << "                            \n";
        } else if (a_key == GLFW_KEY_HOME) {
            float val = m_voxelObj->getOpacityThreshold();
            m_voxelObj->setOpacityThreshold(val + 0.1);
            cout << "> Optical Threshold set to " << cStr(m_voxelObj->getOpacityThreshold(), 1)
                 << "                            \n";
        } else if (a_key == GLFW_KEY_END) {
            float val = m_voxelObj->getOpacityThreshold();
            m_voxelObj->setOpacityThreshold(val - 0.1);
            cout << "> Optical Threshold set to " << cStr(m_voxelObj->getOpacityThreshold(), 1)
                 << "                            \n";
        }

        // controls rotational motion of tool
        else if(a_key == GLFW_KEY_KP_5) {

            cVector3d rotDir(0, 1, 0) ;
            m_drillManager.incrementDeviceRot(rotDir);
        }

        else if(a_key == GLFW_KEY_KP_8) {

            cVector3d rotDir(0, -1, 0);
            m_drillManager.incrementDeviceRot(rotDir);
        }

        else if(a_key == GLFW_KEY_KP_4) {

            cVector3d rotDir(0, 0, -1);
            m_drillManager.incrementDeviceRot(rotDir);
        }

        else if(a_key == GLFW_KEY_KP_6) {

            cVector3d rotDir(0, 0, 1);
            m_drillManager.incrementDeviceRot(rotDir);
        }

        // toggles the functionality of sudden jumping of drill mesh towards the followSphere
        else if(a_key == GLFW_KEY_X){

            m_drillManager.m_suddenJump = !m_drillManager.m_suddenJump;
        }

        // toggles the visibility of drill mesh (Shift+B; bare B is Ikkegol drill pedal)
        else if (shift && a_key == GLFW_KEY_B){
            m_drillManager.m_show = !m_drillManager.m_show;
            m_drillManager.showOnlyActive();
        }

        else if (a_key == GLFW_KEY_KP_ADD){
            if (m_cameraL){
                m_cameraL->setLocalPos(m_cameraL->getLocalPos() - cVector3d(0., 0.001, 0.));
            }
            if (m_cameraR){
                m_cameraR->setLocalPos(m_cameraR->getLocalPos() + cVector3d(0., 0.001, 0.));
            }

            if (m_stereoCamera){
                double stereo_sep = m_stereoCamera->getInternalCamera()->getStereoEyeSeparation();
                m_stereoCamera->getInternalCamera()->setStereoEyeSeparation(stereo_sep + 0.002);
            }
        }
        else if (a_key == GLFW_KEY_KP_SUBTRACT){
            if (m_cameraL){
                m_cameraL->setLocalPos(m_cameraL->getLocalPos() + cVector3d(0., 0.001, 0.));
            }
            if (m_cameraR){
                m_cameraR->setLocalPos(m_cameraR->getLocalPos() - cVector3d(0., 0.001, 0.));
            }

            if (m_stereoCamera){
                double stereo_sep = m_stereoCamera->getInternalCamera()->getStereoEyeSeparation();
                m_stereoCamera->getInternalCamera()->setStereoEyeSeparation(stereo_sep - 0.002);
            }
        }
    }

}


void afVolmetricDrillingPlugin::mouseBtnsUpdate(GLFWwindow *a_window, int a_button, int a_action, int a_modes){
    (void)a_window;
    (void)a_modes;
    if (a_button == GLFW_MOUSE_BUTTON_MIDDLE){
        m_arcballMouseDown = (a_action == GLFW_PRESS);
        if (m_arcballMouseDown && m_mainCamera){
            m_preservedUp = m_mainCamera->getUpVector();
        }
    }
    else if (a_button == GLFW_MOUSE_BUTTON_RIGHT){
        m_rightMouseDown = (a_action == GLFW_PRESS);
        if (m_rightMouseDown && m_mainCamera){
            m_rightDragRot = m_mainCamera->getLocalRot();
        }
    }
}

void afVolmetricDrillingPlugin::mouseScrollUpdate(GLFWwindow *a_window, double x_pos, double y_pos)
{
    (void)x_pos;
    if (!isCtrlHeld() || m_vrHudVisible){
        return;
    }
    if (fabs(y_pos) < 1e-6){
        return;
    }
    // AMBF already dollied along look; undo that so roll keeps position.
    undoAmbfWheelDolly(a_window, y_pos);
    nudgeCameraRoll(-5.0 * y_pos);
}

void afVolmetricDrillingPlugin::reset(){
    cerr << "INFO! PLUGIN RESET CALLED" << endl;
    m_activeCameraRollDeg = 0.0;
    m_customViewActive = false;
    m_arcballMouseDown = false;
    m_rightMouseDown = false;
    m_drillManager.reset();
}

bool afVolmetricDrillingPlugin::close()
{
    m_drillManager.cleanup();
    return true;
}
