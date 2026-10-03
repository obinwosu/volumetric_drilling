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

#include "drill_manager.h"
#include <boost/program_options.hpp>
#include <cmath>

DrillManager::DrillManager(){
    m_units_mmToSim = 0.01007;
}

void DrillManager::cleanup()
{
    for(auto tool : m_toolCursorList)
        {
            tool->stop();
        }

        if (m_audioSource){
            delete m_audioSource;
        }
        if (m_audioBuffer){
            delete m_audioBuffer;
        }
        if (m_audioDevice){
            m_mainCamera->getInternalCamera()->detachAudioDevice();
            delete m_audioDevice;
        }
        delete m_deviceHandler;
}

int DrillManager::init(afWorldPtr a_worldPtr, CameraPanelManager* a_panelManager, p_opt::variables_map& var_map){
    // importing drill model

    int nt = var_map["nt"].as<int>();
    bool mute = var_map["mute"].as<bool>();

    m_mainCamera = a_worldPtr->getCamera("main_camera");
    m_panelManager = a_panelManager;

    if (nt > 0 && nt <= 8){
        m_toolCursorList.resize(nt);
    }
    else{
        cerr << "ERROR! VALID NUMBER OF TOOL CURSORS ARE BETWEEN 1 - 8. Specified value = " << nt << endl;
        return -1;
    }

    m_drillReferenceBody = a_worldPtr->getRigidBody("mastoidectomy_drill");

    if (!m_drillReferenceBody){
        cerr << "ERROR! FAILED TO FIND REFERENCE DRILL RIGID BODY " << "mastoidectomy_drill" << endl;
        return -1;
    }

    vector<int> drillTypes = {1, 2, 4, 6};
    vector<int> voxelRemovalThresholds = {1, 3, 6, 10};

    for (int i = 0 ; i < drillTypes.size() ; i++){
        string drillName = to_string(drillTypes[i]) + "mm";
        afRigidBodyPtr drillRB = a_worldPtr->getRigidBody(drillName);
        if (drillRB){
            Drill* drill = new Drill();
            drill->m_name = drillName;
            drill->m_rigidBody = drillRB;
            drill->m_size = drillTypes[i] * m_units_mmToSim;
            drill->setVoxelRemvalThreshold(voxelRemovalThresholds[i]);
            m_drills.push_back(drill);
        }
    }

    if (m_drills.size() == 0){
        cerr << "ERROR! FAILED TO FIND DRILLS WITH PREFIX " << "mastoidectomy_drill" << endl;
        return -1;
    }

    m_activeDrillIdx = m_drills.size() - 1;
    m_activeDrill = m_drills[m_activeDrillIdx];
    if (var_map.count("sb")){
        int sb = var_map["sb"].as<int>();
        if (!setDrillSizeMm(sb)){
            cerr << "WARNING! Invalid --sb " << sb
                 << " mm; keeping default " << m_activeDrill->m_name << endl;
        }
    }
    showOnlyActive();

    m_dX = var_map["ds"].as<float>();

    string drill_matcap = var_map["dm"].as<string>();

    string file_path = __FILE__;
    string current_filepath = file_path.substr(0, file_path.rfind("/"));

    m_burrMesh = new cMultiMesh();
    string burrMeshPath = current_filepath + "/../../ADF/meshes/high_res/fluted_burr.obj";
    if (m_burrMesh->loadFromFile(burrMeshPath)){
        // Keep the authored hard edge normals. Smoothing them rounds the cutting lips
        // and makes the head look smaller than the old sphere.
        m_burrVisualRadius = 1.0;
        applyBurrVisualScale();
        cerr << "INFO! LOADED FLUTED BURR " << burrMeshPath << endl;
    }
    else{
        cerr << "FAILED TO LOAD FLUTED BURR FROM " << burrMeshPath << endl;
    }
    m_burrMesh->m_material->setWhite();
    m_burrMesh->m_material->m_specular.set(0.75, 0.75, 0.75);
    m_burrMesh->m_material->setShininess(80);
    m_burrMesh->setShowEnabled(true);

    a_worldPtr->addSceneObjectToWorld(m_burrMesh);

    initializeLabels();

    // Get drills initial pose
    m_T_d_init = m_drillReferenceBody->getLocalTransform();

    m_T_d = m_T_d_init;

    // Set up voxels_removed publisher
    m_drillingPub = new DrillingPublisher(a_worldPtr->getNamespace(), "/plugin/volumetric_drilling");

    cTexture2dPtr drillMatCap = cTexture2d::create();
    string drillMatcapFilepath = current_filepath + "/../../resources/matcap/" + drill_matcap;
    if(drillMatCap->loadFromFile(drillMatcapFilepath)){
        cerr << "SUCCESFULLY LOADED DRILL'S MATCAP TEXTURE" << endl;
        for (int di = 0 ; di < m_drills.size() ; di++){
            if (m_drills[di]->m_rigidBody->getShaderProgram()){
                for (int mi = 0 ; mi < m_drills[di]->m_rigidBody->getVisualObject()->getNumMeshes(); mi++){
                    m_drills[di]->m_rigidBody->getVisualObject()->getMesh(mi)->m_metallicTexture = drillMatCap;
                    m_drills[di]->m_rigidBody->getVisualObject()->getMesh(mi)->m_metallicTexture->setTextureUnit(GL_TEXTURE3);
                }
                m_drills[di]->m_rigidBody->getShaderProgram()->setUniformi("uMatcapMap", C_TU_METALLIC);
            }
        }
        if (m_burrMesh){
            cShaderProgramPtr burrShader = nullptr;
            for (int di = 0; di < (int)m_drills.size(); di++){
                burrShader = m_drills[di]->m_rigidBody->getShaderProgram();
                if (burrShader){
                    break;
                }
            }
            if (burrShader){
                m_burrMesh->setShaderProgram(burrShader, true);
                burrShader->setUniformi("uMatcapMap", C_TU_METALLIC);
                for (int mi = 0; mi < m_burrMesh->getNumMeshes(); mi++){
                    cMesh* mesh = m_burrMesh->getMesh(mi);
                    if (!mesh){
                        continue;
                    }
                    mesh->setShaderProgram(burrShader);
                    mesh->m_metallicTexture = drillMatCap;
                    mesh->m_metallicTexture->setTextureUnit(GL_TEXTURE3);
                }
            }
        }
    }
    else{
        cerr << "FAILED TO LOAD DRILL'S MATCAP TEXTURE" << endl;
    }

    if (!mute){
        m_audioDevice = new cAudioDevice();
        m_mainCamera->getInternalCamera()->attachAudioDevice(m_audioDevice);

        m_audioBuffer = new cAudioBuffer();
        string drillAudioFilepath = current_filepath + "/../../resources/sounds/drill.wav";
        if (m_audioBuffer->loadFromFile(drillAudioFilepath)){
            m_audioSource = new cAudioSource();
            //        m_drillAudioBuffer->convertToMono();
            m_audioSource->setAudioBuffer(m_audioBuffer);
            m_audioSource->setLoop(true);
            m_audioSource->setGain(5.0);
            m_audioState = AudioState::STOPPED;
        }
        else{
            delete m_audioSource;
            delete m_audioBuffer;
            m_audioSource = nullptr;
            m_audioBuffer = nullptr;
            cerr << "FAILED TO LOAD DRILL AUDIO FROM " << drillAudioFilepath << endl;
        }
    }

    // create a haptic device handler
    m_deviceHandler = new cHapticDeviceHandler();

    // get access to the first available haptic device found
    m_deviceHandler->getDevice(m_hapticDevice, 0);

    // Initializing tool cursors
    toolCursorInit(a_worldPtr);

    return 1;
}

void DrillManager::update(double dt)
{

    cTransform T_c_w = m_mainCamera->getLocalTransform();

    // If a valid haptic device is found, then it should be available
    if (getOverrideControl()){
        m_T_d = m_drillReferenceBody->getLocalTransform();
    }
    else if(m_hapticDevice->isDeviceAvailable()){
        m_hapticDevice->getTransform(m_T_i);
        m_hapticDevice->getLinearVelocity(m_V_i);
        // cMatrix3d extraRotation;
        // extraRotation.setExtrinsicEulerRotationDeg(0, 90, 0, C_EULER_ORDER_XYZ);
        // m_V_i = T_c_w.getLocalRot() * extraRotation * (m_V_i / m_toolCursorList[0]->getWorkspaceScaleFactor());
        m_V_i = T_c_w.getLocalRot() * (m_V_i / m_toolCursorList[0]->getWorkspaceScaleFactor());
        m_T_d.setLocalPos(m_T_d.getLocalPos() + (m_V_i * 0.8 * !m_deviceClutch * !m_camClutch * !m_camOrbitBurr));
        m_T_d.setLocalRot(T_c_w.getLocalRot() *  m_T_i.getLocalRot());


        // set zero forces when manipulating objects
        if (m_deviceClutch || m_camClutch || m_camOrbitBurr){
            if (m_camOrbitBurr){
                cVector3d pivot = m_T_d.getLocalPos();
                if (m_burrMesh){
                    pivot = m_burrMesh->getLocalPos();
                }
                else if (m_activeDrill && m_activeDrill->m_rigidBody){
                    pivot = m_activeDrill->m_rigidBody->getLocalPos();
                }

                cVector3d camPos = T_c_w.getLocalPos();
                cVector3d offset = camPos - pivot;
                double dist = offset.length();
                if (dist > 1e-4 &&
                    std::isfinite(m_V_i.x()) && std::isfinite(m_V_i.y()) && std::isfinite(m_V_i.z())){
                    cVector3d right = m_mainCamera->getRightVector();
                    cVector3d up = m_mainCamera->getUpVector();
                    if (right.length() > 1e-6 && up.length() > 1e-6){
                        right.normalize();
                        up.normalize();
                        double yaw = -m_V_i.dot(right) / dist;
                        double pitch = m_V_i.dot(up) / dist;
                        // Avoid flipping through the poles (offset parallel to up).
                        cVector3d offsetDir = offset / dist;
                        if (fabs(offsetDir.dot(up)) > 0.98){
                            pitch = 0.0;
                        }
                        cMatrix3d R;
                        R.identity();
                        if (fabs(yaw) > 1e-12){
                            cMatrix3d Ry;
                            Ry.setAxisAngleRotationRad(up, yaw);
                            R = Ry * R;
                        }
                        if (fabs(pitch) > 1e-12){
                            cMatrix3d Rp;
                            Rp.setAxisAngleRotationRad(right, pitch);
                            R = Rp * R;
                        }
                        // Rotate the whole view around the burr. Forcing the look
                        // target onto the burr recenters it and reads as a jump.
                        cVector3d lookAt = m_mainCamera->getTargetPosLocal();
                        cVector3d newPos = pivot + R * offset;
                        cVector3d newLook = pivot + R * (lookAt - pivot);
                        cVector3d newUp = R * up;
                        if (newUp.length() > 1e-6 && (newPos - newLook).lengthsq() > 1e-8){
                            newUp.normalize();
                            m_mainCamera->setView(newPos, newLook, newUp);
                        }
                    }
                }
            }
            else if (m_camClutch){
                m_mainCamera->setView(T_c_w.getLocalPos() + m_V_i * !m_deviceClutch, m_mainCamera->getTargetPosLocal(), m_mainCamera->getUpVector());
            }
            m_toolCursorList[0]->setDeviceLocalForce(0.0, 0.0, 0.0);
        }
        m_drillReferenceBody->setLocalTransform(m_T_d);
    }

    toolCursorsPosUpdate(m_T_d);

    // check for shaft collision
    checkShaftCollision();

//    if (getOverrideControl() == false){
        // updates position of drill mesh
        updatePoseFromCursors();
//    }

    if (m_burrMesh && m_activeDrill && m_activeDrill->m_rigidBody){
        if (m_isOn){
            // Four visible turns per second. Real burr RPM would alias at this frame rate.
            const double spinRadPerSec = 4.0 * 6.283185307179586;
            m_burrSpinAngle += dt * spinRadPerSec;
            if (m_burrSpinAngle > 6.283185307179586){
                m_burrSpinAngle = std::fmod(m_burrSpinAngle, 6.283185307179586);
            }
        }
        cTransform burrPose = m_activeDrill->m_rigidBody->getLocalTransform();
        cMatrix3d spin;
        spin.setAxisAngleRotationRad(cVector3d(1.0, 0.0, 0.0), m_burrSpinAngle);
        burrPose.setLocalRot(burrPose.getLocalRot() * spin);
        m_burrMesh->setLocalTransform(burrPose);
    }

    if (m_isOn && m_audioSource){
        if (m_audioState == AudioState::STOPPED){
            m_audioSource->play();
            m_audioState = AudioState::PLAYING;
        }
    }
    else{
        if (m_audioState == AudioState::PLAYING){
            m_audioSource->stop();
            m_audioState = AudioState::STOPPED;
        }
    }
}

void DrillManager::initializeLabels(){
    // create a font
    cFontPtr font = NEW_CFONTCALIBRI32();

    m_sizeLabel = new cLabel(font);
    m_sizeLabel->setLocalPos(10, 10); // Relative to Panel
    m_sizeLabel->m_fontColor.setBlack();
    m_sizeLabel->setText("Drill Type: " + m_activeDrill->m_name);
    m_sizeLabel->setColor(cColorf(1, 1, 1));
    m_sizeLabel->setTransparencyLevel(0.8);
    m_sizeLabel->setCornerRadius(10, 10, 10, 10);
    m_sizeLabel->setShowPanel(true);

    m_panelManager->addPanel(m_sizeLabel, 15, 60, PanelReferenceOrigin::LOWER_LEFT, PanelReferenceType::PIXEL);
//    m_panelManager->setVisible(m_sizePanel, true);

    m_controlModeLabel = new cLabel(font);
    m_controlModeLabel->m_fontColor.setGreen();
    m_controlModeLabel->setFontScale(.5);
    m_controlModeLabel->setText("[CTRL+O] Drill Control Mode = Haptic Device / Keyboard");
    m_panelManager->addPanel(m_controlModeLabel, 20, 35, PanelReferenceOrigin::LOWER_LEFT, PanelReferenceType::PIXEL);
}

void DrillManager::setOverrideControl(bool val){
    m_overrideControl = val;
    cColorf color;
    if (getOverrideControl()){
        color.setRed();
        m_panelManager->setText(m_controlModeLabel, "[CTRL+O] Drill Control Mode = External afComm");
        m_panelManager->setFontColor(m_controlModeLabel, color);
    }
    else{
        color.setGreen();
        m_panelManager->setText(m_controlModeLabel, "[CTRL+O] Drill Control Mode = Haptic Device / Keyboard");
        m_panelManager->setFontColor(m_controlModeLabel, color);
    }
}

void DrillManager::setAudioPitch(double pitch)
{
    if (m_audioSource){
        m_audioSource->setPitch(pitch);
    }
}

///
/// \brief This method initializes the tool cursors.
/// \param a_afWorld    A world that contains all objects of the virtual environment
/// \return
///
void DrillManager::toolCursorInit(const afWorldPtr a_afWorld){

    for(int i=0; i < m_toolCursorList.size(); i++)
    {
        m_toolCursorList[i] = new cToolCursor(a_afWorld->getChaiWorld());

        a_afWorld->addSceneObjectToWorld(m_toolCursorList[i]);

        if(i == 0)
        {
            m_toolCursorList[i]->setHapticDevice(m_hapticDevice);

            // map the physical workspace of the haptic device to a larger virtual workspace.

            m_toolCursorList[i]->setWorkspaceRadius(5.0);
            m_toolCursorList[i]->setWaitForSmallForce(true);
            m_toolCursorList[i]->start();
            m_toolCursorList[i]->m_hapticPoint->m_sphereProxy->setShowFrame(false);

            m_toolCursorList[i]->m_name = "mastoidectomy_drill";
            m_toolCursorList[i]->m_hapticPoint->setShow(m_showGoalProxySpheres, m_showGoalProxySpheres);
            m_toolCursorList[i]->m_hapticPoint->m_sphereProxy->m_material->setRedCrimson();
            m_toolCursorList[i]->m_hapticPoint->m_sphereGoal->m_material->setBlueAquamarine();

            // if the haptic device has a gripper, enable it as a user switch
            m_hapticDevice->setEnableGripperUserSwitch(true);
            m_toolCursorList[i]->setRadius(m_activeDrill->m_size * 0.5); // Set the correct radius for the tip which is not from the list of cursor radii
        }
        else
        {
            m_toolCursorList[i]->setShowContactPoints(m_showGoalProxySpheres, m_showGoalProxySpheres);
            m_toolCursorList[i]->m_hapticPoint->m_sphereProxy->m_material->setGreenChartreuse();
            m_toolCursorList[i]->m_hapticPoint->m_sphereGoal->m_material->setOrangeCoral();
            m_toolCursorList[i]->setRadius(m_toolCursorRadius[i]);
        }
     }

    // Initialize the start pose of the tool cursors
    toolCursorsPosUpdate(m_T_d);
    toolCursorsInitialize();
}


///
/// \brief incrementDevicePos
/// \param a_vel
///
void DrillManager::incrementDevicePos(cVector3d a_vel){
    m_T_d.setLocalPos(m_T_d.getLocalPos() + a_vel);
}


///
/// \brief incrementDeviceRot
/// \param a_rot
///
void DrillManager::incrementDeviceRot(cVector3d a_rot){
    cMatrix3d R_cmd;
    R_cmd.setExtrinsicEulerRotationDeg(a_rot(0), a_rot(1), a_rot(2), C_EULER_ORDER_XYZ);
    R_cmd = m_T_d.getLocalRot() * R_cmd;
    m_T_d.setLocalRot(R_cmd);
}

///
/// \brief afVolmetricDrillingPlugin::toolCursorsInitialize
///
void DrillManager::toolCursorsInitialize(){
    for (int i = 0 ;  i < m_toolCursorList.size() ; i++){
        m_toolCursorList[i]->initialize();
    }
}

///
/// \brief This method updates the position of the shaft tool cursors
/// which eventually updates the position of the whole tool.
///
void DrillManager::toolCursorsPosUpdate(cTransform a_targetPose){
    cVector3d n_x = a_targetPose.getLocalRot().getCol0() * m_dX;
    for (int i = 0 ; i < m_toolCursorList.size() ; i++){
        cVector3d P = a_targetPose.getLocalPos() + n_x * i;
        m_toolCursorList[i]->setDeviceLocalPos(P);
        m_toolCursorList[i]->setDeviceLocalRot(a_targetPose.getLocalRot());
    }
}

void DrillManager::reset(){
    m_hapticDevice->setForce(cVector3d(0., 0., 0.));
    m_T_d = m_T_d_init;
    toolCursorsPosUpdate(m_T_d);
    toolCursorsInitialize();
    updatePoseFromCursors();
}

static cVector3d worldToVolumeLocal(const cVector3d& pWorld, const cTransform& volumePose)
{
    return cTranspose(volumePose.getLocalRot()) * (pWorld - volumePose.getLocalPos());
}

static bool inExpandedAabb(const cVector3d& pLocal, const cVector3d& half)
{
    return (fabs(pLocal.x()) <= half.x())
        && (fabs(pLocal.y()) <= half.y())
        && (fabs(pLocal.z()) <= half.z());
}

static double vecComp(const cVector3d& v, int i)
{
    return (i == 0) ? v.x() : (i == 1) ? v.y() : v.z();
}

// Slab test. tEnter is the near hit along the ray (may be negative if origin is inside).
static bool rayAabbIntersect(const cVector3d& origin,
                             const cVector3d& dir,
                             const cVector3d& half,
                             double& tEnter,
                             double& tExit)
{
    tEnter = -1.0e30;
    tExit = 1.0e30;
    for (int i = 0; i < 3; i++){
        double oi = vecComp(origin, i);
        double di = vecComp(dir, i);
        double h = vecComp(half, i);
        if (fabs(di) < 1e-12){
            if (oi < -h || oi > h){
                return false;
            }
            continue;
        }
        double t0 = (-h - oi) / di;
        double t1 = ( h - oi) / di;
        if (t0 > t1){
            double tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tEnter = cMax(tEnter, t0);
        tExit = cMin(tExit, t1);
        if (tEnter > tExit){
            return false;
        }
    }
    return tExit >= 0.0;
}

static cVector3d volumeLocalToWorld(const cVector3d& pLocal, const cTransform& volumePose)
{
    return volumePose.getLocalPos() + volumePose.getLocalRot() * pLocal;
}

bool DrillManager::snapToCameraView(const cVector3d& volumeOrigin,
                                    const cVector3d& volumeDims,
                                    const cTransform& volumePose)
{
    if (!m_mainCamera){
        return false;
    }

    cVector3d camPos = m_mainCamera->getLocalPos();
    double volR = 0.5 * cMax(cMax(volumeDims.x(), volumeDims.y()), volumeDims.z());
    if (volR < 1e-6){
        volR = 0.5;
    }
    cVector3d right = m_mainCamera->getRightVector();
    cVector3d up = m_mainCamera->getUpVector();
    // Aim slightly off-center so the burr sits in view without covering volume center.
    cVector3d aim = volumeOrigin + right * (0.05 * volR) + up * (-0.04 * volR);
    cVector3d worldDir = aim - camPos;
    if (worldDir.length() < 1e-6){
        worldDir = cVector3d(1.0, 0.0, 0.0);
    }
    worldDir.normalize();
    cVector3d towardCam = -worldDir;

    double burrR = m_activeDrill ? (m_activeDrill->m_size * 0.5) : 0.02;
    double maxR = burrR;
    for (int i = 1; i < (int)m_toolCursorRadius.size() && i < (int)m_toolCursorList.size(); i++){
        maxR = cMax(maxR, m_toolCursorRadius[i]);
    }
    const double kStandoffMm = 10.0;
    const double aabbPad = 0.05;
    double standoff = cMax(burrR + kStandoffMm * m_units_mmToSim, aabbPad);

    cVector3d geomHalf(volumeDims.x() * 0.5, volumeDims.y() * 0.5, volumeDims.z() * 0.5);
    cVector3d expandedHalf(geomHalf.x() + maxR + aabbPad,
                           geomHalf.y() + maxR + aabbPad,
                           geomHalf.z() + maxR + aabbPad);

    auto anyCursorInside = [&](const cVector3d& tip) -> bool {
        cVector3d n_x = m_T_d.getLocalRot().getCol0() * m_dX;
        int n = (int)m_toolCursorList.size();
        if (n < 1){
            n = 1;
        }
        for (int i = 0; i < n; i++){
            cVector3d P = tip + n_x * i;
            if (inExpandedAabb(worldToVolumeLocal(P, volumePose), expandedHalf)){
                return true;
            }
        }
        return false;
    };

    cVector3d oLocal = worldToVolumeLocal(camPos, volumePose);
    cVector3d dLocal = cTranspose(volumePose.getLocalRot()) * worldDir;
    if (dLocal.length() < 1e-6){
        dLocal = cVector3d(1.0, 0.0, 0.0);
    }
    dLocal.normalize();

    cVector3d tip;
    if (inExpandedAabb(oLocal, expandedHalf)){
        tip = camPos + worldDir * 0.2;
    }
    else{
        double tEnter = 0.0;
        double tExit = 0.0;
        if (rayAabbIntersect(oLocal, dLocal, geomHalf, tEnter, tExit) && tEnter > 1e-6){
            cVector3d hitLocal = oLocal + dLocal * tEnter;
            cVector3d hitWorld = volumeLocalToWorld(hitLocal, volumePose);
            tip = hitWorld - worldDir * standoff;
        }
        else{
            tip = volumeOrigin - worldDir * (volR + standoff);
        }
    }

    const double step = 0.02;
    for (int iter = 0; iter < 80 && anyCursorInside(tip); iter++){
        tip += towardCam * step;
    }

    if (m_hapticDevice){
        m_hapticDevice->setForce(cVector3d(0., 0., 0.));
    }
    m_T_d.setLocalPos(tip);
    if (m_drillReferenceBody){
        m_drillReferenceBody->setLocalTransform(m_T_d);
    }
    toolCursorsPosUpdate(m_T_d);
    toolCursorsInitialize();
    updatePoseFromCursors();
    beginVolumeContactSuppression(10);
    return true;
}

void DrillManager::beginVolumeContactSuppression(int ticks)
{
    m_suppressVolumeContact = true;
    m_suppressVolumeContactTicks = ticks;
}

void DrillManager::tickVolumeContactSuppression(bool anyCursorInContact)
{
    if (!m_suppressVolumeContact){
        return;
    }
    m_suppressVolumeContactTicks--;
    // Always hold at least one physics tick so stale pre-snap contact cannot remove voxels.
    if (m_suppressVolumeContactTicks <= 0){
        m_suppressVolumeContact = false;
        m_suppressVolumeContactTicks = 0;
        return;
    }
    if (!anyCursorInContact && m_suppressVolumeContactTicks <= 8){
        m_suppressVolumeContact = false;
        m_suppressVolumeContactTicks = 0;
    }
}

///
/// \brief This method checks for collision between the tool shaft and the volume.
/// The error between the proxy and goal position of each of the shaft tool cursors is constantly
/// computed. The shaft tool cursor having the maximum error is set as g_targetToolCursor. Further, the
/// position of the drill mesh is set such that it follows the proxy position of the g_targetToolCursor.
/// If there's no collision, the drill mesh follows the proxy position of the shaft tool cursor which is
/// closest to the tip tool cursor.
///
void DrillManager::checkShaftCollision(){

    m_maxError = 0;
    m_targetToolCursor = m_toolCursorList[0];
    m_targetToolCursorIdx = 0;
    for(int i=0; i<m_toolCursorList.size(); i++)
    {

        m_currError = cDistance(m_toolCursorList[i]->m_hapticPoint->getLocalPosProxy(), m_toolCursorList[i]->m_hapticPoint->getLocalPosGoal());

        if(abs(m_currError) > abs(m_maxError + 0.00001))
        {
            m_maxError = m_currError;
            m_targetToolCursor = m_toolCursorList[i];
            m_targetToolCursorIdx = i;
        }
    }
}


///
/// \brief This method updates the position of the drill mesh.
/// After obtaining g_targetToolCursor, the drill mesh adjust it's position and rotation
/// such that it follows the proxy position of the g_targetToolCursor.
///
void DrillManager::updatePoseFromCursors(){
    cMatrix3d newDrillRot;
    newDrillRot = m_toolCursorList[0]->getDeviceLocalRot();
//    cerr << newDrillRot.str(2) << endl;

    if(m_targetToolCursorIdx == 0){
        cTransform T_tip;
        T_tip.setLocalPos(m_toolCursorList[0]->m_hapticPoint->getLocalPosProxy());
        T_tip.setLocalRot(newDrillRot);
        m_activeDrill->m_rigidBody->setLocalTransform(T_tip);
    }
    else if(cDistance(m_targetToolCursor->m_hapticPoint->getLocalPosProxy(), m_targetToolCursor->m_hapticPoint->getLocalPosGoal()) <= 0.001)
    {
        // direction of positive x-axis of drill mesh
        cVector3d xDir = m_activeDrill->m_rigidBody->getLocalRot().getCol0();

        cVector3d newDrillPos;

        // drill mesh will make a sudden jump towards the followSphere
        if(!m_suddenJump)
        {
            newDrillPos = (m_targetToolCursor->m_hapticPoint->getLocalPosProxy() - xDir * m_dX * m_targetToolCursorIdx);
        }

        // drill mesh slowly moves towards the followSphere
        else
        {
            newDrillPos = m_activeDrill->m_rigidBody->getLocalPos() + ((m_targetToolCursor->m_hapticPoint->getLocalPosProxy() - xDir * m_dX * m_targetToolCursorIdx) - m_activeDrill->m_rigidBody->getLocalPos()) * 0.04;
        }

//        cVector3d L = g_targetToolCursor->m_hapticPoint->getLocalPosProxy() - g_toolCursorList[0]->getDeviceLocalPos();

//        cerr << "Colliding Cursor " << g_targetToolCursorIdx << " Error " << L.str(2) << endl;
//        if ( L.length() < 0.01){
//            newDrillRot = g_toolCursorList[0]->getDeviceLocalRot();
//        }
//        else{
//            newDrillRot = afUtils::getRotBetweenVectors<cMatrix3d>(L, cVector3d(1, 0, 0));
//        }

        cTransform trans;
        trans.setLocalPos(newDrillPos);
        trans.setLocalRot(newDrillRot);

//        g_drillRigidBody->setLocalPos(g_drillRigidBody->getLocalPos() + newDrillPos);
        m_activeDrill->m_rigidBody->setLocalTransform(trans);
        if (m_audioSource){
            m_audioSource->setSourcePos(newDrillPos);
        }
    }
}

bool DrillManager::setDrillSizeMm(int burrSizeMm)
{
    string targetName = to_string(burrSizeMm) + "mm";
    for (int di = 0; di < (int)m_drills.size(); di++){
        if (m_drills[di]->m_name == targetName){
            m_activeDrillIdx = di;
            m_activeDrill = m_drills[di];
            applyBurrVisualScale();
            if (!m_toolCursorList.empty() && m_toolCursorList[0]){
                m_toolCursorList[0]->setRadius(m_activeDrill->m_size * 0.5);
            }
            if (m_panelManager && m_sizeLabel){
                m_panelManager->setText(m_sizeLabel, "Drill Type: " + m_activeDrill->m_name);
            }
            showOnlyActive();
            cerr << "INFO! Starting burr set to " << m_activeDrill->m_name << endl;
            return true;
        }
    }
    return false;
}

void DrillManager::cycleDrillTypes(){
    m_activeDrillIdx = (m_activeDrillIdx - 1) % m_drills.size();
    m_activeDrill = m_drills[m_activeDrillIdx];
    m_toolCursorList[0]->setRadius(m_activeDrill->m_size * 0.5);
    applyBurrVisualScale();
    cout << "Drill Type changed to " << m_activeDrill->m_name << endl;
    m_panelManager->setText(m_sizeLabel, "Drill Type: " + m_activeDrill->m_name);

    updatePoseFromCursors();
    showOnlyActive();

    double sim_time = m_activeDrill->m_rigidBody->getCurrentTimeStamp();
    m_drillingPub->publishDrillSize((int)(m_activeDrill->m_size / m_units_mmToSim), sim_time);
}

void DrillManager::applyBurrVisualScale()
{
    if (!m_burrMesh || !m_activeDrill){
        return;
    }
    double newR = m_activeDrill->m_size * 0.5;
    if (newR < 1e-8 || m_burrVisualRadius < 1e-8){
        return;
    }
    if (fabs(newR - m_burrVisualRadius) > 1e-8){
        m_burrMesh->scale(newR / m_burrVisualRadius);
        // The matcap shader draws a vertex buffer. scale() updates CPU
        // positions only; the buffer must be flagged or the flutes stay put.
        for (int mi = 0; mi < m_burrMesh->getNumMeshes(); mi++){
            cMesh* mesh = m_burrMesh->getMesh(mi);
            if (mesh && mesh->m_vertices){
                mesh->m_vertices->m_flagPositionData = true;
                mesh->markForUpdate(false);
            }
        }
        m_burrVisualRadius = newR;
    }
}

void DrillManager::showOnlyActive(){
    for (int di = 0 ; di < m_drills.size() ; di++){
        bool show = false;
        if (di == m_activeDrillIdx){
            show = true;
        }
        m_drills[di]->m_rigidBody->getVisualObject()->setShowEnabled(show);
    }
}
