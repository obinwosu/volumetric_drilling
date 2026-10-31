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
#include "graphics/CPrimitives.h"

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

// The shaft OBJs still carry the old tip sphere. Local +X points from the
// burr toward the handle, and a gap of about 20 mm separates that sphere
// from the neck, so every vertex with X below 10 mm belongs to it. Those
// vertices also anchor the triangles that run back to the neck, so they are
// pulled onto a ring inside the burr instead of deleted. The collision shape
// was already built from its own mesh.
static int tuckBakedTip(cMultiMesh* visual, double cut, double radius)
{
    if (!visual){
        return 0;
    }
    const double ringR = 0.45 * radius;
    const double ringX = 0.20 * radius;
    int moved = 0;
    for (int mi = 0; mi < visual->getNumMeshes(); mi++){
        cMesh* mesh = visual->getMesh(mi);
        if (!mesh || !mesh->m_vertices){
            continue;
        }
        int meshMoved = 0;
        const unsigned int count = mesh->m_vertices->getNumElements();
        for (unsigned int vi = 0; vi < count; vi++){
            cVector3d pos = mesh->m_vertices->getLocalPos(vi);
            if (pos.x() >= cut){
                continue;
            }
            const double radial = sqrt(pos.y() * pos.y() + pos.z() * pos.z());
            if (radial > 1e-8){
                const double scale = ringR / radial;
                mesh->m_vertices->setLocalPos(vi, ringX, pos.y() * scale, pos.z() * scale);
            }
            else{
                mesh->m_vertices->setLocalPos(vi, ringX, 0.0, 0.0);
            }
            meshMoved++;
        }
        if (meshMoved){
            mesh->markForUpdate(false);
            moved += meshMoved;
        }
    }
    return moved;
}

int DrillManager::init(afWorldPtr a_worldPtr, CameraPanelManager* a_panelManager, p_opt::variables_map& var_map){
    // importing drill model
    m_afWorld = a_worldPtr;

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

    const double tipCut = 10.0 * m_units_mmToSim;
    for (int di = 0; di < (int)m_drills.size(); di++){
        const double radius = m_drills[di]->m_size * 0.5;
        int moved = tuckBakedTip(m_drills[di]->m_rigidBody->getVisualObject(), tipCut, radius);
        if (moved){
            cerr << "INFO! Tucked baked tip sphere inside " << m_drills[di]->m_name
                 << " (" << moved << " vertices). Collision mesh unchanged." << endl;
        }
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

    // Diamond tip: unit radius is the outermost grain, same scale as the flutes.
    m_diamondBurrMesh = new cMultiMesh();
    string diamondMeshPath = current_filepath + "/../../ADF/meshes/high_res/diamond_burr.obj";
    if (m_diamondBurrMesh->loadFromFile(diamondMeshPath)){
        m_diamondVisualRadius = 1.0;
        cerr << "INFO! LOADED DIAMOND BURR " << diamondMeshPath << endl;
    }
    else{
        cMesh* diamondMesh = m_diamondBurrMesh->newMesh();
        cCreateSphere(diamondMesh, 1.0, 32, 32);
        diamondMesh->computeAllNormals();
        m_diamondVisualRadius = 1.0;
        cerr << "FAILED TO LOAD DIAMOND BURR FROM " << diamondMeshPath << endl;
    }
    m_diamondBurrMesh->m_material->setWhite();
    m_diamondBurrMesh->m_material->m_specular.set(0.75, 0.75, 0.75);
    m_diamondBurrMesh->m_material->setShininess(80);
    m_diamondBurrMesh->setShowEnabled(false);
    a_worldPtr->addSceneObjectToWorld(m_diamondBurrMesh);

    applyBurrVisualScale();

    // Irrigation tube. Its own mesh so the shaft's silver/gold/black/grey
    // regions do not paint it. Pose follows the drill body, without burr spin.
    for (int di = 0; di < (int)m_drills.size(); di++){
        string cannulaPath = current_filepath + "/../../ADF/meshes/high_res/cannula_"
            + m_drills[di]->m_name + ".obj";
        cMultiMesh* cannula = new cMultiMesh();
        if (!cannula->loadFromFile(cannulaPath)){
            cerr << "FAILED TO LOAD IRRIGATION CANNULA " << cannulaPath << endl;
            delete cannula;
            m_cannulaMeshes.push_back(nullptr);
            continue;
        }
        cannula->m_material->m_ambient.set(0.55, 0.56, 0.58);
        cannula->m_material->m_diffuse.set(0.78, 0.80, 0.82);
        cannula->m_material->m_specular.set(0.96, 0.97, 0.98);
        cannula->m_material->setShininess(110);
        for (int mi = 0; mi < cannula->getNumMeshes(); mi++){
            cMesh* mesh = cannula->getMesh(mi);
            if (!mesh){
                continue;
            }
            mesh->computeAllNormals();
            mesh->m_material->m_ambient.set(0.55, 0.56, 0.58);
            mesh->m_material->m_diffuse.set(0.78, 0.80, 0.82);
            mesh->m_material->m_specular.set(0.96, 0.97, 0.98);
            mesh->m_material->setShininess(110);
        }
        cannula->setShowEnabled(false);
        if (m_drills[di]->m_rigidBody){
            cannula->setLocalTransform(m_drills[di]->m_rigidBody->getLocalTransform());
        }
        a_worldPtr->addSceneObjectToWorld(cannula);
        m_cannulaMeshes.push_back(cannula);
        cerr << "INFO! LOADED IRRIGATION CANNULA " << cannulaPath << endl;
    }

    m_suctionBody = a_worldPtr->getRigidBody("frazier_suction");
    if (!m_suctionBody){
        cerr << "WARNING! No frazier_suction body. Suction will not be tracked or logged." << endl;
    }
    else{
        m_T_s_init = m_suctionBody->getLocalTransform();
        m_T_s = m_T_s_init;
        const int sizes[] = {12, 7, 5, 3};
        for (int fr : sizes){
            string suctionPath = current_filepath + "/../../ADF/meshes/high_res/suction_"
                + to_string(fr) + "fr.obj";
            cMultiMesh* tube = new cMultiMesh();
            if (!tube->loadFromFile(suctionPath)){
                cerr << "FAILED TO LOAD SUCTION " << suctionPath << endl;
                delete tube;
                continue;
            }
            // Satin stainless: same family as the irrigation tube, less shine.
            tube->m_material->m_ambient.set(0.22, 0.23, 0.24);
            tube->m_material->m_diffuse.set(0.64, 0.66, 0.68);
            tube->m_material->m_specular.set(0.45, 0.46, 0.48);
            tube->m_material->setShininess(42);
            for (int mi = 0; mi < tube->getNumMeshes(); mi++){
                cMesh* mesh = tube->getMesh(mi);
                if (!mesh){
                    continue;
                }
                mesh->computeAllNormals();
                mesh->m_material->m_ambient.set(0.22, 0.23, 0.24);
                mesh->m_material->m_diffuse.set(0.64, 0.66, 0.68);
                mesh->m_material->m_specular.set(0.45, 0.46, 0.48);
                mesh->m_material->setShininess(42);
            }
            tube->setShowEnabled(false);
            tube->setLocalTransform(m_T_s);
            a_worldPtr->addSceneObjectToWorld(tube);
            m_suctionMeshes.push_back(tube);
            m_suctionSizesFr.push_back(fr);
            vector<ContactSphere> spheres;
            buildSuctionContacts(fr, spheres);
            m_suctionSpheres.push_back(spheres);
            cerr << "INFO! LOADED SUCTION " << fr << " Fr" << endl;
        }
        m_suctionIdx = 0;
        m_suctionSizeFr = m_suctionSizesFr.empty() ? 12 : m_suctionSizesFr[0];
        showActiveSuction();
    }

    showOnlyActive();

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
        cShaderProgramPtr burrShader = nullptr;
        for (int di = 0; di < (int)m_drills.size(); di++){
            burrShader = m_drills[di]->m_rigidBody->getShaderProgram();
            if (burrShader){
                break;
            }
        }
        auto applyMatcap = [&](cMultiMesh* tip){
            if (!tip || !burrShader){
                return;
            }
            tip->setShaderProgram(burrShader, true);
            burrShader->setUniformi("uMatcapMap", C_TU_METALLIC);
            for (int mi = 0; mi < tip->getNumMeshes(); mi++){
                cMesh* mesh = tip->getMesh(mi);
                if (!mesh){
                    continue;
                }
                mesh->setShaderProgram(burrShader);
                mesh->m_metallicTexture = drillMatCap;
                mesh->m_metallicTexture->setTextureUnit(GL_TEXTURE3);
            }
        };
        applyMatcap(m_burrMesh);
        applyMatcap(m_diamondBurrMesh);
        for (cMultiMesh* tube : m_suctionMeshes){
            applyMatcap(tube);
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
    // (Geomagic Touch / Phantom via OpenHaptics + libhdPhantom; keep AMBF --ndevs 0)
    m_deviceHandler->getDevice(m_hapticDevice, 0);
    if (m_deviceHandler->getNumDevices() > 0 && m_hapticDevice){
        cHapticDeviceInfo info = m_hapticDevice->getSpecifications();
        cout << "INFO! Haptic device [" << info.m_modelName << "] from "
             << info.m_manufacturerName << " ready for drill control." << endl;
    }
    else{
        cout << "WARNING! No haptic device found; use keyboard (Ctrl+WASD/IK) or Ctrl+O for ROS." << endl;
    }

    cout << "INFO! Haptic devices found: " << m_deviceHandler->getNumDevices() << endl;
    m_suctionDevice = nullptr;
    m_suctionTracked = false;
    if (m_deviceHandler->getNumDevices() > 1 && m_deviceHandler->getDevice(m_suctionDevice, 1) && m_suctionDevice){
        // The drill cursor calls open() on device 0. Nothing else opens device 1,
        // and chai3d ignores position, rotation, and velocity until open().
        if (m_suctionDevice->open()){
            cHapticDeviceInfo info = m_suctionDevice->getSpecifications();
            m_suctionTracked = true;
            cout << "INFO! Haptic device [" << info.m_modelName << "] from "
                 << info.m_manufacturerName << " opened for suction control." << endl;
        }
        else{
            m_suctionDevice = nullptr;
            cout << "WARNING! Second haptic device was found but could not be opened." << endl;
        }
    }
    else{
        m_suctionDevice = nullptr;
        cout << "INFO! No second haptic device. USB only shows one Geomagic, so the "
             << "12 Fr suction stays at its start pose. Both devices have to be listed "
             << "by the Touch driver before launch." << endl;
    }

    // Initializing tool cursors
    toolCursorInit(a_worldPtr);
    buildDrillContacts();
    refreshSuctionCursors();
    if (m_hapticDevice && !m_toolCursorList.empty() && m_toolCursorList[0]){
        double ws = m_toolCursorList[0]->getWorkspaceScaleFactor();
        if (ws < 1e-6){
            ws = 1.0;
        }
        m_contactStiffness = m_hapticDevice->getSpecifications().m_maxLinearStiffness / ws;
        cout << "INFO! Drill and suction share one world frame, workspace scale " << ws << endl;
    }

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
        // Reject startup/noise spikes (OpenHaptics can report huge first samples).
        if (!std::isfinite(m_V_i.x()) || !std::isfinite(m_V_i.y()) || !std::isfinite(m_V_i.z())
            || m_V_i.length() > 5.0){
            m_V_i.zero();
        }
        // cMatrix3d extraRotation;
        // extraRotation.setExtrinsicEulerRotationDeg(0, 90, 0, C_EULER_ORDER_XYZ);
        // m_V_i = T_c_w.getLocalRot() * extraRotation * (m_V_i / m_toolCursorList[0]->getWorkspaceScaleFactor());
        double ws = m_toolCursorList[0]->getWorkspaceScaleFactor();
        if (ws < 1e-6){
            ws = 1.0;
        }
        m_V_i = T_c_w.getLocalRot() * (m_V_i / ws);
        m_T_d.setLocalPos(m_T_d.getLocalPos() + (m_V_i * 0.8 * !m_deviceClutch * !m_camClutch * !m_camOrbitBurr));
        m_T_d.setLocalRot(T_c_w.getLocalRot() *  m_T_i.getLocalRot());


        // set zero forces when manipulating objects
        if (m_deviceClutch || m_camClutch || m_camOrbitBurr){
            if (m_camOrbitBurr){
                cVector3d pivot = m_T_d.getLocalPos();
                cMultiMesh* tip = (m_burrStyle == BurrStyle::DIAMOND) ? m_diamondBurrMesh : m_burrMesh;
                if (tip){
                    pivot = tip->getLocalPos();
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
            else if (m_camClutch && !m_vrHmdOwnsCamClutch){
                m_mainCamera->setView(T_c_w.getLocalPos() + m_V_i * !m_deviceClutch, m_mainCamera->getTargetPosLocal(), m_mainCamera->getUpVector());
            }
            m_toolCursorList[0]->setDeviceLocalForce(0.0, 0.0, 0.0);
        }
        m_drillReferenceBody->setLocalTransform(m_T_d);
    }

    updateSuctionPose(T_c_w);
    // Contact force only. The drill pose above is left as the solver wrote it.
    separateInstruments();
    if (m_suctionBody){
        m_suctionBody->setLocalTransform(m_T_s);
    }
    showActiveSuction();

    toolCursorsPosUpdate(m_T_d);

    // check for shaft collision
    checkShaftCollision();

//    if (getOverrideControl() == false){
        // updates position of drill mesh
        updatePoseFromCursors();
//    }

    cMultiMesh* activeTip = (m_burrStyle == BurrStyle::DIAMOND) ? m_diamondBurrMesh : m_burrMesh;
    if (activeTip && m_activeDrill && m_activeDrill->m_rigidBody){
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
        activeTip->setLocalTransform(burrPose);
    }
    if (m_activeDrillIdx >= 0 && m_activeDrillIdx < (int)m_cannulaMeshes.size()){
        cMultiMesh* cannula = m_cannulaMeshes[m_activeDrillIdx];
        if (cannula && m_activeDrill && m_activeDrill->m_rigidBody){
            cannula->setLocalTransform(m_activeDrill->m_rigidBody->getLocalTransform());
        }
    }
    updateSuctionCursors(false);

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
    m_sizeLabel->setText("Burr: " + burrStyleName() + " " + m_activeDrill->m_name);
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

    // Seed cursors at the operative drill pose. Do NOT finish with chai3d's
    // tool->initialize() on the tip cursor: that calls updateFromDevice() and
    // snaps the tip to the physical Touch workspace (drill "disappears").
    toolCursorsPosUpdate(m_T_d);
    toolCursorsInitialize();
    updatePoseFromCursors();
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
        cToolCursor* cursor = m_toolCursorList[i];
        if (!cursor){
            continue;
        }
        // Tip cursor owns the Geomagic/Touch. chai3d::cGenericTool::initialize()
        // always calls updateFromDevice(), which overwrites the virtual drill
        // pose with the scaled physical device pose and throws the mesh out of
        // the camera frustum. Re-init only the proxy model at the seeded pose.
        if (i == 0 && cursor->getHapticDevice()){
            if (cursor->m_hapticPoint){
                cursor->m_hapticPoint->initialize(cursor->getDeviceGlobalPos());
            }
        }
        else{
            cursor->initialize();
        }
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
    if (m_suctionDevice){
        m_suctionDevice->setForce(cVector3d(0., 0., 0.));
    }
    m_instrumentForceDrill.zero();
    m_instrumentForceSuction.zero();
    m_haveDrillDevicePos = false;
    m_haveSuctionDevicePos = false;
    m_T_s = m_T_s_init;
    if (m_suctionBody){
        m_suctionBody->setLocalTransform(m_T_s);
    }
    showActiveSuction();
    updateSuctionCursors(true);
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

bool DrillManager::snapSuctionToCameraView(const cVector3d& volumeOrigin,
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
    // Same aim as the drill recenter: just off the volume center, inside the view.
    cVector3d aim = volumeOrigin + right * (0.05 * volR) + up * (-0.04 * volR);
    cVector3d worldDir = aim - camPos;
    if (worldDir.length() < 1e-6){
        worldDir = cVector3d(1.0, 0.0, 0.0);
    }
    worldDir.normalize();
    cVector3d towardCam = -worldDir;

    const vector<ContactSphere>* spheres = nullptr;
    if (m_suctionIdx >= 0 && m_suctionIdx < (int)m_suctionSpheres.size() && !m_suctionSpheres[m_suctionIdx].empty()){
        spheres = &m_suctionSpheres[m_suctionIdx];
    }
    double tubeR = (spheres && !spheres->empty()) ? (*spheres)[0].radius : 0.012;
    double maxR = tubeR;
    if (spheres){
        for (const ContactSphere& sphere : *spheres){
            maxR = cMax(maxR, sphere.radius);
        }
    }
    const double kStandoffMm = 10.0;
    const double aabbPad = 0.05;
    double standoff = cMax(tubeR + kStandoffMm * m_units_mmToSim, aabbPad);

    cVector3d geomHalf(volumeDims.x() * 0.5, volumeDims.y() * 0.5, volumeDims.z() * 0.5);
    cVector3d expandedHalf(geomHalf.x() + maxR + aabbPad,
                           geomHalf.y() + maxR + aabbPad,
                           geomHalf.z() + maxR + aabbPad);
    const cMatrix3d rot = m_T_s.getLocalRot();

    auto anySuctionInside = [&](const cVector3d& tip) -> bool {
        if (!spheres){
            return inExpandedAabb(worldToVolumeLocal(tip, volumePose), expandedHalf);
        }
        for (const ContactSphere& sphere : *spheres){
            cVector3d P = tip + rot * sphere.local;
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
    for (int iter = 0; iter < 80 && anySuctionInside(tip); iter++){
        tip += towardCam * step;
    }

    m_T_s.setLocalPos(tip);
    m_instrumentForceSuction.zero();
    if (m_suctionDevice){
        m_suctionDevice->setForce(cVector3d(0.0, 0.0, 0.0));
    }
    if (m_suctionBody){
        m_suctionBody->setLocalTransform(m_T_s);
    }
    showActiveSuction();
    updateSuctionCursors(true);
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
            showOnlyActive();
            updateBurrLabel();
            cerr << "INFO! Starting burr set to " << burrStyleName()
                 << " " << m_activeDrill->m_name << endl;
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
    cout << "Burr size changed to " << m_activeDrill->m_name << endl;
    updateBurrLabel();

    updatePoseFromCursors();
    showOnlyActive();

    double sim_time = m_activeDrill->m_rigidBody->getCurrentTimeStamp();
    m_drillingPub->publishDrillSize((int)(m_activeDrill->m_size / m_units_mmToSim), sim_time);
}

void DrillManager::toggleBurrStyle(){
    m_burrStyle = (m_burrStyle == BurrStyle::CUTTING) ? BurrStyle::DIAMOND : BurrStyle::CUTTING;
    showOnlyActive();
    updateBurrLabel();
    cout << "Burr style changed to " << burrStyleName() << endl;
}

string DrillManager::burrStyleName() const{
    return (m_burrStyle == BurrStyle::DIAMOND) ? "Diamond" : "Cutting";
}

void DrillManager::updateBurrLabel(){
    if (!m_panelManager || !m_sizeLabel || !m_activeDrill){
        return;
    }
    string text = "Burr: " + burrStyleName() + " " + m_activeDrill->m_name
                + (m_isOn ? "   DRILL ON" : "   drill off");
    m_panelManager->setText(m_sizeLabel, text);
    cColorf color;
    if (m_isOn){
        color.setGreenDark();
    }
    else{
        color.setBlack();
    }
    m_panelManager->setFontColor(m_sizeLabel, color);
}

void DrillManager::applyBurrVisualScale()
{
    if (!m_activeDrill){
        return;
    }
    double newR = m_activeDrill->m_size * 0.5;
    if (newR < 1e-8){
        return;
    }

    auto scaleTip = [&](cMultiMesh* tip, double& visualRadius){
        if (!tip || visualRadius < 1e-8){
            return;
        }
        if (fabs(newR - visualRadius) <= 1e-8){
            return;
        }
        tip->scale(newR / visualRadius);
        // The matcap shader draws a vertex buffer. scale() updates CPU
        // positions only; the buffer must be flagged or the geometry stays put.
        for (int mi = 0; mi < tip->getNumMeshes(); mi++){
            cMesh* mesh = tip->getMesh(mi);
            if (mesh && mesh->m_vertices){
                mesh->m_vertices->m_flagPositionData = true;
                mesh->markForUpdate(false);
            }
        }
        visualRadius = newR;
    };

    scaleTip(m_burrMesh, m_burrVisualRadius);
    scaleTip(m_diamondBurrMesh, m_diamondVisualRadius);
}

void DrillManager::showOnlyActive(){
    for (int di = 0 ; di < m_drills.size() ; di++){
        bool show = false;
        if (di == m_activeDrillIdx){
            show = true;
        }
        m_drills[di]->m_rigidBody->getVisualObject()->setShowEnabled(show);
    }
    if (m_burrMesh){
        m_burrMesh->setShowEnabled(m_show && m_burrStyle == BurrStyle::CUTTING);
    }
    if (m_diamondBurrMesh){
        m_diamondBurrMesh->setShowEnabled(m_show && m_burrStyle == BurrStyle::DIAMOND);
    }
    for (int di = 0; di < (int)m_cannulaMeshes.size(); di++){
        if (!m_cannulaMeshes[di]){
            continue;
        }
        m_cannulaMeshes[di]->setShowEnabled(m_show && di == m_activeDrillIdx);
    }
}

void DrillManager::showActiveSuction()
{
    for (int i = 0; i < (int)m_suctionMeshes.size(); i++){
        if (!m_suctionMeshes[i]){
            continue;
        }
        const bool show = (i == m_suctionIdx);
        m_suctionMeshes[i]->setShowEnabled(show);
        if (show){
            m_suctionMeshes[i]->setLocalTransform(m_T_s);
        }
    }
}

int DrillManager::cycleSuctionSize()
{
    if (m_suctionSizesFr.empty()){
        return m_suctionSizeFr;
    }
    m_suctionIdx = (m_suctionIdx + 1) % (int)m_suctionSizesFr.size();
    m_suctionSizeFr = m_suctionSizesFr[m_suctionIdx];
    showActiveSuction();
    refreshSuctionCursors();
    cout << "Suction size changed to " << m_suctionSizeFr << " Fr" << endl;
    return m_suctionSizeFr;
}

// 180° about the mesh shaft. Right-multiply so the tip stays put and the bend flips.
static cMatrix3d suctionShaftRoll()
{
    cMatrix3d roll;
    roll.setAxisAngleRotationRad(cVector3d(1.0, 0.0, 0.0), 3.141592653589793);
    return roll;
}

void DrillManager::updateSuctionPose(const cTransform& cameraWorld)
{
    if (!m_suctionTracked || !m_suctionDevice || !m_suctionDevice->isDeviceAvailable()){
        return;
    }
    cTransform devicePose;
    cVector3d velocity;
    m_suctionDevice->getTransform(devicePose);
    m_suctionDevice->getLinearVelocity(velocity);
    if (!std::isfinite(velocity.x()) || !std::isfinite(velocity.y()) || !std::isfinite(velocity.z())
        || velocity.length() > 5.0){
        velocity.zero();
    }
    double ws = 1.0;
    if (!m_toolCursorList.empty() && m_toolCursorList[0]){
        ws = m_toolCursorList[0]->getWorkspaceScaleFactor();
        if (ws < 1e-6){
            ws = 1.0;
        }
    }
    velocity = cameraWorld.getLocalRot() * (velocity / ws);
    // Same integration as the drill. Button 1 freezes translation only.
    // The tip is the body origin, so the tube orbits that point when the hand rotates.
    // Roll 180° about the shaft (+X) so the bend stays on the same side as the parked pose.
    m_T_s.setLocalPos(m_T_s.getLocalPos() + velocity * 0.8 * !m_suctionClutch);
    m_T_s.setLocalRot(cameraWorld.getLocalRot() * devicePose.getLocalRot() * suctionShaftRoll());
}

void DrillManager::shiftDrillHome(const cVector3d& offset)
{
    m_T_d_init.setLocalPos(m_T_d_init.getLocalPos() + offset);
    m_T_d = m_T_d_init;
    if (m_drillReferenceBody){
        m_drillReferenceBody->setLocalTransform(m_T_d);
    }
    if (!m_toolCursorList.empty()){
        toolCursorsPosUpdate(m_T_d);
        toolCursorsInitialize();
        updatePoseFromCursors();
    }
}

void DrillManager::placeSuctionHome(const cVector3d& tip, const cMatrix3d& rotation)
{
    m_T_s_init.setLocalPos(tip);
    m_T_s_init.setLocalRot(rotation * suctionShaftRoll());
    m_T_s = m_T_s_init;
    if (m_suctionBody){
        m_suctionBody->setLocalTransform(m_T_s);
    }
    showActiveSuction();
    updateSuctionCursors(true);
}

void DrillManager::refreshSuctionCursors()
{
    m_suctionCursorSphere.clear();
    if (!m_afWorld || m_suctionIdx < 0 || m_suctionIdx >= (int)m_suctionSpheres.size()){
        return;
    }
    const vector<ContactSphere>& spheres = m_suctionSpheres[m_suctionIdx];
    if (spheres.empty()){
        return;
    }
    // Distal tip first. Spacing is about 2 mm so the thin tube cannot slip between proxies.
    const double spacing = 0.02;
    const int kMax = 12;
    m_suctionCursorSphere.push_back(0);
    cVector3d last = spheres[0].local;
    for (int i = 1; i < (int)spheres.size() && (int)m_suctionCursorSphere.size() < kMax; i++){
        if ((spheres[i].local - last).length() < spacing){
            continue;
        }
        m_suctionCursorSphere.push_back(i);
        last = spheres[i].local;
    }
    while ((int)m_suctionCursors.size() < (int)m_suctionCursorSphere.size()){
        cToolCursor* cursor = new cToolCursor(m_afWorld->getChaiWorld());
        m_afWorld->addSceneObjectToWorld(cursor);
        cursor->setShowEnabled(false);
        cursor->setShowContactPoints(false, false);
        if (cursor->m_hapticPoint){
            cursor->m_hapticPoint->setShow(false, false);
            if (cursor->m_hapticPoint->m_sphereProxy){
                cursor->m_hapticPoint->m_sphereProxy->setShowEnabled(false);
            }
            if (cursor->m_hapticPoint->m_sphereGoal){
                cursor->m_hapticPoint->m_sphereGoal->setShowEnabled(false);
            }
        }
        m_suctionCursors.push_back(cursor);
    }
    updateSuctionCursors(true);
    cout << "INFO! Suction bone contact uses " << m_suctionCursorSphere.size()
         << " proxies on the " << m_suctionSizeFr << " Fr tube." << endl;
}

void DrillManager::updateSuctionCursors(bool snapProxies)
{
    if (m_suctionIdx < 0 || m_suctionIdx >= (int)m_suctionSpheres.size()){
        return;
    }
    const vector<ContactSphere>& spheres = m_suctionSpheres[m_suctionIdx];
    const int active = (int)m_suctionCursorSphere.size();
    for (int i = 0; i < (int)m_suctionCursors.size(); i++){
        cToolCursor* cursor = m_suctionCursors[i];
        if (!cursor){
            continue;
        }
        if (i >= active || m_suctionCursorSphere[i] < 0 || m_suctionCursorSphere[i] >= (int)spheres.size()){
            cursor->setDeviceLocalForce(cVector3d(0.0, 0.0, 0.0));
            continue;
        }
        const ContactSphere& sphere = spheres[m_suctionCursorSphere[i]];
        const cVector3d world = m_T_s.getLocalPos() + m_T_s.getLocalRot() * sphere.local;
        cursor->setRadius(sphere.radius);
        cursor->setDeviceLocalPos(world);
        cursor->setDeviceLocalRot(m_T_s.getLocalRot());
        if (snapProxies && cursor->m_hapticPoint){
            cursor->m_hapticPoint->initialize(cursor->getDeviceGlobalPos());
        }
    }
}

void DrillManager::computeSuctionInteractionForces()
{
    const int active = (int)m_suctionCursorSphere.size();
    for (int i = 0; i < (int)m_suctionCursors.size(); i++){
        cToolCursor* cursor = m_suctionCursors[i];
        if (!cursor){
            continue;
        }
        if (i < active){
            cursor->computeInteractionForces();
        }
        else{
            cursor->setDeviceLocalForce(cVector3d(0.0, 0.0, 0.0));
        }
    }
}

cVector3d DrillManager::suctionVolumeForce() const
{
    cVector3d sum(0.0, 0.0, 0.0);
    const int active = (int)m_suctionCursorSphere.size();
    for (int i = 0; i < active && i < (int)m_suctionCursors.size(); i++){
        if (m_suctionCursors[i]){
            sum.add(m_suctionCursors[i]->getDeviceLocalForce());
        }
    }
    return sum;
}

void DrillManager::buildDrillContacts()
{
    m_drillSpheres.clear();
    m_drillSpheres.resize(m_drills.size());
    for (int di = 0; di < (int)m_drills.size(); di++){
        vector<ContactSphere>& out = m_drillSpheres[di];
        double tipRadius = m_drills[di]->m_size * 0.5;
        out.push_back({cVector3d(0.0, 0.0, 0.0), tipRadius});
        cMultiMesh* visual = m_drills[di]->m_rigidBody
            ? m_drills[di]->m_rigidBody->getVisualObject()
            : nullptr;
        if (!visual){
            continue;
        }
        const double bin = 0.02;
        vector<double> radiusAt;
        for (int mi = 0; mi < visual->getNumMeshes(); mi++){
            cMesh* mesh = visual->getMesh(mi);
            if (!mesh || !mesh->m_vertices){
                continue;
            }
            const unsigned int count = mesh->m_vertices->getNumElements();
            for (unsigned int vi = 0; vi < count; vi++){
                cVector3d pos = mesh->m_vertices->getLocalPos(vi);
                if (pos.x() < 0.004){
                    continue;
                }
                int slot = (int)(pos.x() / bin);
                if (slot < 0){
                    continue;
                }
                if (slot >= (int)radiusAt.size()){
                    radiusAt.resize(slot + 1, 0.0);
                }
                double radial = sqrt(pos.y() * pos.y() + pos.z() * pos.z());
                radiusAt[slot] = cMax(radiusAt[slot], radial);
            }
        }
        for (int slot = 1; slot < (int)radiusAt.size(); slot++){
            if (radiusAt[slot] <= 0.0){
                continue;
            }
            out.push_back({cVector3d((slot + 0.5) * bin, 0.0, 0.0), cMax(tipRadius, radiusAt[slot])});
        }
    }
}

void DrillManager::buildSuctionContacts(int fr, vector<ContactSphere>& out)
{
    out.clear();
    const double tubeRadius = 0.5 * (fr / 3.0) * m_units_mmToSim;
    const double bendStart = 100.0 * m_units_mmToSim;
    const double bendRadius = 18.0 * m_units_mmToSim;
    const double step = cMax(0.008, 0.75 * tubeRadius);
    for (double x = 0.0; x <= bendStart; x += step){
        out.push_back({cVector3d(x, 0.0, 0.0), tubeRadius});
    }
    const double bendAngle = 35.0 * 3.141592653589793 / 180.0;
    for (int deg = 4; deg <= 35; deg += 4){
        const double angle = deg * 3.141592653589793 / 180.0;
        out.push_back({cVector3d(bendStart + bendRadius * sin(angle),
                                 bendRadius * (1.0 - cos(angle)),
                                 0.0), tubeRadius});
    }
    cVector3d exitPos(bendStart + bendRadius * sin(bendAngle),
                      bendRadius * (1.0 - cos(bendAngle)),
                      0.0);
    cVector3d axial(cos(bendAngle), sin(bendAngle), 0.0);
    for (double distMm = 2.0; distMm <= 26.5; distMm += 3.0){
        double outerMm = fr / 3.0;
        if (distMm >= 5.5 && distMm <= 19.0){
            outerMm = 7.2;
        }
        else if (distMm > 19.0){
            outerMm = 5.2;
        }
        else if (distMm >= 4.0){
            outerMm = cMax(fr / 3.0, 6.4);
        }
        out.push_back({exitPos + axial * (distMm * m_units_mmToSim), 0.5 * outerMm * m_units_mmToSim});
    }
}

void DrillManager::separateInstruments()
{
    m_instrumentForceDrill.zero();
    m_instrumentForceSuction.zero();
    if (!m_activeDrill || m_drillSpheres.empty() || m_suctionSpheres.empty()){
        return;
    }
    if (m_activeDrillIdx < 0 || m_activeDrillIdx >= (int)m_drillSpheres.size()){
        return;
    }
    if (m_suctionIdx < 0 || m_suctionIdx >= (int)m_suctionSpheres.size()){
        return;
    }

    const vector<ContactSphere>& drillSpheres = m_drillSpheres[m_activeDrillIdx];
    const vector<ContactSphere>& suctionSpheres = m_suctionSpheres[m_suctionIdx];
    const cVector3d drillPos = m_T_d.getLocalPos();
    const cMatrix3d drillRot = m_T_d.getLocalRot();
    const cVector3d suctionPos = m_T_s.getLocalPos();
    const cMatrix3d suctionRot = m_T_s.getLocalRot();
    // Ignore less than about 0.3 mm so a resting touch does not buzz.
    const double ignore = 0.0003;
    double maxPen = 0.0;
    cVector3d normal(0.0, 0.0, 1.0);
    for (const ContactSphere& drillSphere : drillSpheres){
        const cVector3d drillCenter = drillPos + drillRot * drillSphere.local;
        for (const ContactSphere& suctionSphere : suctionSpheres){
            const cVector3d suctionCenter = suctionPos + suctionRot * suctionSphere.local;
            const cVector3d gap = drillCenter - suctionCenter;
            const double dist = gap.length();
            const double pen = drillSphere.radius + suctionSphere.radius - dist;
            if (pen > maxPen){
                maxPen = pen;
                normal = (dist > 1e-6) ? gap / dist : cVector3d(0.0, 0.0, 1.0);
            }
        }
    }
    if (maxPen <= ignore || m_contactStiffness <= 0.0){
        return;
    }
    // Equal and opposite. Poses stay on their own tracks.
    const cVector3d push = normal * (m_contactStiffness * maxPen);
    m_instrumentForceDrill = push;
    m_instrumentForceSuction = -push;
}

