# Meta Quest HMD via OpenVR (ALVR + SteamVR)

This is the supported path for Ubuntu 24.04 + AMBF3 on the A51, using:

- [AMBF-ORG/ambf_openvr_plugin](https://github.com/AMBF-ORG/ambf_openvr_plugin) — submits AMBF stereo frames to SteamVR and tracks the HMD
- **ALVR** — streams SteamVR to Meta Quest over Wi‑Fi (official Quest Link is Windows-only)
- **SteamVR** — OpenVR runtime

## 1. One-time PC setup

### OpenVR SDK (done if you used the helper script)
```bash
sudo apt-get install -y libopenvr-dev libopenvr-api1t64
```

### Build the AMBF OpenVR plugin
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros_ambf_ws/install/setup.bash
export CMAKE_PREFIX_PATH="$HOME/ros_ambf_ws/install/AMBF:${CMAKE_PREFIX_PATH}"
cd ~/ros_ambf_ws/src/ambf_openvr_plugin
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
# produces libambf_openvr_camera.so
```

Or: `bash ~/ros_ambf_ws/src/volumetric_drilling/scripts/build_openvr_plugin.sh`

### Steam + SteamVR
1. Install Steam (Ubuntu): https://store.steampowered.com/about/
2. In Steam → Library → Tools → install **SteamVR**
3. Launch SteamVR once and complete room setup (seated is fine for the simulator)
4. **Hybrid NVIDIA (A51 / RTX):** set SteamVR launch options (Steam → Library → Tools → SteamVR → Properties → Launch Options) to:
   ```text
   __GLX_VENDOR_LIBRARY_NAME=nvidia __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json /home/obihome/.local/share/Steam/steamapps/common/SteamVR/bin/vrmonitor.sh %command%
   ```
   `start_quest_vr_session.sh` also writes this automatically. Restart SteamVR after changing it.
   The `vrmonitor.sh` wrap is **required by ALVR on Linux** (black HMD otherwise). Keep the NVIDIA env prefix on hybrid systems.
5. In SteamVR settings (or via `steamvr.vrsettings`): turn **off SteamVR Home** (`enableHomeApp: false`) so the Steam menu/environment does not overlay the sim.
6. Optional (helps ALVR’s VA-API probe find NVIDIA instead of Intel):
   ```bash
   sudo apt-get install -y nvidia-vaapi-driver
   ```

### ALVR (Quest ↔ SteamVR) — A51 paths
PC streamer extract (this machine):
```text
/home/obihome/Downloads/alvr_streamer_linux/
  bin/alvr_dashboard
  lib64/
```

Launchers installed for the dash:
- **ALVR Dashboard** — starts the streamer only
- **Volumetric Drilling (Quest VR)** — starts ALVR (+ SteamVR if installed), then the study GUI

Or from a terminal:
```bash
# ALVR only
env LD_LIBRARY_PATH=$HOME/Downloads/alvr_streamer_linux/lib64 \
  $HOME/Downloads/alvr_streamer_linux/bin/alvr_dashboard

# ALVR + study GUI helper
~/ros_ambf_ws/src/volumetric_drilling/scripts/study_gui/start_quest_vr_session.sh --gui
```

1. Quest: install **ALVR** client (SideQuest / Meta store / ALVR docs)
2. Same low-latency Wi‑Fi (prefer 5/6 GHz; wired PC if possible)
3. Start ALVR on PC → put on Quest → connect → SteamVR shows headset Ready

## 2. Run the drilling simulator in the headset

1. Start **SteamVR** and connect the Quest through **ALVR** (headset status OK)
2. Open the study GUI (dash icon or `launch_study_gui.sh`)
3. Check **OpenVR HMD**
4. Select an anatomy → **Start Simulation**

That adds ADF index **14** (`ADF/openvr_camera.yaml`) on top of drills + gaze marker.

**Intended layout:** PC window = world `main_camera` (mono). Headset = `openvr_cam` stereo submits via SteamVR (`visible: false`, so it does not own a desktop window). Leave **Stream Stereo / Stream Depth** unchecked when using OpenVR HMD.

### CLI smoke test (without study GUI)
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros_ambf_ws/install/setup.bash
cd ~/ros_ambf_ws/src/volumetric_drilling
ambf_simulator \
  --launch_file "$PWD/launch.yaml" \
  -l 0,7,14 \
  -a "$PWD/ADF/volume_256_RT143.yaml"
```

Optional tracking space (plugin default is seated):
```bash
export AMBF_OPENVR_TRACKING_SPACE=standing   # or omit / seated
```

## 3. Layout in this repo

| Piece | Location |
| --- | --- |
| OpenVR plugin source | `~/ros_ambf_ws/src/ambf_openvr_plugin` |
| Built `.so` | `~/ros_ambf_ws/src/ambf_openvr_plugin/build/libambf_openvr_camera.so` |
| Drilling ADF | `ADF/openvr_camera.yaml` (launch index **14**) |
| Upstream example ADF | `ambf_openvr_plugin/ADF/camera.yaml` |

## 4. Troubleshooting

| Symptom | Check |
| --- | --- |
| Plugin init fails / no HMD | SteamVR running? ALVR connected? `libopenvr` installed? |
| Black headset, desktop OK | SteamVR showing headset Ready; firewall allowing ALVR |
| **Black headset, but PC camera tracks head motion** | Poses work; video path is on the wrong GPU. Apply SteamVR NVIDIA launch options (above), fully quit SteamVR + ALVR, relaunch via `start_quest_vr_session.sh`, reconnect Quest, then start sim with **OpenVR HMD**. ALVR log should stop warning about NVIDIA launch options; ideally encoder is not stuck on `Intel iHD`. Install `nvidia-vaapi-driver` if `nvidia_drv_video.so` is missing. |
| **Headset still too far / zoomed out** | Edit `ADF/openvr_camera.yaml`: more **negative** `location.x` / `look at.x` dollies closer (e.g. `-2.5` / `-3.5`, then `-2.8` / `-3.8`). **Positive X moves away.** Plugin must honor ADF (rebuild if an older build forced identity). Restart AMBF; console prints `OpenVR dolly (local pos)`. |
| **Delayed slide when turning head (no clutch)** | Inherent to Quest/ALVR + a locked camera: the headset still late-warps frames. Mitigations: plugin `Submit_TextureWithPose` + frame discontinuity; `start_quest_vr_session.sh` sets SteamVR `disableAsync` / `enableLinuxVulkanAsync=false`; fully restart SteamVR after that. Prefer **wired** ALVR. You cannot fully eliminate headset-side reprojection on Quest. |
| **Jumpy HMD tracking** | Often chaperone/room-setup thrash + full 6DoF over wireless/USB. Use position-lock (default), seated tracking, finish SteamVR room setup once, prefer wired ALVR. SteamVR **303** with a green headset usually means reconnect ALVR / restart SteamVR (compositor), not a real reinstall. |
| **SteamVR menu / Home in the headset** | Disable SteamVR Home only (`enableHomeApp: false`). `start_quest_vr_session.sh` stubs `steamtours.sh`. **Keep the SteamVR dashboard enabled** — turning it off causes Fail **-203** and a black headset. |
| **SteamVR Fail (-203) / restart loop** | Compositor died or never started (often after sim quit, or HMD not Streaming yet). Keep `dashboard.enableDashboard: true`, dismiss restart if SteamVR recovers after ALVR shows **Streaming**, or fully restart SteamVR then reconnect wired/Wi‑Fi ALVR **before** launching the sim. |
| **PC sim window frozen / shows desktop behind it** | Usually `__NV_PRIME_RENDER_OFFLOAD` while the desktop is already on NVIDIA. Launch via `launch_study_gui.sh` (offload unset). `openvr_cam` is `visible: false`. Restart the study GUI. Do not enable Stream Stereo with OpenVR HMD. |
| **Black HMD but SteamVR headset icon green / presents happening** | Missing SteamVR `vrmonitor.sh` launch wrap (ALVR Linux mandatory). Re-run `start_quest_vr_session.sh` or set launch options per §1, restart SteamVR, reconnect ALVR, then start sim. |
| **SteamVR error 303** | Compositor IPC failed (`Unable to get compositor interface`). Use the NVIDIA `%command%` launch options **without** `vrmonitor.sh`, restart SteamVR, stay on SteamVR stable (not beta). Reinstall is rarely the real fix. |
| **Black after SteamVR restart** | Normal until Quest reconnects in the ALVR app. Status should be **Streaming**, SteamVR headset Ready, then start sim with OpenVR HMD. |
| **Laggy haptic / motion** | Wireless ALVR adds latency; prefer wired USB (below). On Wi‑Fi: ~50 Mbps, buffering 1, 5/6 GHz, PC Ethernet if possible. |
| Wrong facing / origin | Camera initial pose in `ADF/openvr_camera.yaml`; try seated vs standing env |
| `.so` not found | Rebuild plugin; confirm `path:` in `openvr_camera.yaml` points at `ambf_openvr_plugin/build/` |
| Colcon picks up openvr repo | Add `COLCON_IGNORE` in `ambf_openvr_plugin/` (standalone CMake build) |

## 5. Wired ALVR (USB) — preferred for latency

Official Quest Link is Windows-only. On Linux, **wired = ALVR over USB** (ADB port-forward), not Meta Link.

### One-time on the Quest (needs the headset owner / Meta account admin)
Ask the person who owns the Quest / org admin (e.g. Adnan) to:

1. Put the headset in **Developer Mode** (Meta Horizon / developer.oculus.com device setup — not just “USB debugging” in a random app).
2. On the Quest: **Settings → System → Developer → USB debugging = ON** (appears after Developer Mode).
3. Optionally enable **Developer Mode** via the Meta Horizon mobile app for that device if it is not already.
4. Confirm which Meta account is the device admin if Developer Mode is greyed out.

PC side does **not** need them to “refresh Horizon”; it needs **Developer Mode + USB debugging authorized for this PC**.

### Every session on this PC
1. USB-C cable Quest ↔ A51 (USB 3 preferred).
2. Put headset on → accept **Allow USB debugging?** → Always allow from this computer.
3. ALVR Dashboard → **Devices** → enable **Wired Connection** (ALVR 20.12+ bundles ADB).
4. **Connection → Wired Client Type** must match how ALVR was installed on the Quest (Store vs GitHub).
5. Quest ALVR client → connect (stream goes over USB; Wi‑Fi can stay on).

Docs: [ALVR wired setup](https://github.com/alvr-org/ALVR/wiki/ALVR-wired-setup-(ALVR-over-USB))  
Meta device setup: [mobile-device-setup](https://developers.meta.com/horizon/documentation/native/android/mobile-device-setup/)

### Copy-paste ask for Adnan
```text
Subject: Quest 3 — enable Developer Mode + USB debugging for ALVR wired on A51

We need the Quest 3 set up so ALVR can stream over USB to the A51 (Ubuntu).
Official Quest Link is Windows-only; wired on Linux is ALVR + ADB.

Please on the Quest (and Meta admin account if needed):
1. Enable Developer Mode for this headset (Meta Horizon developer device setup).
2. Settings → System → Developer → turn ON "USB debugging".
3. When we plug into the A51, accept "Allow USB debugging?" and check Always allow.

No need to change Horizon app streaming settings — this is for ADB/ALVR, not Meta Link.
After that we can use ALVR Dashboard → Devices → Wired Connection on the A51.
```

## References

- Plugin: https://github.com/AMBF-ORG/ambf_openvr_plugin  
- ALVR: https://github.com/alvr-org/ALVR  
- ALVR wired USB: https://github.com/alvr-org/ALVR/wiki/ALVR-wired-setup-(ALVR-over-USB)  
- SteamVR on Linux: install via Steam → Tools  
