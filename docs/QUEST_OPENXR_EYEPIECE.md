# Quest eyepiece pipeline (WiVRn + OpenXR)

Operating-microscope model: headset shows a **head-locked stereo view of `main_camera`**.
HMD 6DoF moves `main_camera` **only while cam-clutch is held** (stylus btn 1).

This replaces ALVR → SteamVR → OpenVR *scene* Submit (which always reprojects the world).

## What to install

### Linux PC

```bash
# OpenXR headers/loader (build the AMBF plugin)
sudo apt-get update
sudo apt-get install -y libopenxr-dev libopenxr-loader1 libglx-dev libx11-dev

# WiVRn dashboard + OpenXR runtime (Flatpak — recommended)
sudo apt-get install -y flatpak
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install -y flathub io.github.wivrn.wivrn
```

Or use the helper:

```bash
bash ~/ros_ambf_ws/src/volumetric_drilling/scripts/setup_wivrn_openxr.sh
```

Build the eyepiece plugin:

```bash
bash ~/ros_ambf_ws/src/volumetric_drilling/scripts/build_openxr_eyepiece.sh
```

### Meta Quest

1. Enable **Developer Mode** (Meta Horizon / SideQuest / device settings) — already needed for ALVR USB.
2. Install the **WiVRn** client:
   - Easiest: open **WiVRn dashboard** on the PC once; the wizard links the **Meta Store** listing, **or**
   - Sideload matching APK from [WiVRn-APK releases](https://github.com/WiVRn/WiVRn-APK/releases) via `adb install …apk`
3. PC server and Quest client **must be the same WiVRn version**.

You do **not** need SteamVR or ALVR for this path (can leave them installed unused).

## Wired connect (USB)

```bash
# Terminal A — WiVRn dashboard
flatpak run io.github.wivrn.wivrn

# Terminal B — after Quest plugged in + Developer Mode
adb reverse tcp:9757 tcp:9757
# Meta Store package id (change if you sideloaded):
adb shell am start -a android.intent.action.VIEW -d "wivrn+tcp://localhost" org.meumeu.wivrn
```

On the Quest: WiVRn should show **Connection ready**. Then start AMBF with the OpenXR eyepiece ADF.

Helper:

```bash
bash ~/ros_ambf_ws/src/volumetric_drilling/scripts/study_gui/start_wivrn_eyepiece_session.sh
```

## Launch the sim

Study GUI: check **OpenXR eyepiece (WiVRn)** (uses launch ADF index **15**), leave OpenVR/SteamVR unchecked.

Or CLI (example):

```bash
export XR_RUNTIME_JSON="${HOME}/.config/openxr/1/active_runtime.json"   # set by WiVRn when connected
# confirm it points at WiVRn, not Monado, while testing:
#   cat "$XR_RUNTIME_JSON"

cd ~/ros_ambf_ws/src/volumetric_drilling
# …same ambf launch as study GUI, with -l including 15 (openxr_eyepiece_camera.yaml)
```

## Expected behavior

| Cam-clutch (stylus btn 1 or hold Alt) | HMD |
|---------------------------|-----|
| Released | Stereo locked to `main_camera` (+ ADF dolly). Head motion does **not** orbit the anatomy. |
| Held | HMD 6DoF updates `main_camera` (microscope body). Release locks pose. |

### Seated scene placement (lap / keyboard pose)

By default the WiVRn stereo layer sits in **LOCAL** space so you can place it like a virtual screen (typically down and forward, as if looking at a keyboard in your lap). Looking away leaves the panel in the room; look back to see it again.

| Key | Action |
|-----|--------|
| **F10** (hold) | Drag: scene follows your head (with lap offset). Release to lock. |
| **F11** | Snap to lap pose relative to current head (down + forward). |
| **F12** | Snap to current look direction (flat in front of eyes). |

First tracking sample also auto-places in the lap pose. For the old head-glued eyepiece: `export AMBF_OPENXR_LAYER_SPACE=view`.

Console should log `OpenXR eyepiece` init, placement lines, and `cam-clutch PRESSED/RELEASED`.

## Zoom (FOV vs telephoto)

Same as before: edit `ADF/openxr_eyepiece_camera.yaml` — more **negative** `location.x` / `look at.x` dollies closer.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Plugin init fails / no runtime | WiVRn dashboard running? Quest **Connection ready**? `echo $XR_RUNTIME_JSON` / `active_runtime.json` → WiVRn |
| Black HMD | Wired: `adb reverse` again; NVIDIA encode path; restart WiVRn then AMBF |
| Still “slides” like SteamVR | Confirm you are **not** loading ADF 14 OpenVR; only ADF 15 eyepiece |
| Clutch does nothing | Stylus btn 1; look for clutch log lines; drilling plugin built with `ambf_drilling_cam_clutch` |

## Relation to old OpenVR path

| | OpenVR (ADF 14) | OpenXR eyepiece (ADF 15) |
|--|-----------------|---------------------------|
| Transport | ALVR + SteamVR | WiVRn |
| Submit | OpenVR scene textures | OpenXR **VIEW-space** projection (head-locked stereo) |
| Microscope clutch | Partial / fights compositor | Intended model |

## RTX view in the headset

The path-traced RTX view can replace AMBF's own rendering in the headset.

**Turning it on**
- **Study GUI:** check both **OpenXR eyepiece** and **RTX view**, then Start Simulation. The GUI starts the RTX window with `--headset`.
- **By hand:** add `--headset` to `scripts/rtx_view/fivrs_rtx_view.py`.

**What happens**
- **Two eyes like a microscope:** the RTX window renders two eyes, 23 mm apart, both aimed at the field. Its own window shows the left eye.
- **Hand-off to the plugin:** each frame both images go to `/dev/shm/fivrs_rtx_eyes`.
- **What the headset shows:** while new frames arrive, the eyepiece plugin (ADF 15) shows them instead of AMBF's rendering. If no RTX frame arrives for 1 s, it switches back to AMBF's rendering, and the AMBF log says which one it is showing.
- **What still works:** the clutch, recenter (F10/F11/F12) and placement work as before, and zoom (Ctrl + stylus) changes both eyes.

**Cost:** two path-traced eyes cost about twice one view, so expect a lower FPS than the desktop RTX window alone.

**Settings**

| Setting | Where | Default | Effect |
|---|---|---|---|
| `--stereo-base-mm` | RTX window | 23 | Distance between the eyes. More = stronger depth. |
| `--stereo-focus-mm` | RTX window | scope to volume centre | Depth with no disparity, where the eyes converge. |
| `--width`, `--height` | RTX window | 1152 × 864 with `--headset` | Size of each eye, and the window. |
| `AMBF_OPENXR_RTX_EYEPIECE_DEG` | AMBF environment | 65 | How wide the eyepiece image looks in the headset (degrees). |
| `AMBF_OPENXR_RTX_FUSE_M` | AMBF environment | 1.5 | Distance (m) at which the two images fuse. |
| `AMBF_OPENXR_RTX=0` | AMBF environment | on | Always use AMBF's own rendering. |

**Not shown in RTX mode:** the AMBF HUD and pause menu are drawn by AMBF, so they don't appear while the headset shows the RTX eyes.

**Rebuilding the plugin:** `bash scripts/build_openxr_eyepiece.sh`. A copy of the plugin source from before this change is in `~/plugin_backups/ambf_openxr_eyepiece_2026-11-02`, because that folder has no git.

**Sheared picture with sawtooth lines from the first drill contact (fixed).** CHAI's partial voxel-texture updates leave OpenGL's unpack row length and skips set, and the plugin used to upload the RTX eyes with them. The plugin now uploads with packed rows and restores AMBF's settings.

**Headset brighter than the desktop (fixed in RTX-only mode).** The RTX images are sRGB-encoded; the plugin now asks for an sRGB swapchain so the compositor does not encode them twice. AMBF's log prints the format it got.

**Picture breaks up with blocky corruption.** This is the WiVRn video stream losing data, not the renderer. WiVRn's log shows "IDR frame needed": the Quest has lost part of the stream and asked for a fresh full frame. Drilling changes the picture a lot at once, and over Wi-Fi at about 25 Mbit/s the stream drops parts of it. Fixes, in order:
1. **Use the USB cable.** If `adb devices` says "no permissions", add a udev rule for the Quest (vendor 2833), then replug the cable and run `start_wivrn_eyepiece_session.sh` again so it can set `adb reverse`.
2. **Otherwise use a strong 5 or 6 GHz Wi-Fi link, close to the router.**
3. **Change the encoder** in the WiVRn dashboard settings: H.265 rather than AV1, and a fixed bitrate the link can carry. Set it higher on USB and lower on weak Wi-Fi.

The server log is in `~/.var/app/io.github.wivrn.wivrn/.local/state/wivrn/wivrn-dashboard/`.
