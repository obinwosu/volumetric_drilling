# Desk MR passthrough trial (SEPARATE from eyepiece)

This is a **separate experimental path**. It does **not** replace or modify the
OpenXR eyepiece (`ambf_openxr_eyepiece`, ADF 15) or OpenVR (ADF 14) workflows.

| | Eyepiece (production) | Desk MR (this trial) |
|--|----------------------|----------------------|
| Plugin | `ambf_openxr_eyepiece` | `ambf_openxr_desk_mr` |
| Launch ADF | **15** | **16** |
| Metaphor | Microscope / cam-clutch | Volume on real desk |
| Blend | Opaque VR | `ALPHA_BLEND` passthrough |
| Study GUI | OpenXR eyepiece (WiVRn) | Desk MR passthrough (trial) |

## Build (once)

```bash
bash ~/ros_ambf_ws/src/volumetric_drilling/scripts/build_openxr_desk_mr.sh
# rebuild drilling plugin if you pulled volume-place hotkeys:
cd ~/ros_ambf_ws && colcon build --packages-select volumetric_plugin
```

Eyepiece rebuild is unchanged:

```bash
bash ~/ros_ambf_ws/src/volumetric_drilling/scripts/build_openxr_eyepiece.sh
```

## Run

1. Start WiVRn and connect the Quest (**Connection ready**), same as eyepiece.
   Helper: `bash scripts/study_gui/start_wivrn_desk_mr_session.sh`
2. Study GUI → check **Desk MR passthrough (trial)** only (leave eyepiece / OpenVR unchecked).
3. Select anatomy → Start Simulation.

CLI smoke test:

```bash
export XR_RUNTIME_JSON="${HOME}/.config/openxr/1/active_runtime.json"
cd ~/ros_ambf_ws/src/volumetric_drilling
ambf_simulator \
  --launch_file "$PWD/launch.yaml" \
  -l 0,7,16 \
  -a "$PWD/ADF/volume_256_RT143.yaml"
```

## Place the volume on the desk

With Desk MR running (Ctrl+Shift held):

| Keys | Action |
|------|--------|
| Arrows | Nudge forward/back/left/right (~2 cm) |
| PgUp / PgDn | Raise / lower |
| `,` / `.` | Yaw |
| `-` / `=` | Scale down / up |
| `S` | Save pose → `~/.config/fivrs/desk_mr_volume_pose.yaml` |
| `L` | Reload saved pose (or default if none) |
| **F9** | Resnap **drill** beside the volume |

On first HMD track the volume snaps in front of you and the **drill** is placed next to it (both visible in passthrough). Omni still drives the tip in the camera frame — not full desk mm co-location yet.

## Out of scope (this trial)

- Full Omni ↔ Quest STAGE calibration (tip locked to physical stylus on the desk)
- Hand / desk occlusion
- Changing eyepiece clutch or lap-panel behavior

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Plugin `.so` missing | Run `build_openxr_desk_mr.sh` |
| **AMBF window looks tiny / zoomed out** | CT is ~5 cm at 1:1; Desk MR uses **6× scale** + closer camera. Rebuild plugins. Delete `~/.config/fivrs/desk_mr_volume_pose.yaml` if an old pose loads. |
| **Quest blank** | Console must log `OpenXR desk_mr session READY`, `ALPHA_BLEND` or `OPAQUE`, then `snapped volume`. If blank with passthrough, retry with `AMBF_OPENXR_PASSTHROUGH=0` (opaque VR debug). |
| Still no passthrough | Log should say `ALPHA_BLEND`; WiVRn + Quest version; wired preferred |
| Eyepiece broken | Confirm you did **not** edit `ambf_openxr_eyepiece`; rebuild eyepiece `.so` if needed |
| Volume in wrong place | Ctrl+Shift nudge + save; delete `~/.config/fivrs/desk_mr_volume_pose.yaml` to reset |

## Files

| Piece | Path |
|-------|------|
| Plugin | `~/ros_ambf_ws/src/ambf_openxr_desk_mr/` |
| ADF | `ADF/openxr_desk_mr_camera.yaml` (launch **16**) |
| Eyepiece (untouched) | `~/ros_ambf_ws/src/ambf_openxr_eyepiece/` |
