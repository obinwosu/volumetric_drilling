# ambf_openxr_eyepiece

Head-locked stereo eyepiece for AMBF via **OpenXR** (intended runtime: **WiVRn** → Meta Quest).

- Renders `main_camera` (+ ADF dolly) into stereo swapchains
- Submits an OpenXR **VIEW-space projection layer** (eyepiece / ocular lock)
- Applies HMD 6DoF to `main_camera` only while cam-clutch is held

See `../volumetric_drilling/docs/QUEST_OPENXR_EYEPIECE.md`.
