# AMBF3 camera_navigation Port — Conflict Decision Log

Branch: `cursor/ambf3-camera-nav-2b83`  
Base: `LCSR-SICKKIDS/volumetric_drilling` `master` (AMBF 3 / ROS1+ROS2 colcon)  
Feature source: `obinwosu/volumetric_drilling_VRE` `camera_navigation`

| Area | VRE (`camera_navigation`) | Upstream (`master`) | Chosen | Rationale |
| --- | --- | --- | --- | --- |
| Build system | Root `CMakeLists.txt` AMBF2 + catkin-only, C++17, X11/udev/cnpy | `plugin/CMakeLists.txt` + `package.xml` dual catkin/ament | **Upstream AMBF3**, extended | Keep colcon/ROS2 Jazzy layout; add VRE sources (video recording, gaze jagged, X11, vendored cnpy) without reintroducing AMBF2 root build |
| Plugin load path | `path: build/` (AMBF2 local build) | `path: build/volumetric_drilling` | **Upstream pattern adapted** → `../../install/volumetric_plugin/lib` | Colcon installs `libvolumetric_drilling.so` under `install/volumetric_plugin/lib` for `~/ros_ambf_ws` |
| Camera navigation / pause menu / canonical & transtemporal views / dolly/orbit/roll | Full feature set in ~2300-line plugin | Minimal ~900-line plugin | **camera_navigation** | Core VRE usability |
| DVR shaders + microscope beam lighting uniforms + fluted burr | `ADF/shaders/dvr`, world light parented to camera, fluted burr mesh | Stock volume shaders / lighting | **camera_navigation** | Required for nav+lighting fidelity |
| Volume placement / `conversion factor` / operative camera framing | `conversion factor: 0.049664`, camera/volume poses tuned for DVR | `conversion factor: 1.0`, different framing | **camera_navigation** | Features depend on this scale/framing |
| Study GUI UX (profiles, video checkbox, volume list, launch helpers) | Expanded GUI + profiles + assets | Smaller upstream study GUI | **camera_navigation**, paths relativized | Keep UX; strip `/home/amunawa2/...` |
| Hardcoded absolute paths (`launch.yaml`, `gui_setup.yaml`, desktop, plugins) | Machine-specific `/home/amunawa2/...` | Repo-relative / `ambf_simulator` on PATH | **Replace with repo-relative / configurable** | Portable A51 + any workstation |
| Footpedal / joystick / drill reposition / `DrillReposition.msg` | ROS1-only publisher + msg | Dual ROS1/ROS2 `ros_interface`, no reposition | **camera_navigation features wired through upstream ROS2 `ros_interface`** | Keep AMBF_ROS1/ROS2 guards; add `DrillReposition` pubs for both |
| Video recording | In-plugin `VideoRecordingController` + optional external `.so` | Not present | **camera_navigation** in-plugin; external plugin optional via `gui_setup.yaml` | External path no longer hard-coded |
| Gaze marker variants | `gaze_marker_controller_{jagged,smooth}` | Single `gaze_marker_controller` | **Jagged (VRE default)**; smooth kept in tree | Matches VRE study default; switchable in CMake/includes later |
| Extra volumes (NASBS, EMMJ, DG, …) + ADF/assets | Many ADF + GUI icons | Core temporal-bone set | **camera_navigation** ADF/GUI list | Volume *image stacks* for extras may need separate data/LFS sync (see Notes) |
| Pupil calibration | Present in GUI / scripts | Present (simpler) | **Bring files; do not block** | Optional; deferred if `pupil_capture` missing |
| Launch scripts (`drilling_simulator.sh`, `launch_study_gui.sh`) | ROS1 Noetic + `roscore` + Anaconda | Catkin `devel/setup.bash` + `roscore` | **Upstream/AMBF3-correct ROS2 Jazzy** (source `~/ros_ambf_ws/install/setup.bash`, no `roscore`) | Usability conflict → AMBF3-correct |
| Spacenav plugin arg | Hardcoded absolute `.so` always passed | N/A | **Optional** `spacenav_plugin` in `gui_setup.yaml` | Avoid broken launch when plugin absent |
| Duplicate equivalent implementations | — | — | **Keep one** (upstream when equivalent) | e.g. wave_generator / common.h left as shared |

## Implementation notes (A51)

- NVIDIA driver OK on A51 (`nvidia-smi` shows RTX 5090) before first successful sim launch.
- `cnpy` was empty in the GitHub zip of VRE; vendored from [rogersce/cnpy](https://github.com/rogersce/cnpy) under `plugin/volumetric_drilling/cnpy/`.
- Extra volume PNG stacks referenced by NASBS/EMMJ/LT*/RT* ADFs are **not** all present in the shallow GitHub zip (only core `ear3_*` / RT143/RT138/RT147/LT143/LT151 stacks). Smoke-test with `volume_256_RT143` / Anatomy A until full volume data is synced.
- After first successful study-GUI launch, append any build/runtime surprises below.

## Runtime surprises

Filled after first successful A51 sim launch (2026-10-03).

| Surprise | Resolution |
| --- | --- |
| `setup_ambf3_ubuntu24.sh` failed sourcing `/opt/ros/jazzy/setup.bash` under `set -u` (`AMENT_TRACE_SETUP_FILES` unbound) | Source ROS with `set +u` around setup.bash |
| Zip extracts (`volumetric_drilling-master`, `*_VRE-camera_navigation`) under `src/` caused duplicate package names for colcon/rosdep | Moved refs to `~/ambf3_port_refs/`; setup script now deletes sibling zip extracts under `src/` |
| Footpedal `/dev/input/js0` absent on A51 (keyboard/mouse only) | Plugin falls back to default state; GUI still configurable when pedal present |
| `timeout` kill of `ambf_simulator` reported `double free or corruption` on shutdown | Observed only on forced SIGTERM teardown; sim ran and loaded plugin/shaders before exit |
| Extra volume PNG stacks (NASBS/EMMJ/many LT*/RT*) missing from GitHub zip | Smoke-tested with `ADF/volume_256_RT143.yaml`; sync volume data separately for full list |

### A51 smoke evidence

- `nvidia-smi`: RTX 5090 OK
- `colcon build`: 8 packages finished including `AMBF`, `volumetric_drilling_msgs`, `volumetric_plugin`
- `ambf_simulator` loaded `libvolumetric_drilling.so` from `install/volumetric_plugin/lib`
- DVR shaders compiled/linked for `mastoidectomy_volume_low`
- Fluted burr + 6 canonical views loaded; applied “Right transtemporal”
- Pause-menu / resume log line present (`Ctrl+P`)
