# Claude working notes (FIVRS volumetric drilling)

This is Obi Nwosu's fork of the FIVRS mastoidectomy simulator: AMBF voxel volume + Geomagic haptics, plus a Newton/OVRTX path-traced microscope that follows the same volume, drill, suction, and camera.

Read this before changing anything. The README is upstream install docs; this file is the live working state.

## Repo and remotes

Workspace: `/home/obihome/ros_ambf_ws/src/volumetric_drilling`

| Remote | URL |
|---|---|
| `fivrs` | `https://github.com/obinwosu/volumetric_drilling.git` (push here) |
| `origin` | `https://github.com/obinwosu/surreal-temporal-bone.git` |
| `upstream` | `https://github.com/LCSR-SICKKIDS/volumetric_drilling.git` |

**Stay on `feature/graphics-compare`**, tracking `fivrs/feature/graphics-compare`. The lab machine (RTX 5060) pushes there and is the reference: when it has new commits, fast-forward this checkout to it (2026-11-13: pulled 814bd15 with the graphics study, study chooser, pedals, pause menu, Quest fps work). `graphics-study` is merged into it and no longer used. The feedback_system submodule follows the commit the branch records (`git submodule update feedback_system`). After pulling, rebuild `scripts/build_openxr_eyepiece.sh` and `scripts/study_gui/rebuild.sh` (the latter exits 1 on `COLCON_TRACE: unbound variable` after a successful build; harmless).

Do **not** checkout `feature/fluid-simulation`. Do **not** apply or pop `stash@{0}` (film draft on that branch) or any other stash.

Other local branches (`feature/quest-microscope`, `feature/desk-passthrough`, `homeFIVRS`, `feature/soft-tissue-interaction`, `feature/tm-canal-reconstruction`, `cursor/ambf3-camera-nav-2b83`) are separate work. Do not mix them in.

Soft tissue work (`scripts/soft_tissue/`: soft ear demo, fast C++ solver, atlas/CT meshes) lives on **`feature/soft-tissue`** (worktree `../volumetric_drilling_soft_tissue`), moved off graphics-compare on 2026-11-11. The folder is git-ignored on graphics-compare; commit soft tissue changes from the worktree, never here.

## Git

- Commit and push only when asked.
- Never `git config`. When committing, set author/committer via env:
  ```
  GIT_AUTHOR_NAME='obi nwosu'
  GIT_AUTHOR_EMAIL='obinwosu@users.noreply.github.com'
  GIT_COMMITTER_NAME='obi nwosu'
  GIT_COMMITTER_EMAIL='obinwosu@users.noreply.github.com'
  ```
- Push with `git push fivrs HEAD`. No force-push.

## Hard constraints

- Do not regress suction, dust, drill pose, lighting, or voxel authority except as needed for the flake work below.
- Spacebar must **not** activate the drill.
- Suction is the rigid body `frazier_suction`. Button mappings stay unchanged, except the pause menu, which the user changed on 2026-11-08: tap Stylus 2 = down, tap Stylus 1 = up, left pedal = select, middle pedal = back / close (the drill then waits for the pedal to be released). Double-tap Stylus 1 still opens the menu. The study tutorial no longer asks for a burr change. 2026-11-16 (user): a tap of the middle pedal outside the pause menu steps to the next burr size (1 -> 2 -> 4 -> 6 -> 1 mm, on release; not with the microscope button held, which zooms out, and not for a press that began in the menu). Double-tap recenter puts the burr on the middle of the view (start-up placement stays right of centre); stylus taps ignore presses after a < 40 ms release (bounce) and for 0.45 s after a recenter, so one double-tap recenters once.
- World down is `(0, 0, -1)`.
- Do not change bone rendering or AMBF voxel removal except for the authorized flake fix.
- Irrigation is a **film on bone**, not SPH. Do not port Mantaflow, SPlisHSPlasH, DualSPHysics, FluidX3D, Taichi, tyFlow, or Flowmap Generator, and do not silently substitute SPH.
- Do not redo Touch hardware: no chmod 777, fake ttyACM300, firmware flash, `sg dialout`, Channel=300, or replacing `99-3dsystems.rules`. The Dec 2025 driver (`~/haptic_setup/driver_2025_libs`) may be tried on this home machine (not the lab machine), user decision 2026-11-15.
- Leave the user's `scripts/rtx_view/tuner_materials.json` coat change alone (blue glimmer on bone).
- Water transmission stays **on**. Turning it off looks like oil.
- Target 40–60 FPS. Irrigation + drilling is about 55–60 FPS with DLSS performance.

## How to run tests

Use `middle_ear_newton/.venv/bin/python`. The sandbox has no GPU; Warp's kernel cache cannot be written in the sandbox. GPU/Warp runs need unrestricted permissions. Writing `/tmp` often does too.

```
middle_ear_newton/.venv/bin/python scripts/fluid/test_impact_film.py
middle_ear_newton/.venv/bin/python scripts/fluid/test_irrigation_bench.py
middle_ear_newton/.venv/bin/python scripts/rtx_view/test_or_room.py
middle_ear_newton/.venv/bin/python scripts/rtx_view/test_warp_mesh.py
middle_ear_newton/.venv/bin/python scripts/rtx_view/test_label_surfaces.py
```

Do not `pkill -f` broadly to restart the sim. Use specific PIDs.

## Architecture (what is already shipped)

### RTX microscope

- Live window: `scripts/rtx_view/fivrs_rtx_view.py`
- Tuner: `scripts/rtx_view/fivrs_rtx_tuner.py`
- Bone meshed in **32³ dirty blocks** (`scripts/rtx_view/surface.py`, `warp_mesh.py`) with a 3-voxel halo so seams match.
- Deletions publish once per graphics frame. GPU Warp rebuild keeps a sustained cut from lagging.
- Camera, drill pose, and contact force come from the simulator.
- Microscope light (disk, fill, cone) is loaded from the tuner so live and tuner match.
- OR-room backdrop: `scripts/rtx_view/or_room.py`.
- DLSS is explicitly `omni:rtx:post:dlss:execMode = performance` in **both** apps. Frame generation is **off** in the live window. Do not go back to `"auto"`.
- The study GUI's Advanced tab has a **Rendering / Instruments** box (saved as `sim_options` in `scripts/study_gui/gui_setup.yaml`; presets in `gui_setup.PRESETS`: Fastest / Balanced = today's look / Most beautiful / Custom). Launch env read by `fivrs_rtx_view.py`: `FIVRS_DLSS` (`dlss_settings()`; `ultra_performance` = `execMode manual` + `manualScaling 1/3`, the per-axis fraction traced), `FIVRS_SPP`, `FIVRS_MAX_BOUNCES` (unset = tuner value), `FIVRS_BONE_DUST`, `FIVRS_IRRIGATION`, and `FIVRS_SUCTION_DEVICE` for the plugin. Study mode forces irrigation and suction off.

### Quest hands: removed (2026-11-12)

Hand tracking in the RTX view did not work and the user asked to remove it so it does not complicate the study. The code is gone from the plugin, RTX view and launcher; the first version is still in git history (commit `cfd1c57` on `graphics-study`). Do not bring it back unless asked.

### Irrigation film

Not particles. GPU film is `scripts/fluid/film_gpu.py` (Warp kernels on their own stream). CPU reference is `scripts/fluid/impact_film.py`. `IrrigationBench` draws it.

- Cannula on the drill; jet from the nozzle.
- Film pools downhill and runs off a lip. Dust is blocked on wet bone.
- Sheet triangles stay fixed over **4 mm blocks** around the water; most frames only move vertices so OVRTX refits instead of rebuilding BLAS.
- **No CUDA graph capture.** The bone mesher thread does `.numpy()` on the legacy stream; capturing a graph froze the RTX window (CUDA 906). Use recorded `Launch` objects.
- Look the user chose: transmission on, IOR **1.578**, film width **4.5 mm**, depth ~4 mm, `max_bounces` **4**. Width lives in `scripts/rtx_view/tuner_irrigation.json` (`film_spread_mm`).

Black streaks in water (dark parallel shards when looking into a drilled hole) were a film bug, **fixed** in `86918cf`:
- Drop quads whose corners differ by more than `CLIFF_MM = 1.0`.
- `_outline_on_shelf` does not round across a drop.
- `_snap_boneless` pulls vertices with no bone onto a neighbour.
- Topology rebuilds after a bone re-read **before** the next draw.
- Test: `test_gpu_sheet_does_not_bridge_drilled_walls` in `scripts/fluid/test_impact_film.py`.

Residual “shard” look *inside* the water is refraction of ripple normals + IOR 1.578, not glass path-splitting. Tuner knobs if asked: Ripple → 0, IOR toward 1.33, Smooth up. Not started.

### Study GUI / volumes / labels (`86918cf`)

- Delta volumes: `ADF/volume_ambf_delta_0p125mm.yaml`, `ADF/volume_ambf_delta_0p250mm.yaml`.
- Buried labels meshed as their own solids (`surface.py` occupancy-for-RGB, `scripts/rtx_view/test_label_surfaces.py`).
- Tuner launch dialog, NRRD→volume helper, study GUI volume picker.

### EAC is air (user decision, 2026-11-06)

The EAC label (`#ffe1d6`) is a solid canal plug, not anatomy. It is cleared at load: not drillable, no haptic contact, not rendered. AMBF: `clearAirLabels()` / `kAirLabelRgb` in `volumetric_drilling.cpp`, re-run when a reset swaps the texture. RTX: `scripts/rtx_view/air_labels.py`, used by `_load_occupancy` and the tuner. Only the cortex-mask lateral frame still sees the intact EAC. Keep both color lists equal (`test_air_labels.py` checks). Canal skin will come back as a deformable sheet, not voxels.

### Voxel removal (AMBF)

`plugin/volumetric_drilling/volumetric_drilling.cpp` (~1034–1060): only voxels in CHAI haptic **contact events** are cleared, capped by `voxelRemovalThreshold`. Specks the proxy never hits are never deleted. CHAI isosurface is 0.1 (alpha ≥ ~26/255). RTX occupancy cutoff is alpha > 10. On these volumes every solid voxel is 254/255, so load-time occupancy matches.

## Graphics study (built 2026-11-12, not yet run on the rig)

Launcher tab **Graphics study** (`scripts/study_gui/graphics_study_mode.py`, logic in `scripts/study/graphics_study.py`). Resident-study tutorial in arm 1's graphics, then the same task twice in balanced random order: cortical mastoidectomy + identify the incus short process (proctor presses **Incus identified**; 10 min drilling cap), raw NASA-TLX on the desktop after each arm. After both, two preference questions (more realistic / rather train on: Session 1, Session 2, No difference) go to `preferences.csv` with the graphics each answer points to. Anatomy A, headset, device 1, no irrigation. The simulator restarts between arms. Proctor picks the comparison: `upstream_vs_rtx` or `chai_vs_rtx`.

- CHAI3D arms launch AMBF with `--rtx_view 1` (keeps drill_state, restore, drill lock) but no RTX window. `AMBF_OPENXR_CHAI_EYES=1` makes the eyepiece plugin render CHAI3D as the RTX eyes are rendered (1152x864 each, 23 mm base x 0.01007, toe-in to the volume, main_camera FOV) and show it in the same eyepiece. AMBF's own front-layer panels are not drawn.
- Study text for those arms: `scripts/rtx_view/hud_overlay.py` (started by the launcher) draws `headset_hud` layers into `/dev/shm/fivrs_hud_overlay`; the plugin blends it over both eyes.
- Upstream look = `FIVRS_LOOK=upstream` (plugin: no gray room backdrop, light1 cutoff 0.5 rad and upstream offset) + `ADF/volume_256_RT143_upstream_look.yaml` (original `shaders/volume`). Geometry, camera, haptics unchanged. Upstream had no checkered floor (it was commented out).
- Files: `<recording base>/graphics_study/` (`graphics_study_dev/` for desktop runs): `participant_key.csv`, `assignments.csv`, `tlx.csv`, `trials.csv`, `recordings/G###_<stamp>/arm<n>_<graphics>/`. IDs are `G###`.
- Tests: `scripts/study/test_graphics_study.py`, `scripts/rtx_view/test_hud_overlay.py`, `QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_graphics_study_mode.py` (system python3, it has PyQt5; the newton venv does not).

## Tutorial (changed 2026-11-13, lab day 1 feedback)

- Six steps after Welcome: moving the drill, recentering, moving the microscope (look left / right / up / down, ticked off on the card; no target), zoom, drill, pause menu. The device clutch (reposition) step is gone.
- The card is also on the proctor's desktop view: RTX headset sessions draw the headset text onto the RTX desktop window on the GPU (`headset_hud.mirror_to_desktop`, ~0.05 ms; `FIVRS_DESKTOP_CARD=0` off); CHAI3D sessions show `/tmp/fivrs-rtx/study_card.txt` (written by `SessionIO.write_hud`) as a label in AMBF's window, upper left, only while `study_hud.json` is fresh.
- Slides: `scripts/study/make_tutorial_slides.py` -> `docs/tutorial_slides/` (one PNG per step + `tutorial_steps.pdf`), background `scene.png` = tuner start-up view of Anatomy A.

## Saved views (2026-11-14)

- One file: `ADF/world/canonical_views.yaml`. The tuner (`camera_views.save_views`), the pause menu (`saveCurrentIntoView`) and the launcher's **Saved Views** list (was "Canonical view") all use it. "Default" = the view whose id is the anatomy's laterality (`left` / `right`); the list shows its label and re-reads the file each time it is opened.
- Bug fixed: the RTX desktop card's last command in `/tmp/fivrs-rtx/pause_menu.txt` (from 2026-10-31: `1 apply right_transtemporal`) replayed on every launch and replaced the startup view with "Right mastoid". The plugin now deletes that file at startup.
- Anatomies can be renamed in the launcher (button under the list, or right-click): `add_volume_tab.rename_volume` changes only that entry's `name:` line in `gui_setup.yaml`.

## Add a volume: CT -> segmentation -> anatomy (built 2026-11-13)

Launcher tab **Add a volume** (`scripts/study_gui/add_volume_tab.py`) runs `scripts/segmentation/tbone_worker.py` in the venv of `../tbone-seg-nnunetv2` (andy-s-ding/tbone-seg-nnunetv2; config key `tbone_seg_dir` overrides the path).

- Install on a new machine (the lab): `scripts/segmentation/setup_tbone_seg.sh` clones the repo at the tested commit (ac0290d) with its nnUNet submodule, builds the venv as below, downloads + checks the weights, and checks torch on the GPU. Re-running skips finished steps (~5 s).
- "Then start the simulator on it" (tickbox next to Segment): after the volume is added, the launcher (`start_simulation_on_selected_volume`) switches to Main and presses Start simulation with Main's settings; not if the simulator is already running or a study is in progress.
- The worker keeps the logits in host memory when the GPU has < ~14 GiB free (5060, or the simulator running): slower, fits.
- Venv: `python3 -m venv --without-pip .venv` (no python3-venv on this box), pip borrowed from `middle_ear_newton/.venv` (`-m pip --python .venv/bin/python install pip`), then torch cu128 (the 5090 needs it), `-e ./nnUNet` (the repo's fork, 2.5.1), SimpleITK, pydicom, pillow, pyyaml, edt. Weights: the 1.1 GB LFS zip from media.githubusercontent.com unzipped into `nnUNet_results/` (sha256 b669bb17...).
- PyTorch >= 2.6 loads with weights_only; the worker allow-lists the checkpoint's numpy scalar types (`allow_checkpoint_numpy`), not weights_only=False.
- Training scans were ~50 mm cubes at 0.098 mm, so the worker crops a box (default 50 mm) around the ear and B-spline resamples it to 0.098 mm before inference. 1 fold + mirroring: ~70 s, 12 GB GPU. User: more than one fold runs out of memory; default and recommendation is 1 fold.
- Laterality comes from the series name ("Left Dehiscence 0.4"), else from a single-ear scan's position (LPS +x = left); only asked if neither says. Arrays are LPS index order like the JHU anatomies; left ears get LT143's ADF pose.
- Colors are tbone_colortable as Anatomy A-I store them (truncated: malleus e900ff, chorda 97831d). **Sigmoid sinus and dura are split since 2026-11-15** (user): sigmoid `2864a0` (darker), dura `aad7e8` (lighter) instead of the merged `6eb8d1`; named `sigmoid_sinus` / `dura` in `ADF/color/volume_labels.yaml`; `tuner_materials.json` got two new segments copied from `sinus_dura` (diffuse x0.45 / 45% toward white; nothing else in the file changed); `thin_bone.py` counts both as dura. Existing volumes: `scripts/segmentation/split_sinus_dura.py <volume folder> <run folder>` recolours only merged voxels from the run's `segmentation.nii.gz` (done for Left Dehiscence and + VS: 0 voxels left merged, 0 other changes). Cannot split: Anatomy A-U (JHU source labels "sigmoid sinus + dura" as one), AMBF Delta (one Sinus_Dura segment), Anatomy 512 / ear3 (no source here). Icons not regenerated. The feedback_system EDT tool still keys on the merged colour.
- Each run appends to `scripts/segmentation/crop_log.jsonl` (auto vs chosen box, landmark offsets, bone cut at each face, hashed series UID, no patient data): data for an automatic crop later. Not built yet.
- Work files (incl. the CT as NIfTI) stay in git-ignored `resources/intermediate_volumes/segmentation/`. Generated volumes go to `resources/volumes/<name>_0p196mm/` (git-ignored: commit with `git add -f`), `ADF/volume_<name>.yaml`, `scripts/study_gui/assets/<name>.png`, and a new `volumeN` in `gui_setup.yaml`.
- First volume: **Left Dehiscence CT** (volume33) from `~/Downloads/pcseg.zip` (unzipped to `~/Downloads/pcseg_dicom`). Orientation matches Anatomy D; not yet drilled in the simulator.
- Test: `QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_add_volume_tab.py`.

## Register MRI -> CT (built 2026-11-15, committed dd1366f)

Launcher tab **Register MRI** (`scripts/study_gui/register_mri_tab.py`) runs `scripts/registration/mri_ct_worker.py` in the tbone-seg venv (SimpleITK + torch; same `@progress/@result/@error` protocol as `tbone_worker.py`). Rigid only. Results: `resources/intermediate_volumes/registration/<stamp>/` (`ct.nii.gz`, `mri.nii.gz`, `mri_to_ct.tfm` = ITK CT point -> MRI point, `mri_on_ct.nii.gz`, previews, `registration.json`).

- Method (`scripts/registration/rigid_register.py`): normalized gradient fields at the CT's strongest edges, **only in 44 mm boxes around the temporal bones** (`guess_ear_centers`), summed over all samples (outside the MRI earns 0). GPU brute force at 3 mm (pitch +-60 / roll, yaw +-10, every 3 mm: 24 M poses, ~7 s on the 5090), then Powell at 3/2/1/0.5 mm on 24 starts; best by the 1 mm score. ~30 s total. Uses the GPU whenever >= 0.75 GB is free (batch size follows free memory; 0.44 GB peak when capped at 0.8 GB, same pose, 27 s), so it can run beside the simulator on the 5060. CPU fallback: coarser grid, 8 starts (74 s on the first pair, same pose).
- What failed on the first pair (bone-kernel CT + C+ FIESTA, slabs tilted **41 deg** to each other, scanner coords 240 mm apart): Mattes MI (padded or not), normalized MI, MI with an outside bin, and NGF on the whole head all locked onto wrong poses (MI: CT soft tissue is noise, FIESTA bone/air/brain all dark; whole-head outline fits a wrong 30 deg tilt). Validated by eye: MRI labyrinth fluid inside the CT otic capsule on both sides.
- Robustness checks (same answer within 0.35 mm): MRI blurred to 0.9x0.9x3 mm with T1-like contrast; CT cut to one ear. Confidence shown in the tab: >= 2 starts agreeing and next different pose <= 80% of the score.
- 3D Slicer 5.12.4 installed 2026-11-15 at `~/Slicer-5.12.4-linux-amd64` (tarball, sha512 checked, in `~/Downloads/slicer_download`); checked: the loaded transform maps the MRI onto the CT (0.0000 mm). Not on the lab machine yet. `find_slicer` looks at `slicer_path` in `gui_setup.yaml`, PATH, `~/Slicer*/Slicer`, `/opt/Slicer*`. **Open in 3D Slicer** runs `scripts/registration/slicer_show.py` (loads CT, MRI under the transform, bone window, 4-up). Hand edits saved over `mri_to_ct.tfm` + **Reload after Slicer** (`mri_ct_worker.py preview <work>`) redo the views.
- Test scans (patient data, not in the repo): `~/Downloads/pcct_dicom` (bilateral CT) + `~/Downloads/pcmri1_dicom` (C+ FIESTA). Result: pitch -41.3, roll +1.0, yaw +0.4 deg.
- Tests: `../tbone-seg-nnunetv2/.venv/bin/python scripts/registration/test_rigid_register.py` (synthetic tilted-slab phantom, GPU ~25 s), `QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_register_mri_tab.py`.
- VS tumor (2026-11-15): Kujawa et al. 2024 nnU-Net v2 models (Zenodo 10363647, CC BY 4.0; Multi-Centre Routine Clinical set, 8.7 GB, unpacked in `../vs-seg-kujawa`), run by `scripts/segmentation/vs_seg_run.py` in the tbone-seg venv (`--model t1`: Dataset910 3d_fullres+2d ensemble; `--model t2`: Dataset911 on FIESTA/CISS). Same-session series share the FIESTA's FrameOfReferenceUID, so `mri_to_ct.tfm` carries them straight onto the CT (checked: no motion between C+ T1 and FIESTA). First patient: left VS with an IAC part, C+ T1 5.77 mL, FIESTA 5.41 mL, Dice 0.915 on the CT; outputs + figures in `resources/intermediate_volumes/vs_seg/`. Not in the launcher yet.
- The models missed the **middle fossa component** (probability ~0 there; atypical for their training data). Cropping the input made it worse (70 mm crop: 0.07 mL, lost the CPA part too: z-score normalization/context). Fix used: **nnInteractive** (DKFZ 2025, weights CC BY-NC-SA 4.0, `~/.nninteractive/models/nnInteractive_v1.0`) in its own venv `../nninteractive/.venv`, which borrows tbone's torch via `zz_tbone_torch.pth` (+ torchvision 0.26.0+cu128 from the PyTorch index; the PyPI one breaks). One positive click on the C+ T1 resampled to 0.5 mm isotropic: 4.20 mL, clean (on FIESTA only 0.28 mL: that part has little T2 contrast). Combined label on the CT: `tumor_full_on_ct.nii.gz` (1 = CPA/IAC from the nnU-Net consensus, 5.48 mL; 2 = middle fossa, 4.17 mL).
- **Tumor box in the Register MRI tab** (built 2026-11-15): `scripts/study_gui/tumor_panel.py` drives `scripts/registration/tumor_worker.py`, a persistent worker in `../nninteractive/.venv` (JSON lines on stdin; models load once). Tumor series = any series sharing the registered MRI's frame of reference (`mri_frame_uid` in `registration.json`; a contrast T1 is listed first). "Find tumor automatically" = Kujawa model -> part 1; Click mode: click a view = add to the current part, Shift+click = remove (nnInteractive; thick-slice series resampled to 0.5 mm iso); New part / Undo (own 20-step history, nnInteractive's undo is one step) / Delete / Save on CT. Outputs `<registration>/tumor/` (`tumor_on_ct.nii.gz` labels 1..n), loaded by `slicer_show.py` as segments. First patient in the tab: auto 5.75 mL (38 s), one click on the middle fossa part 4.40 mL (7 s incl. model load). nnInteractive on the CPA part alone: one click Dice 0.943 vs nnU-Net but only 58% of the IAC part; a second click in the canal -> 97%, Dice 0.946.
- Install on the lab machine: `scripts/segmentation/setup_tumor_seg.sh` (after `setup_tbone_seg.sh`; venv borrows tbone's torch via `.pth`, torchvision from the PyTorch cu index, 8.7 GB Zenodo download md5-checked, nnInteractive weights). Reruns in ~4 s. gui_setup.yaml key `nninteractive_python` overrides the venv path.
- Tests: `../nninteractive/.venv/bin/python scripts/registration/test_tumor_worker.py` (synthetic ball, GPU, ~10 s), plus the tab test above.
- **Tumor in the simulator** (2026-11-15): anatomy **"Left Dehiscence CT + VS"** (volume34, `ADF/volume_left_dehiscence_ct_vs_0p196mm.yaml`, 319x316x258 at 0.196 mm), made by `scripts/segmentation/add_tumor_to_volume.py` (tbone venv). The original 50 mm box held only half the tumor, so the box grows to ear box + tumor (62x62x51 mm). Inside the old box: the existing Left Dehiscence labels voxel for voxel (`--ear-box-from`; checked 96.8% identical, all differences are tumor except 8 voxels); outside: the tbone model on the bigger box (`--outside-from` reuses that run, `..._tumor_full_model`). Tumor label `be6eaa` = `vestibular_schwannoma` (added to `ADF/color/volume_labels.yaml`), 9.75 mL, smoothed sigma 1 mm (2.5 mm T1 slices terraced); it replaces empty, IAC, vestibular/cochlear nerves and the model's ~3 mm dura band, not bone/labyrinth/facial nerve/sinus/carotid/ossicles. `--keep-pose-of` moves the ADF position (8.7 mm) so the ear lies where the original anatomy has it (saved views still fit). Memory: the model's 18-class float32 output is ~220 bytes/voxel on the host; 208 M voxels OOM-killed the app's process (24.7 GB of 30), so the script segments in pieces of <= 80 M voxels with 8 mm overlap; run it under `systemd-run --user --scope -p MemoryMax=20G`. The tumor is ordinary voxels: drillable and touchable like bone in AMBF; in RTX it gets the fallback material (no tuned look yet).
- Not built: using the registered MRI in the simulator beyond the tumor (e.g. nerves / soft tissue labels).

## Lab machine: after pulling the 2026-11-15 push

1. `git pull` on `feature/graphics-compare`, then `git submodule update feedback_system`.
2. Rebuild: `scripts/study_gui/rebuild.sh` (plugin: haptic monitor feed + RTX voxel sync fix; exits 1 on `COLCON_TRACE: unbound variable` after a good build), `scripts/build_openxr_eyepiece.sh`, `scripts/haptic_monitor/build_probe.sh`.
3. Haptics (merged from `haptics-updates`): `haptics/install_hdphantom_warmup.sh` patches AMBF's libhdPhantom (open/close warm-up + motor self-check / re-open) and installs `~/.local/lib/libhdPhantom.so`, which the launcher loads first. Made for the HID Touch (USB 2988:0304); real launches 0/40 dead with it. The launcher shows the verdict ("Force feedback (Touch motors)") from `/tmp/fivrs-rtx/haptic_selfcheck.json`. Notes: `haptics/README.md`.
4. Tumor tools (needs internet, ~10 GB disk, ~30 min for the Zenodo download): `scripts/segmentation/setup_tbone_seg.sh` if not done yet, then `scripts/segmentation/setup_tumor_seg.sh`. Weights are not in git (5.4 GB + 393 MB).
5. 3D Slicer (optional, for Open in 3D Slicer): unpack Slicer 5.12.4 Linux from download.slicer.org into `~` (or set `slicer_path` in gui_setup.yaml).
6. Nothing to build for the anatomies: Left Dehiscence CT (sinus/dura split) and Left Dehiscence CT + VS are in git. Making a new tumor anatomy with `add_tumor_to_volume.py` needs ~20 GB of host RAM even in pieces; registration and the tumor box fit the 5060.

## Haptic monitor (built 2026-11-14, committed 2026-11-15)

Separate window, toggled from the launcher's Advanced tab (**Haptic device** box, saved as `haptic_monitor:` in `gui_setup.yaml`; `scripts/study_gui/haptic_monitor_toggle.py`). Window: `scripts/haptic_monitor/haptic_monitor.py` (also runs on its own).

- Sim running: AMBF owns the Touch. `plugin/volumetric_drilling/haptic_monitor_feed.h` (physics thread only stores atomics) reads both Touches ~20x/s via one `hdScheduleSynchronous` callback (`haptic_device_probe.h`: no frame, no force set; handles from libhdPhantom's exported `phantomDevices`) and writes `/tmp/fivrs-rtx/haptic_state.json` (20 Hz while the window's `haptic_monitor.alive` is fresh, else 2 Hz).
- Sim closed: the window runs `scripts/haptic_monitor/bin/haptic_probe` (built by `build_probe.sh`, git-ignored; started with `scripts/setup_haptic_env.sh`, without GTDD_HOME the drill Touch gives HD error 0x302) with force output **off**, writing `haptic_direct.json`. It lets go on `haptic_probe.release` (launcher calls `release_device_for_sim()` right before starting AMBF, then waits 1 s), when an `ambf_simulator` appears, or when the window closes.
- Health log, always while a Touch is read: `~/.local/share/fivrs/haptic_health.log` (servo < 900 Hz, force output off, HD errors drained in the servo thread, calibration, motor heat > 80%, physics stalls/stops, safety trips, "force dropped", 60 s summary).
- Readable: encoders, joint/gimbal angles, buttons, inkwell, motor temperature (3 motors, 0-1 heat model), calibration, servo rate, force output, max force. Not readable: firmware version.
- Test: `QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_haptic_monitor.py` (also compiles and runs `scripts/haptic_monitor/test_health_tracker.cpp`).

## Known issue: haptics drop out (global, not yet investigated)

The haptic monitor's health log (above) is the place to start: CHAI3D's libhdPhantom never calls `hdGetError`, so OpenHaptics errors such as max velocity / force impulse exceeded (which can switch force output off) were invisible until now.


Reported 2026-11-07: haptic feedback from the Touch is frequently dropped, seemingly at random, during normal use (not only in study mode). Cause unknown. It affects the whole simulator, so study recordings may contain stretches without force feedback. Do not "fix" it by redoing Touch hardware setup (see Hard constraints). When investigated, start by logging the haptic loop rate and device status in `plugin/volumetric_drilling/volumetric_drilling.cpp` around the time of a drop.

## Known issue: headset link sometimes stalls at launch (2026-11-08)

**Actual cause found later that night:** the launcher minimized AMBF's window every 0.5 s from launch. Minimizing it *before its first frame* stops AMBF's graphics loop entirely (no swap on an unmapped window) until the window is restored: no headset session, no RTX camera, no published deletions. Reproduced with `xdotool windowminimize` right after launch. Minimizing an already-drawing window is harmless. Fix (user's choice): the launcher no longer minimizes AMBF at all (`_ambf_minimize_timer` is never started). The notes below were written before this was found; the WiVRn background-connect change is still in place.

Headset-mode RTX cost (benchmarked 2026-11-08, idle, no irrigation): two eyes ~25 ms a frame, one eye ~14.5 ms. DLSS ultra performance, 2 bounces, or 960x720 eyes each save only 1-2 ms (24-26 ms); the second view is a fixed ~10 ms in ovrtx, with the GPU ~70-75% busy. `EyeShare.write` now copies straight into the shared file (5.5 -> 0.4 ms). `summary.json` has `headset_eyes_ms` (window, readback ~3.5-5 ms, text, write).

**Mitigated 2026-11-08:** the OpenXR eyepiece plugin (source in this repo at `plugin/ambf_openxr_eyepiece/` since 2026-11-10, built by `scripts/build_openxr_eyepiece.sh`; pre-change copy in `../ambf_openxr_eyepiece_backup_2026-11-08-*`) now runs `xrCreateInstance` / `xrGetSystem` / the GL requirements query on a background thread (`startConnect` / `pollConnect`) and retries every 3 s if WiVRn refuses. AMBF keeps drawing (RTX camera, deletions, timers) while it waits; the session is still created on the graphics thread. The headset shows only the RTX eyes (`AMBF_OPENXR_RTX=only`). Tested with a stand-in runtime that stalls 20 s, and a real launch.

Before that fix, the plugin called `xrCreateInstance` on AMBF's first graphics frame, and AMBF's whole graphics loop waited until WiVRn let it in. Usually 2-6 s; once 33 s on its own; once stuck >2 min (black headset) until WiVRn was reopened on the Quest, after which AMBF connected 14 s later. In that run the RTX window's frame loop also stopped silently at 21:38:16 (cause unknown). Not caused by minimizing AMBF (tested) or the NVRM `refcntRequestReference` kernel message (appears in good runs too).

The Study tab now flags it: `study_checks.launch_status` watches AMBF's `status.json` heartbeat, the "OpenXR eyepiece session" lines in `ambf.log`, and the RTX `summary.json` heartbeat. It says "slow" at 15 s and "quit WiVRn and open it again" at 45 s. Each problem goes to `/tmp/fivrs-rtx/launch_problems.log` with the AMBF and RTX threads' kernel wait channels. The RTX window dumps all thread stacks into `view.log` if its loop stalls 8 s (`LoopWatchdog`), and logs a closed window and the exit code. Next time, read those first. Logs of the failed run are not kept in the repo.

## Open work: floating bone flakes

Tiny folded / paper-thin bits of bone that appear at volume load, catch water, go dark, cannot be drilled away, and would not exist in real bone.

Video: `/home/obihome/Videos/Screencasts/Screencast from 2026-11-01 01-37-50.webm`

### Already known

- **Not a 32³ block-meshing seam bug.** Cuts near a face rebuild both neighbours; shared-face pins match.
- **Not an alpha mismatch.** AMBF and RTX agree at load.
- **They are in the volume data.** Island counts at load (26-connected, ≤50 voxels unless noted):
  - `volume_256.yaml`: 24 islands
  - `volume_256_RT138.yaml`: 347 islands
  - `volume_256_LT143.yaml`: 55 islands
  - plus thousands of one-voxel-thin flaps
  - Live volume `ADF/volume_ambf_delta_0p125mm.yaml`: **41 islands ≤200 voxels** at load
- Drill contact never reports these specks, so AMBF never deletes them.

### Cause of the mid-air pieces after drilling: found and fixed (2026-11-15, not yet verified live)

The pieces the user sees hanging in mid-air during drilling were a **sync bug**, not volume data: AMBF had deleted them, the RTX view never heard. Live snapshot on `volume_256_RT143.yaml` mid-drill: AMBF had 0 floating pieces; RTX held 196 bone voxels AMBF had deleted (49 clusters, all at the cut edge, 36 floating). `rtx_timing.jsonl` vs `ambf_timing.jsonl`: 12 batches arrived with `n: 0` though AMBF sent 7-30 voxels (344 voxels lost that session). Because the serial arrived, the RTX gap check never asked for a catch-up.

Race: in RTX mode the graphics thread (`serviceRtxFlush`) fills and publishes the shared ROS voxel message, while the physics thread's drill loop still called `clearVoxelMsg()` on it every drilling step. Fix: that clear only runs when `!m_rtxView`. Verify with a fresh session: drill, then compare the two timing logs (no `n: 0` receipts for nonzero batches) and run the snapshot below (0 RTX-only voxels).

The old probe timed out because it ran in the sandbox (private `/tmp`). Run snapshots unsandboxed. One-shot check used: request `rtx_occupied.request` + `resync.request`, compare `rtx_occupied.npy` with `occupancy.bin` (`_load_dump`). Note RTX occupancy leaves out closed labels (ossicles, chorda, etc.), so "AMBF-only" voxels of those colours are expected.

Island tracker (`scripts/rtx_view/islands.py`, 26-connected, all-bone, ≤2000 voxels) does remove load-time islands (23 on RT143). Its many "cleared 0" for cut-loose pieces were these lost voxels. Remaining smaller question: pieces attached to bone only at a voxel corner/edge (39 on the 0.125 mm volume at load) render as floating; counting face contact only would catch them.

### Authorized fix (do this)

User wants A+C after finishing D:

1. **Finish D** if possible: why the drill cannot remove them. Dump live AMBF occupancy vs RTX after a cut on `volume_ambf_delta_0p125mm.yaml`.
2. **Cut-loose islands disappear** as soon as they detach from the main bone (visual; keep haptics in sync if they exist in AMBF).
3. **Islands and one-voxel-thin flaps already in the original volume, or produced by blocking, melt into neighbouring bone** — absorb into a neighbour, do not delete into air. AMBF and RTX occupancy must stay in agreement.

This is an explicit exception to “do not change bone rendering / voxel authority.” Do not also change irrigation, DLSS, suction, or lighting.

## Key files

```
scripts/rtx_view/fivrs_rtx_view.py       live RTX window
scripts/rtx_view/fivrs_rtx_tuner.py      material / light / irrigation tuner
scripts/rtx_view/surface.py              block volume, label occupancy
scripts/rtx_view/warp_mesh.py            GPU marching cubes
plugin/volumetric_drilling/volumetric_drilling.cpp   voxel removal
scripts/fluid/film_gpu.py                GPU irrigation film
scripts/fluid/impact_film.py             CPU reference film
scripts/rtx_view/tuner_irrigation.json   film width, IOR-related irrigation
ADF/volume_ambf_delta_0p125mm.yaml       volume used in the flake session
```

## Suggested first move

Determine whether post-drill flakes exist in AMBF, then implement melt-into-neighbour at load and drop islands cut loose from the main component, keeping AMBF and RTX occupancy in sync.
