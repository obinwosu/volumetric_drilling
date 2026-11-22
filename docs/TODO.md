# FIVRS Master To-Do — Lab Day 1 Feedback

Feedback from lab day 1 (Nov 13, 2026), plus fixes to make before day 2. Branch: `homeFIVRS`.

## Before day 2 (tomorrow)

### Tutorial simplification
- [ ] Remove the microscope target step. Replace it with: clutch the microscope and look left, right, up and down.
- [ ] Remove the device clutch step. Teach recentering only (double-tap the dark grey button).
- [ ] Show the current tutorial card on the desktop simulator view (not just the GUI) so the proctor can see which step the participant is on, in both the resident study and the graphics study.
- [x] Proctor advances steps with the right arrow key — already works, no change needed.
- [ ] Make a PDF slideshow with one full-screen slide per tutorial step: a screenshot of the simulator scene with that step's card on it.
- [ ] Afterwards: look for more tutorial steps to trim.

### Quest sleep
- [ ] About 30 seconds after someone takes the headset off, the Quest sleeps and the WiVRn stream drops. Lengthen or turn off the sleep timeout for the study, and write the setting down.

### Otic capsule shading (should be fairly easy)
- [ ] The labyrinth segmentation is the membranous labyrinth (the open canal). Build a 3 mm shell around it as a mask; no new scene segmentation needed.
- [ ] Shade the bone inside that mask more yellow than the base bone colour so the otic capsule (e.g. the lateral semicircular canal) stands out, and make the tint adjustable.

## Haptics
1. [ ] **Reliability (top priority):** haptics sometimes cut out completely, and the only fix right now is rebuilding and restarting the plugin. Build better debugging and haptic logs that show why it failed.
2. [ ] **Bone feels too soft (Pete):** review how drilling force feedback is calculated now. Add a tuner that opens from the study GUI: a small window to turn bone resistance up or down.
3. [ ] **Burr vibration:** a light hum in the handle to mimic a burr at 75,000 RPM.

## Backlog
- [ ] **Leftover voxel bits:** small pieces of bone that can't be drilled away. Look into how voxel blocks are made and removed.
- [ ] **Rotate the volume in the scene:** let users rotate the anatomy instead of only moving the camera. Requirement: ground truth stays correct. Keep track of the transform so drilling, voxel removal and the mesh drawing all use the current pose, and nothing is calculated against an old, untransformed copy.
