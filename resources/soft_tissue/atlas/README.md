# Atlas meshes (right side), copied for the soft-tissue scene

Snapshot (2026-11-11) of `resources/atlas_meshes/z_anatomy/` from the "3D mesh generation from images"
session's worktree (`claude/3d-mesh-generation-images-e9c3a2`), right side only. See that folder's
README for how they were exported and built.

- **Source:** Z-Anatomy (github.com/Z-Anatomy/The-blend), built on BodyParts3D (DBCLS), from Visible Human data.
- **License:** CC BY-SA 4.0 for these mesh files and meshes derived from them (the registered copies in
  `resources/soft_tissue/composite/*/atlas_*.ply`). Credit Z-Anatomy and BodyParts3D in any paper or release.
- `posterior_auricular_muscle_r.obj` is not from the atlas: it was built from published measurements on
  atlas landmarks (`posterior_auricular_muscle.json`; thickness 1.5 mm assumed).
- Units metres; +X subject's left, -Y anterior, +Z superior. `scripts/soft_tissue/atlas_register.py`
  places them on the patient (RT143) by fitting the atlas temporal bone to the CT bone.
