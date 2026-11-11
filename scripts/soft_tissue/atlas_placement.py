"""Place the atlas temporalis and posterior auricular muscle on the patient, by hand, on the CT bone.

Starts from the automatic fit (atlas_register.py: the atlas temporal bone fitted to the CT bone).
Shows the CT bone, the ear canal, the atlas temporal bone (see-through blue: line its zygoma root
and mastoid up with the CT's), the temporalis (red) and the posterior auricular muscle (orange).

Sliders move, turn and scale the whole atlas group in anatomical axes; turns and scale are about
the canal's bony mouth. Landmark snap: tick "Pick on CT bone", Ctrl+click the zygoma root (or any
point) on the CT bone; tick "Pick on atlas bone", Ctrl+click the same place on the blue bone;
"Snap" moves the group so the two meet. Save writes atlas_placement.json and the placed meshes
(atlas_*.ply) next to the composite, where the soft-ear viewer loads them.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/atlas_placement.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import atlas_register as ar  # noqa: E402
import auricle_sim as asim  # noqa: E402
import envelope as ev  # noqa: E402

SHOW = [("atlas temporal bone", "temporal_bone_r", (90, 140, 255), 0.45),
        ("temporalis", "temporalis_muscle_r", (215, 80, 70), 1.0),
        ("posterior auricular muscle", "posterior_auricular_muscle_r", (255, 150, 40), 1.0),
        ("temporoparietalis", "temporoparietalis_muscle_r", (230, 150, 150), 0.5),
        ("superficial temporal fascia", "superficial_layer_of_temporal_fascia_r", (240, 230, 200), 0.5)]
HIDDEN = {"temporoparietalis", "superficial temporal fascia"}


class AtlasPlacement:
    def __init__(self):
        import open3d as o3d
        from open3d.visualization import gui, rendering

        self.o3d, self.gui, self.rendering = o3d, gui, rendering
        self.vol = ev.Volume(asim.ADF)
        f = self.vol.frame
        self.A, self.S, self.L = (np.asarray(x, float) for x in (f.anterior, f.superior, f.lateral))
        import trimesh

        bone = trimesh.load(asim.COMPOSITE / "bone.ply", process=False)
        self.bone_v, self.bone_t = np.asarray(bone.vertices, float), np.asarray(bone.faces)
        self.pivot, _ = ar.patient_meatus(self.vol, self.bone_v)
        self.fit = ar.fit_matrix()
        self.src = {key: ar.load_atlas(name) for _l, name, _c, _a in SHOW for key in [name]}
        self.move, self.turn, self.scale = np.zeros(3), np.zeros(3), 1.0
        self.picks = {}
        if ar.PLACEMENT.exists():
            d = json.loads(ar.PLACEMENT.read_text())
            self.move, self.turn, self.scale = np.asarray(d["move_mm"], float), np.asarray(d["turn_deg"], float), float(d["scale"])
            self.picks = {k: np.asarray(v, float) for k, v in d.get("picks", {}).items()}

        app = gui.Application.instance
        app.initialize()
        self.win = app.create_window("Atlas placement: temporalis and posterior auricular muscle", 1600, 1000)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.win.renderer)
        self.scene.scene.set_background([0.09, 0.10, 0.11, 1.0])
        self.scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.MED_SHADOWS, [0.3, -0.6, -0.75])
        self.scene.set_on_mouse(self._on_mouse)
        self.win.add_child(self.scene)
        self._panel()
        self.win.set_on_layout(self._layout)
        self.scene.scene.add_geometry("CT bone", self._mesh(self.bone_v, self.bone_t), self._mat((242, 238, 226)))
        eav, eat = self.vol.eac_mesh()
        self.scene.scene.add_geometry("canal", self._mesh(eav, eat), self._mat((255, 110, 60)))
        self.ct_pick = o3d.t.geometry.RaycastingScene()
        self.ct_pick.add_triangles(o3d.core.Tensor(self.bone_v.astype(np.float32)), o3d.core.Tensor(self.bone_t.astype(np.uint32)))
        self._update()
        c = self.pivot
        self.scene.setup_camera(35, o3d.geometry.AxisAlignedBoundingBox(c - 60, c + 60), c)
        self.scene.look_at(c, c + self.L * 220, self.S)

    # -- geometry ------------------------------------------------------------------------
    def _mat(self, rgb, alpha=1.0):
        m = self.rendering.MaterialRecord()
        m.shader = "defaultLitTransparency" if alpha < 1.0 else "defaultLit"
        m.base_color = [*(np.asarray(rgb) / 255.0), alpha]
        return m

    def _mesh(self, v, t):
        o3d = self.o3d
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(v, float)), o3d.utility.Vector3iVector(np.asarray(t, np.int32)))
        m.compute_vertex_normals()
        return m

    def matrix(self):
        return ar.user_matrix(self.vol.frame, self.pivot, self.move, self.turn, self.scale) @ self.fit

    def _update(self):
        T = self.matrix()
        sc = self.scene.scene
        self.placed = {}
        for label, name, col, alpha in SHOW:
            v, t = self.src[name]
            y = v @ T[:3, :3].T + T[:3, 3]
            self.placed[name] = (y, t)
            if sc.has_geometry(label):
                sc.remove_geometry(label)
            sc.add_geometry(label, self._mesh(y, t), self._mat(col, alpha))
            sc.show_geometry(label, self.boxes[label].checked)
        self._draw_picks()
        self._report()

    def _draw_picks(self):
        sc = self.scene.scene
        for key, col in (("ct", (60, 220, 90)), ("atlas", (60, 120, 255))):
            if sc.has_geometry("pick " + key):
                sc.remove_geometry("pick " + key)
            if key in self.picks:
                p = self.picks[key]
                if key == "atlas":  # stored in atlas mm: show where it is now
                    T = self.matrix()
                    p = T[:3, :3] @ p + T[:3, 3]
                s = self.o3d.geometry.TriangleMesh.create_sphere(1.2)
                s.translate(p)
                s.compute_vertex_normals()
                sc.add_geometry("pick " + key, s, self._mat(col))

    def _report(self):
        from scipy.spatial import cKDTree

        y, _t = self.placed["temporal_bone_r"]
        lo, hi = self.bone_v.min(0) + 2, self.bone_v.max(0) - 2
        inside = y[((y > lo) & (y < hi)).all(1)]
        d = cKDTree(self.bone_v[::3]).query(inside)[0] if len(inside) else np.array([np.nan])
        tm, _ = self.placed["temporalis_muscle_r"]
        tin = ((tm > self.bone_v.min(0)) & (tm < self.bone_v.max(0))).all(1).mean()
        gap = ""
        if "ct" in self.picks and "atlas" in self.picks:
            T = self.matrix()
            gap = f"; picks {np.linalg.norm(T[:3, :3] @ self.picks['atlas'] + T[:3, 3] - self.picks['ct']):.1f} mm apart"
        self.readout.text = (f"atlas bone to CT bone: median {np.median(d):.2f} mm, 90% {np.percentile(d, 90):.2f} mm\n"
                             f"temporalis inside the CT cube: {tin:.0%}{gap}")

    # -- panel -----------------------------------------------------------------------------
    def _panel(self):
        gui = self.gui
        em = self.win.theme.font_size
        p = gui.ScrollableVert(0.25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.panel = p
        p.add_child(gui.Label("Show"))
        self.boxes = {}
        for label, _n, _c, _a in SHOW:
            b = gui.Checkbox(label)
            b.checked = label not in HIDDEN
            b.set_on_checked(lambda on, label=label: self.scene.scene.show_geometry(label, on))
            self.boxes[label] = b
            p.add_child(b)
        p.add_child(gui.Label("Move (mm): anterior, superior, lateral"))
        self.sl = []
        for i in range(3):
            self.sl.append(self._slider(p, "ASL"[i], -25, 25, self.move[i], lambda v, i=i: self._set("move", i, v)))
        p.add_child(gui.Label("Turn (deg) about anterior, superior, lateral"))
        for i in range(3):
            self.sl.append(self._slider(p, "ASL"[i], -30, 30, self.turn[i], lambda v, i=i: self._set("turn", i, v)))
        self.sl.append(self._slider(p, "scale", 0.7, 1.3, self.scale, lambda v: self._set("scale", 0, v)))
        p.add_child(gui.Label("Landmark snap (Ctrl+click)"))
        self.pick_ct = gui.Checkbox("Pick on CT bone (green)")
        self.pick_at = gui.Checkbox("Pick on atlas bone (blue)")
        self.pick_ct.set_on_checked(lambda on: setattr(self.pick_at, "checked", False) if on else None)
        self.pick_at.set_on_checked(lambda on: setattr(self.pick_ct, "checked", False) if on else None)
        p.add_child(self.pick_ct)
        p.add_child(self.pick_at)
        for text, fn in (("Snap atlas pick onto CT pick", self._snap), ("Reset to automatic fit", self._reset), ("Save", self._save)):
            b = gui.Button(text)
            b.set_on_clicked(fn)
            p.add_child(b)
        self.readout = gui.Label("")
        p.add_child(self.readout)
        self.status = gui.Label("")
        p.add_child(self.status)
        self.win.add_child(p)

    def _slider(self, parent, label, lo, hi, value, fn):
        gui = self.gui
        row = gui.Horiz(4)
        row.add_child(gui.Label(label))
        s = gui.Slider(gui.Slider.DOUBLE)
        s.set_limits(lo, hi)
        s.double_value = float(value)
        s.set_on_value_changed(fn)
        row.add_child(s)
        parent.add_child(row)
        return s

    def _layout(self, ctx):
        r = self.win.content_rect
        w = 24 * ctx.theme.font_size
        self.scene.frame = self.gui.Rect(r.x, r.y, r.width - w, r.height)
        self.panel.frame = self.gui.Rect(r.get_right() - w, r.y, w, r.height)

    def _set(self, what, i, v):
        if what == "move":
            self.move[i] = v
        elif what == "turn":
            self.turn[i] = v
        else:
            self.scale = v
        self._update()

    def _sync_sliders(self):
        vals = [*self.move, *self.turn, self.scale]
        for s, v in zip(self.sl, vals):
            s.double_value = float(v)

    def _snap(self):
        if not ("ct" in self.picks and "atlas" in self.picks):
            self.status.text = "pick the same landmark on both bones first"
            return
        T = self.matrix()
        d = self.picks["ct"] - (T[:3, :3] @ self.picks["atlas"] + T[:3, 3])
        self.move += np.array([d @ self.A, d @ self.S, d @ self.L])
        self._sync_sliders()
        self._update()
        self.status.text = f"snapped: moved {np.linalg.norm(d):.1f} mm"

    def _reset(self):
        self.move, self.turn, self.scale = np.zeros(3), np.zeros(3), 1.0
        self._sync_sliders()
        self._update()

    def _save(self):
        d = {"pivot_mm": self.pivot.tolist(), "move_mm": self.move.tolist(), "turn_deg": self.turn.tolist(),
             "scale": self.scale, "picks": {k: v.tolist() for k, v in self.picks.items()},
             "note": "user adjustment on top of atlas_registration.json (atlas_register.user_matrix)"}
        ar.PLACEMENT.write_text(json.dumps(d, indent=1))
        ar.write_placed(self.matrix(), log=lambda *a: None)
        self.status.text = f"saved {ar.PLACEMENT.name} and the placed meshes"

    # -- mouse -------------------------------------------------------------------------------
    def _on_mouse(self, e):
        gui = self.gui
        if (self.pick_ct.checked or self.pick_at.checked) and e.type == gui.MouseEvent.Type.BUTTON_DOWN \
                and e.is_modifier_down(gui.KeyModifier.CTRL):
            x, y = e.x - self.scene.frame.x, e.y - self.scene.frame.y
            if self.pick_ct.checked:
                p = self._pick(x, y, self.ct_pick)
                if p is not None:
                    self.picks["ct"] = p
            else:
                v, t = self.placed["temporal_bone_r"]
                sc = self.o3d.t.geometry.RaycastingScene()
                sc.add_triangles(self.o3d.core.Tensor(v.astype(np.float32)), self.o3d.core.Tensor(t.astype(np.uint32)))
                p = self._pick(x, y, sc)
                if p is not None:  # kept in atlas mm, so it moves with the atlas
                    T = self.matrix()
                    self.picks["atlas"] = np.linalg.solve(T[:3, :3], p - T[:3, 3])
            self._draw_picks()
            self._report()
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED

    def _pick(self, x, y, scene):
        cam = self.scene.scene.camera
        w, h = self.scene.frame.width, self.scene.frame.height
        V = np.asarray(cam.get_view_matrix())
        eye = -V[:3, :3].T @ V[:3, 3]
        d = np.asarray(cam.unproject(x, y, 0.5, w, h), float) - eye
        d /= max(np.linalg.norm(d), 1e-12)
        t = float(scene.cast_rays(self.o3d.core.Tensor(np.concatenate([eye, d])[None].astype(np.float32)))["t_hit"].numpy()[0])
        return None if not np.isfinite(t) else eye + d * t

    def run(self):
        self.gui.Application.instance.run()


if __name__ == "__main__":
    AtlasPlacement().run()
