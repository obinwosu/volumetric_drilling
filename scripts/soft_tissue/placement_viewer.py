"""Place the auricle on a FIVRS volume and draw the temporalis outline, in 3D.

Shows the bone, the ear canal (EAC label), the soft-tissue envelope layers
(``envelope.py``) and ear.obj. Move, turn and scale the ear with the sliders;
the axes are anatomical (anterior, superior, lateral), centred on the canal's
lateral opening. Readouts say how deep the ear sits under the skin and whether
it touches bone.

Mouse: Ctrl+drag slides the ear in the view plane (it stays under the cursor),
Shift+drag turns it about its centre, and Ctrl+wheel pushes it away from or
toward you. The "Left-drag" picker makes a plain drag (and the wheel) do the
same, which helps in the browser view; set it back to "orbits the camera" to
look around.

Temporalis outline: tick "Pick muscle outline", then click points on the bone
around where the muscle should be (dragging still turns the camera). From the
third point on, the envelope rebuilds: the muscle fills the outline, thickest
inside and tapering to its edge, and the fascia and outer layers rise over it.
Thickness sliders rebuild it too, a moment after you let go.

Cut: ear.obj is a solid block (a thin auricle in front, a filled bulge
behind). The "Cut" sliders set a gently curved surface; the part shown magenta is
trimmed away and the opening closed, leaving the auricle. "Build cartilage
framework" applies the cut and shrinks the auricle by the skin + perichondrium
cover.

Cuts you draw: build the framework, tick "Draw cut on skin" or "... on
cartilage", look straight at the area, click points around it and press Cut.
Everything inside the outline along your view is removed (all the way through,
or to the set depth), and the mesh closes over the cut. Use it for the concha
opening (skin and cartilage), the lobule (cartilage only), and the like. Cuts
are kept in the ear's own frame, so they move with the ear; "Undo last cut"
steps back.

Save writes ``resources/soft_tissue/envelope/<volume>_placement.json`` (ear
transform to volume mm and to the AMBF world, muscle outline, thicknesses) and
the meshes (layers, placed ear, framework) next to it. The viewer reopens the
saved placement.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/placement_viewer.py [ADF] [--ear ear.obj]
    ... --snapshot out.png     render once and exit (for checks)
    ... --open-meatus          open the CT ear's meatus and trim it at launch
    ... --web [--port 8889]    serve the same window to a browser at http://localhost:8888 (or the given port)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import auricle as au  # noqa: E402
import blender_edit as be  # noqa: E402
import ct_meatus as cm  # noqa: E402
import envelope as ev  # noqa: E402
import ear_cuts as ec  # noqa: E402

OBJ_EAR = pathlib.Path.home() / "Downloads" / "ear.obj"
CT_EAR = ev.ROOT / "resources" / "soft_tissue" / "auricle" / "ct_right_ear.ply"  # ct_ear.py
DEFAULT_EAR = CT_EAR if CT_EAR.exists() else OBJ_EAR


def ear_marks_file(ear_path):
    """Landmarks written next to a CT ear by ct_ear.py, or None for a sculpted ear."""
    p = pathlib.Path(ear_path)
    j = p.with_name(p.stem + ".landmarks.json")
    return j if j.exists() else None
DISPLAY_EAR_TRIS = 60000


def rot(axis, degrees):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(degrees)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class Placement:
    """Ear pose: p_vol = anchor + R_user (scale * B * M (p_obj - pivot))."""

    def __init__(self, frame: ev.Frame, pivot):
        self.f = frame
        self.pivot = np.asarray(pivot, float)
        # first guess: obj +x faces lateral, obj +y is superior
        self.base = np.stack([frame.lateral, frame.superior, np.cross(frame.lateral, frame.superior)], axis=1)
        self.mirror = False
        self.move_mm = np.array([0.0, 8.0, 6.0])  # anterior, superior, lateral of the canal opening
        self.turn_deg = np.zeros(3)  # about anterior, superior, lateral
        self.scale = 1.0

    def axes(self):
        return self.f.anterior, self.f.superior, self.f.lateral

    def user_rotation(self):
        a, s, l = self.axes()
        return rot(l, self.turn_deg[2]) @ rot(s, self.turn_deg[1]) @ rot(a, self.turn_deg[0])

    def matrix(self):
        """4x4, obj mm -> volume mm."""
        a, s, l = self.axes()
        M = np.diag([-1.0 if self.mirror else 1.0, 1.0, 1.0])
        L = self.user_rotation() @ (self.scale * self.base @ M)
        anchor = self.f.canal_lateral_end + self.move_mm[0] * a + self.move_mm[1] * s + self.move_mm[2] * l
        T = np.eye(4)
        T[:3, :3] = L
        T[:3, 3] = anchor - L @ self.pivot
        return T

    def to_dict(self):
        return dict(pivot_obj_mm=self.pivot.tolist(), base=self.base.tolist(), mirror=self.mirror,
                    move_mm=self.move_mm.tolist(), turn_deg=self.turn_deg.tolist(), scale=self.scale)

    def load(self, d, saved_frame=None):
        self.pivot = np.asarray(d["pivot_obj_mm"], float)
        self.base = np.asarray(d["base"], float)
        self.mirror = bool(d["mirror"])
        self.move_mm = np.asarray(d["move_mm"], float)
        self.turn_deg = np.asarray(d["turn_deg"], float)
        self.scale = float(d["scale"])
        if saved_frame is not None and "ear_to_volume_mm" in d:
            same = all(np.allclose(saved_frame[k], getattr(self.f, k), atol=1e-6) for k in ("anterior", "superior", "lateral"))
            if not same:
                self.adopt(np.asarray(d["ear_to_volume_mm"], float))

    def auto(self, marks, tilt_deg=15.0, flare_deg=30.0, meatus_mm=14.0):
        """Anatomical starting pose from the ear's landmarks (``auricle.ear_landmarks``).

        The concha floor sits ``meatus_mm`` lateral of the canal's lateral opening, on the canal
        axis. The ear's long axis leans back from superior by ``tilt_deg``. Its face looks
        lateral turned toward anterior by ``flare_deg``, so the posterior helix stands off the
        head (the cranioauricular angle). A left-ear model is mirrored.
        """
        A, S, L = self.axes()
        fl, tl = np.radians(flare_deg), np.radians(tilt_deg)
        face_t = np.cos(fl) * L + np.sin(fl) * A
        sup_t = np.cos(tl) * S - np.sin(tl) * A
        sup_t -= (sup_t @ face_t) * face_t
        sup_t /= np.linalg.norm(sup_t)
        ant_t = A - (A @ face_t) * face_t - (A @ sup_t) * sup_t
        ant_t /= np.linalg.norm(ant_t)
        Bt = np.stack([sup_t, ant_t, face_t], axis=1)
        Bo = np.stack([marks["superior"], marks["anterior"], marks["face"]], axis=1)
        Bo[:, 1] -= (Bo[:, 1] @ Bo[:, 0]) * Bo[:, 0] + (Bo[:, 1] @ Bo[:, 2]) * Bo[:, 2]
        Bo[:, 1] /= np.linalg.norm(Bo[:, 1])
        Lin = self.scale * Bt @ np.linalg.inv(Bo)  # obj -> volume; a reflection for a left-ear model
        self.mirror = bool(np.linalg.det(Lin) < 0)
        T = np.eye(4)
        T[:3, :3] = Lin
        target = self.f.canal_lateral_end + L * meatus_mm
        T[:3, 3] = target - Lin @ marks["concha"]
        self.adopt(T)

    def adopt(self, T):
        """Re-express a saved ear pose (obj mm -> volume mm) in this frame's axes, unchanged in space."""
        a, s, l = self.axes()
        M = np.diag([-1.0 if self.mirror else 1.0, 1.0, 1.0])
        self.turn_deg = np.zeros(3)
        self.base = (T[:3, :3] / self.scale) @ np.linalg.inv(M)
        anchor = T[:3, :3] @ self.pivot + T[:3, 3]
        r = anchor - self.f.canal_lateral_end
        self.move_mm = np.array([r @ a, r @ s, r @ l])


def apply(T, v):
    return np.asarray(v) @ T[:3, :3].T + T[:3, 3]


def load_ear(path):
    import trimesh

    m = trimesh.load(str(path), process=False, force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)


class Checks:
    """How the placed ear sits relative to skin and bone."""

    def __init__(self, vol: ev.Volume):
        self.vol = vol
        self.skin = None

    def set_skin(self, v, t):
        import open3d as o3d

        self.skin = o3d.t.geometry.RaycastingScene()
        self.skin.add_triangles(o3d.core.Tensor(np.asarray(v, np.float32)), o3d.core.Tensor(np.asarray(t, np.uint32)))

    def report(self, pts):
        out = []
        if self.skin is not None:
            import open3d as o3d

            q = self.skin.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))
            cp = q["points"].numpy()
            n = q["primitive_normals"].numpy()
            sd = np.einsum("ij,ij->i", pts - cp, n)  # < 0: under the skin
            under = sd < 0
            out.append(f"ear under skin: {under.mean() * 100:.0f}% of points, deepest {max(0.0, -sd.min()):.1f} mm")
            a = (pts - pts.mean(axis=0)) @ self.vol.frame.anterior
            back = a < np.percentile(a, 10)  # posterior edge of the ear (helix rim)
            out.append(f"posterior helix off the skin: {max(0.0, sd[back].max()):.1f} mm")
        idx = np.round(pts / self.vol.mm).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(self.vol.shape)), axis=1)
        in_bone = np.zeros(len(pts), bool)
        in_bone[ok] = self.vol.solid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        out.append(f"ear inside bone: {in_bone.sum()} points" + ("  (move it out)" if in_bone.any() else ""))
        return "\n".join(out)


class App:
    def __init__(self, adf: pathlib.Path, ear_path: pathlib.Path, snapshot=None, size=(1700, 1050), open_meatus=False):
        import open3d as o3d
        from open3d.visualization import gui, rendering

        self.o3d, self.gui, self.rendering = o3d, gui, rendering
        self.adf, self.ear_path, self.snapshot = adf, ear_path, snapshot
        t0 = time.time()
        self.vol = ev.Volume(adf)
        self.env = ev.Envelope(self.vol)
        self.muscle = ev.Muscle()
        self.thickness = {n: t for n, t, _ in ev.LAYERS}
        self.ramp = ev.AuricleRamp()  # the envelope thins around the canal opening (concha, meatus)
        self.ramp_on = True
        self.ear_v, self.ear_t = load_ear(ear_path)
        self.ear_aw = None  # painted auricle weight per ear vertex (Blender edit), or None
        self.cart_edit = None  # the cartilage as edited in Blender (ear frame), or None
        self.use_blender = True  # use the Blender edit of the CT ear when there is one
        self.framework_max_mm = 0.0  # cartilage thickness limit (0: fill down to the envelope)
        lo, hi = self.ear_v.min(axis=0), self.ear_v.max(axis=0)
        self.place = Placement(self.vol.frame, 0.5 * (lo + hi))
        self.framework = None
        self.cut = au.Cut()
        self.ear_axes = au.ear_axes(self.ear_v, self.ear_t)
        self.auricle = None  # the cut mesh, once built
        self.cuts = []  # hand-drawn ec.MeshCut, applied in order
        self.flattens = []  # CT ear: lassoed non-auricle tissue pressed under the skin (ec.MeshCut, ear frame)
        self.meatus = ec.MeatusShape()  # automatic concha opening into the canal
        self.cut_points = []  # outline being drawn, volume mm
        self.cartilage = None  # framework with cartilage cuts, obj frame
        self.scalp_joined = None  # scalp skin with the ear's footprint removed, volume mm
        self.layers_joined = {}  # deeper layers trimmed under the CT ear's bowls (ct_meatus), volume mm
        self.sleeve = None  # cartilaginous canal (incomplete sleeve), volume mm
        self.funnel = None  # skin lining from the canal's rim to the ear's meatus hole, volume mm
        self.ct_open = None  # ct_meatus.open_meatus result (opened ear, scalp, canal skin), volume mm
        self.ct_open_T = None  # the ear pose it was made for
        self._eac_field = None
        self._open_on_start = False
        self.checks = Checks(self.vol)
        # a CT ear: real back surface and crease, right side, no bulge to trim
        mf = ear_marks_file(ear_path)
        self.ct_marks = {k: np.asarray(x, float) for k, x in json.loads(mf.read_text()).items() if k != "source"} if mf else None
        self.ear_marks = {}  # clicked landmarks (ear model mm): meatus, canal
        if self.ct_marks is not None:
            self.cut.enabled = False
        saved = ev.placement_path(adf, ear_path)
        shared = ev.placement_path(adf)
        if saved.exists():
            doc = json.loads(saved.read_text())
            self.place.load(doc["ear"], doc.get("frame"))
            self.cut = au.Cut(**doc.get("cut", {}))
            self.cuts = [ec.MeshCut(**c) for c in doc.get("mesh_cuts", [])]
            self.flattens = [ec.MeshCut(**c) for c in doc.get("ct_flatten", [])]
            self.ear_marks = {k: np.asarray(x, float) for k, x in doc.get("ear_marks", {}).items()}
            print(f"Loaded {saved}")
        elif shared.exists():
            doc = json.loads(shared.read_text())  # carry the envelope settings over to this ear
            print(f"New ear: envelope settings from {shared.name}")
        else:
            doc = None
        if doc is not None:
            self.muscle = ev.Muscle(**doc.get("muscle", {}))
            self.thickness.update(ev.upgrade_thickness(doc.get("thickness_mm", {})))
            self.meatus = ec.MeatusShape(**doc.get("meatus", {}))
            r = doc.get("auricle_ramp")
            if r is not None:
                self.ramp_on = bool(r.get("on", True))
                self.ramp = ev.AuricleRamp(near={**self.ramp.near, **r.get("near", {})},
                                           near_mm=r.get("near_mm", self.ramp.near_mm), far_mm=r.get("far_mm", self.ramp.far_mm),
                                           base_near_mm=r.get("base_near_mm", 2.0), base_far_mm=r.get("base_far_mm", 12.0))
            b = doc.get("blender", {})
            self.use_blender = bool(b.get("use", True))
            self.framework_max_mm = float(b.get("framework_max_mm", 0.0))
        self.env.auricle_ramp = self.ramp if self.ramp_on else None
        if self.use_blender and self._edited_path().exists():
            self.ear_v, self.ear_t, self.ear_aw = be.read_ply(self._edited_path())
            print(f"CT ear: Blender edit ({int((self.ear_aw > 0.5).sum())} auricle vertices)")
            if self._cart_edited_path().exists():
                self.cart_edit = be.read_ply(self._cart_edited_path())[:2]
                print(f"CT ear: cartilage as edited in Blender ({len(self.cart_edit[1])} tris)")
        self._open_on_start = bool(saved.exists() and doc.get("ct_meatus", {}).get("opened"))
        self._needs_align = self.ct_marks is not None and not saved.exists()
        self._eac_field = getattr(self, "_eac_field", None)
        self._face_envelope()  # the face's tissue up to the auricle's front edge (Blender skin sheet)
        self.layers = self.env.layers(self.muscle, self.thickness)
        if self._needs_align:  # first time with a CT ear: line its canal up with the volume's
            self._align_pose(first=True)
        print(f"ready in {time.time() - t0:.1f} s")

        app = gui.Application.instance
        app.initialize()
        self.app = app
        self.win = app.create_window(f"Ear placement: {adf.stem}", int(size[0]), int(size[1]))
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.win.renderer)
        self.scene.scene.set_background([0.09, 0.10, 0.11, 1.0])  # dark, like the RTX window
        self.scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.MED_SHADOWS, [0.3, -0.6, -0.75])
        self.scene.set_on_mouse(self._on_mouse)
        self.scene.set_on_key(self._on_key)
        self.win.add_child(self.scene)
        self._build_panel()
        self.win.set_on_layout(self._on_layout)
        self._add_static()
        self._add_layers()
        self._add_ear()
        self._draw_outline()
        self._draw_cut_outline()
        self._draw_marks()
        self._update_ear()
        c = self.vol.frame.canal_lateral_end
        bb = o3d.geometry.AxisAlignedBoundingBox(c - 45, c + 45)
        self.scene.setup_camera(35, bb, c)
        self.scene.look_at(c, c + self.vol.frame.lateral * 170, self.vol.frame.superior)
        if self._open_on_start or open_meatus:
            self._open_meatus()
        # edits saved in Blender (Ctrl+S) are picked up by themselves
        self._blender_seen = max([p.stat().st_mtime for p in (self._edited_path(), self._cart_edited_path()) if p.exists()] or [0.0])
        if not snapshot:
            threading.Thread(target=self._watch_blender, daemon=True).start()
        if snapshot:
            threading.Thread(target=self._snap_later, daemon=True).start()

    # -- materials and geometry -----------------------------------------------------
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

    def _add_static(self):
        bv, bt = self.vol.bone_mesh()
        self.scene.scene.add_geometry("bone", self._mesh(bv, bt), self._mat((242, 238, 226)))
        # CPU ray picking on the bone (the skin is see-through; the outline is drawn on bone)
        self.pick_scene = self.o3d.t.geometry.RaycastingScene()
        self.pick_scene.add_triangles(self.o3d.core.Tensor(bv.astype(np.float32)), self.o3d.core.Tensor(bt.astype(np.uint32)))
        eav, eat = self.vol.eac_mesh()
        self.scene.scene.add_geometry("eac", self._mesh(eav, eat), self._mat((255, 110, 60)))

    def _layer_alpha(self, name):
        return self.alpha.double_value if name == "skin" else 0.85

    def _add_layers(self):
        sc = self.scene.scene
        for name in ev.LAYER_NAMES:
            if sc.has_geometry(name):
                sc.remove_geometry(name)
        for name, (v, t) in self.layers.items():
            if name == "skin" and getattr(self, "scalp_joined", None) is not None:
                v, t = self.scalp_joined  # scalp skin with the ear's footprint taken out
            v, t = getattr(self, "layers_joined", {}).get(name, (v, t))  # trimmed under the ear's bowls
            sc.add_geometry(name, self._mesh(v, t), self._mat(ev.LAYER_RGB[name], self._layer_alpha(name)))
            sc.show_geometry(name, self.layer_box[name].checked)
        if "skin" in self.layers:
            self.checks.set_skin(*self.layers["skin"])

    def _add_ear(self):
        m = self._mesh(self.ear_v, self.ear_t)
        if len(self.ear_t) > DISPLAY_EAR_TRIS:
            m = m.simplify_quadric_decimation(DISPLAY_EAR_TRIS)
            m.compute_vertex_normals()
        if self.place.mirror:  # a mirrored pose turns the triangles inside out
            m.triangles = self.o3d.utility.Vector3iVector(np.asarray(m.triangles)[:, [0, 2, 1]])
            m.compute_vertex_normals()
        self.ear_display = m
        self.ear_check_pts = np.asarray(m.vertices)[:: max(1, len(m.vertices) // 8000)]
        self.scene.scene.add_geometry("ear", m, self._mat((255, 255, 255), 1.0))
        self._color_cut()

    def _color_cut(self):
        """Paint the part the cut removes red (on the display mesh)."""
        v = np.asarray(self.ear_display.vertices)
        keep = np.ones(len(v), bool)
        if self.cut.enabled:
            frame = au._cut_frame(*self.ear_axes, self.cut)
            keep = au.cut_value(v, frame, self.cut) < 0
        col = np.where(keep[:, None], np.array([228, 170, 150]) / 255.0, np.array([235, 40, 210]) / 255.0)
        self.ear_display.vertex_colors = self.o3d.utility.Vector3dVector(col)
        sc = self.scene.scene
        if sc.has_geometry("ear"):
            sc.remove_geometry("ear")
        sc.add_geometry("ear", self.ear_display, self._mat((255, 255, 255)))
        sc.show_geometry("ear", self.layer_box["ear"].checked)
        if hasattr(self, "readout"):
            self._update_ear()

    def _update_ear(self):
        T = self.place.matrix()
        sc = self.scene.scene
        if self.ct_open is not None and not np.allclose(T, self.ct_open_T):
            self._clear_open("Ear moved: press Open meatus and trim again")
        sc.set_geometry_transform("ear", T)
        if self.ear_marks and hasattr(self, "marks_label"):
            self._draw_marks()  # the clicked points ride with the ear
        for name in ("framework", "auricle"):
            if sc.has_geometry(name):
                sc.set_geometry_transform(name, T)
        pts = apply(T, self.ear_check_pts)
        f = self.vol.frame
        rel = pts - f.canal_lateral_end
        axial = rel @ f.lateral
        near_axis = np.linalg.norm(rel - np.outer(axial, f.lateral), axis=1) < 2.0
        gap = f"\nconcha floor to canal end: {axial[near_axis].min():+.1f} mm" if near_axis.any() else ""
        self.readout.text = self.checks.report(pts) + gap

    def _draw_outline(self):
        sc = self.scene.scene
        for name in [n for n in getattr(self, "_pt_names", [])]:
            sc.remove_geometry(name)
        self._pt_names = []
        pts = np.asarray(self.muscle.outline_mm, float).reshape(-1, 3)
        for i, p in enumerate(pts):
            s = self.o3d.geometry.TriangleMesh.create_sphere(0.8)
            s.translate(p)
            s.compute_vertex_normals()
            sc.add_geometry(f"pt{i}", s, self._mat((60, 255, 120)))
            self._pt_names.append(f"pt{i}")
        if len(pts) >= 2:
            ls = self.o3d.geometry.LineSet(self.o3d.utility.Vector3dVector(pts),
                                            self.o3d.utility.Vector2iVector([[i, (i + 1) % len(pts)] for i in range(len(pts))]))
            m = self.rendering.MaterialRecord()
            m.shader = "unlitLine"
            m.line_width = 3.0
            m.base_color = [0.25, 1.0, 0.45, 1.0]
            sc.add_geometry("outline", ls, m)
            self._pt_names.append("outline")

    # -- panel ----------------------------------------------------------------------
    def _slider(self, parent, label, lo, hi, value, on_change, integer=False):
        gui = self.gui
        row = gui.Horiz(4)
        row.add_child(gui.Label(label))
        s = gui.Slider(gui.Slider.INT if integer else gui.Slider.DOUBLE)
        s.set_limits(lo, hi)
        if integer:
            s.int_value = int(value)
        else:
            s.double_value = float(value)
        s.set_on_value_changed(on_change)
        row.add_child(s)
        parent.add_child(row)
        return s

    def _button(self, parent, text, fn):
        b = self.gui.Button(text)
        b.set_on_clicked(fn)
        parent.add_child(b)
        return b

    def _build_panel(self):
        """The side panel, in the order of the work: place, build, edit in Blender, envelope, show."""
        gui = self.gui
        em = self.win.theme.font_size
        self.panel = gui.ScrollableVert(0.4 * em, gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em))
        p = self.panel

        def section(title, open_=False, parent=None):
            c = gui.CollapsableVert(title, 0.25 * em, gui.Margins(em, 0, 0, 0))
            c.set_is_open(open_)
            (parent or p).add_child(c)
            return c

        # -- 1. place the ear ------------------------------------------------------------
        ear = section("1  Place the ear")
        self.sl = {}
        for i, name in enumerate(("anterior", "superior", "lateral")):
            self.sl[name] = self._slider(ear, f"{name} mm", -50, 50, self.place.move_mm[i], lambda v, i=i: self._move(i, v))
        for i, name in enumerate(("turn about anterior", "turn about superior", "turn about lateral")):
            self.sl[name] = self._slider(ear, f"{name}", -180, 180, self.place.turn_deg[i], lambda v, i=i: self._turn(i, v))
        self.sl["scale"] = self._slider(ear, "scale", 0.6, 1.4, self.place.scale, self._scale)
        row = gui.Horiz(4)
        for text, axis in (("90° ant", 0), ("90° sup", 1), ("90° lat", 2)):
            self._button(row, text, lambda a=axis: self._quarter(a))
        ear.add_child(row)
        self.mirror = gui.Checkbox("mirror (left ear model on a right ear)")
        self.mirror.checked = self.place.mirror
        self.mirror.set_on_checked(self._mirror)
        ear.add_child(self.mirror)
        row = gui.Horiz(4)
        row.add_child(gui.Label("Left-drag"))
        self.drag_mode = gui.Combobox()
        for text in ("orbits the camera", "slides the ear", "turns the ear"):
            self.drag_mode.add_item(text)
        row.add_child(self.drag_mode)
        ear.add_child(row)
        ear.add_child(gui.Label("Ctrl+drag slides, Shift+drag turns the ear;\n"
                                "arrow keys / PgUp PgDn nudge 0.5 mm"))
        self._button(ear, "Orbit camera around the ear", self._focus_ear)
        self.readout = gui.Label("")
        ear.add_child(self.readout)
        lm = section("Landmarks and canal alignment", parent=ear)
        self.pick_meatus = gui.Checkbox("Pick meatus (centre of the canal opening)")
        self.pick_canal = gui.Checkbox("Pick canal (optional: deepest canal point)")
        self.pick_meatus.set_on_checked(lambda on: self._pick_mark("meatus", on))
        self.pick_canal.set_on_checked(lambda on: self._pick_mark("canal", on))
        lm.add_child(self.pick_meatus)
        lm.add_child(self.pick_canal)
        self.marks_label = gui.Label("")
        lm.add_child(self.marks_label)
        self._button(lm, "Align ear to canal", self._align_ui)
        auto = section("Auto-place (anatomical start)", parent=ear)
        self.sl_tilt = self._slider(auto, "tilt back deg", 0, 40, 15, lambda v: None)
        self.sl_flare = self._slider(auto, "flare deg", 0, 60, 30, lambda v: None)
        self.sl_meatus = self._slider(auto, "concha lateral of canal end mm", -5, 25, 14, lambda v: None)
        self._button(auto, "Auto-place ear", self._auto_place)

        # -- 2. build -----------------------------------------------------------------------
        build = section("2  Build the ear", open_=True)
        build.add_child(gui.Label("Opens the meatus onto the canal, joins the ear to the\n"
                                  "scalp skin, lines the canal, builds the cartilage.\n"
                                  "Moving the ear clears it."))
        row = gui.Horiz(4)
        self.ct_btn = self._button(row, "Build ear", self._open_meatus)
        self._button(row, "Undo (raw ear)", self._revert_open)
        build.add_child(row)
        self.ct_label = gui.Label("")
        build.add_child(self.ct_label)

        # -- 3. edit in Blender ---------------------------------------------------------------
        bl = section("3  Edit in Blender", open_=True)
        bl.add_child(gui.Label("Skin: paint the auricle ('auricle' group), cut it down.\n"
                               "Cartilage: cut the incisura, model it.\n"
                               "Ctrl+S in Blender sends the edits back here."))
        row = gui.Horiz(4)
        self._button(row, "Edit skin", self._edit_in_blender)
        self._button(row, "Edit cartilage", self._edit_cartilage_in_blender)
        bl.add_child(row)
        self.blender_box = gui.Checkbox("use the Blender edits (off: original CT ear)")
        self.blender_box.checked = self.use_blender
        self.blender_box.set_on_checked(self._toggle_blender)
        bl.add_child(self.blender_box)
        blx = section("More", parent=bl)
        self._slider(blx, "cartilage under one skin, mm (0 = 1.5)", 0.0, 10.0, self.framework_max_mm,
                     lambda v: (setattr(self, "framework_max_mm", v), self.ear_aw is not None and self.ct_open is not None and self._open_meatus()))
        row = gui.Horiz(4)
        self._button(row, "Load edits now", self._load_blender_edit)
        self._button(row, "Rebuild cartilage", self._drop_cartilage_edit)
        blx.add_child(row)
        blx.add_child(gui.Label("Rebuild cartilage sets your cartilage edit aside\n"
                                "(.ply.bak) and builds it from the skin again."))

        # -- 4. envelope ----------------------------------------------------------------------
        envp = section("4  Skin envelope")
        thick = section("Layer thickness (mm)", parent=envp)
        for name in ev.LAYER_NAMES:
            if name in ("temporalis_muscle", "auricular_muscles"):  # thickness maps, not sliders
                continue
            self._slider(thick, name.replace("_", " "), 0.0, 8.0, self.thickness[name],
                         lambda v, n=name: (self.thickness.__setitem__(n, v), self._rebuild_soon()))
        ramp = section("Thinner around the ear", parent=envp)
        ramp.add_child(gui.Label("Near the canal opening (concha, meatus) the 'near'\n"
                                 "values; the ones above beyond 'full by'."))
        self.ramp_box = gui.Checkbox("thin around the ear")
        self.ramp_box.checked = self.ramp_on
        self.ramp_box.set_on_checked(self._ramp_toggled)
        ramp.add_child(self.ramp_box)
        ramp.add_child(gui.Label("With the Blender skin it is measured from the auricle's base\n"
                                 "(under the auricle: thin); otherwise from the canal opening."))
        self._slider(ramp, "from base: thin within mm", 0.0, 20.0, self.ramp.base_near_mm,
                     lambda v: (setattr(self.ramp, "base_near_mm", v), self._rebuild_soon()))
        self._slider(ramp, "from base: full by mm", 1.0, 40.0, self.ramp.base_far_mm,
                     lambda v: (setattr(self.ramp, "base_far_mm", v), self._rebuild_soon()))
        self._slider(ramp, "from canal: thin within mm", 0.0, 30.0, self.ramp.near_mm, lambda v: (setattr(self.ramp, "near_mm", v), self._rebuild_soon()))
        self._slider(ramp, "from canal: full by mm", 1.0, 50.0, self.ramp.far_mm, lambda v: (setattr(self.ramp, "far_mm", v), self._rebuild_soon()))
        for name in self.ramp.near:
            self._slider(ramp, f"near: {name.replace('_', ' ')}", 0.0, 8.0, self.ramp.near[name],
                         lambda v, n=name: (self.ramp.near.__setitem__(n, v), self._rebuild_soon()))
        mus = section("Temporalis", parent=envp)
        self.picking = gui.Checkbox("Pick muscle outline (click on bone)")
        self.picking.set_on_checked(lambda on: (setattr(self.draw_skin, "checked", False), setattr(self.draw_cart, "checked", False)) if on else None)
        mus.add_child(self.picking)
        row = gui.Horiz(4)
        self._button(row, "Undo point", self._undo_point)
        self._button(row, "Clear", self._clear_points)
        mus.add_child(row)
        self.sl_mmax = self._slider(mus, "max thickness mm", 1, 20, self.muscle.max_mm, lambda v: (setattr(self.muscle, "max_mm", v), self._rebuild_soon()))
        self.sl_taper = self._slider(mus, "taper mm", 1, 30, self.muscle.taper_mm, lambda v: (setattr(self.muscle, "taper_mm", v), self._rebuild_soon()))
        self.rebuild_btn = self._button(envp, "Rebuild envelope", self._rebuild)

        # -- 5. show -----------------------------------------------------------------------------
        show = section("5  Show")
        self.layer_box = {}
        for name, default in (("bone", True), ("eac", True), ("ear", True), ("opened ear", True), ("canal skin", True),
                              ("auricle cartilage", True), ("auricle", True), ("framework", True), ("funnel", True), ("canal cartilage", True)):
            cb = gui.Checkbox(name)
            cb.checked = default
            cb.set_on_checked(lambda on, n=name: self.scene.scene.show_geometry(n, on) if self.scene.scene.has_geometry(n) else None)
            show.add_child(cb)
            self.layer_box[name] = cb
        for name in ev.LAYER_NAMES:
            cb = gui.Checkbox(name.replace("_", " "))
            cb.checked = name in ("skin", "temporalis_muscle")
            cb.set_on_checked(lambda on, n=name: self.scene.scene.show_geometry(n, on) if self.scene.scene.has_geometry(n) else None)
            show.add_child(cb)
            self.layer_box[name] = cb
        self.alpha = self._slider(show, "skin opacity", 0.1, 1.0, 0.45, lambda v: self._add_layers())
        sol = section("Layers as solids (cutaway)", parent=show)
        self.solids_on = gui.Checkbox("show layer solids")
        self.solids_on.set_on_checked(lambda on: self._show_solids(on))
        sol.add_child(self.solids_on)
        self.cut_on_solids = gui.Checkbox("cutaway")
        self.cut_on_solids.checked = True
        sol.add_child(self.cut_on_solids)
        self.solid_axis = "anterior"
        row = gui.Horiz(4)
        for ax in ("anterior", "superior", "lateral"):
            self._button(row, ax, lambda a=ax: (setattr(self, "solid_axis", a), self._build_solids()))
        sol.add_child(row)
        self.sl_solid = self._slider(sol, "cut position mm", -40, 40, 0, lambda v: self._rebuild_solids_soon())
        self.solid_btn = self._button(sol, "Update solids", self._build_solids)
        self.solid_label = gui.Label("")
        sol.add_child(self.solid_label)

        # -- older tools (the sculpted ear.obj, and drawing in this window) -------------------------
        old = section("Older tools (ear.obj, drawing here)")
        old.add_child(gui.Label("From before the Blender workflow. Kept for ear.obj."))
        cutp = section("Trim the bulge behind the auricle", parent=old)
        self.cut_on = gui.Checkbox("cut on")
        self.cut_on.checked = self.cut.enabled
        self.cut_on.set_on_checked(lambda on: self._set_cut("enabled", on))
        cutp.add_child(self.cut_on)
        for field, label, lo, hi in (("depth_mm", "depth mm", -6, 12), ("tilt1_deg", "tilt about long axis", -40, 40),
                                     ("tilt2_deg", "tilt about cross axis", -40, 40), ("bend1_mm", "bend along length mm", -8, 8),
                                     ("bend2_mm", "bend across mm", -8, 8)):
            self._slider(cutp, label, lo, hi, getattr(self.cut, field), lambda v, f=field: self._set_cut(f, v))
        cutp.add_child(gui.Label("magenta = trimmed away"))
        self._button(cutp, "Build cartilage framework (ear.obj)", self._build_framework)
        self.fw_label = gui.Label("")
        cutp.add_child(self.fw_label)
        cuts = section("Cuts drawn here (ear.obj)", parent=old)
        self.join_on = gui.Checkbox("Join ear to scalp skin (trim what lies deep to it)")
        self.join_on.checked = True
        cuts.add_child(self.join_on)
        self._button(cuts, "Rejoin / rebuild ear", lambda: self._build_framework() if self.framework is None else self._apply_cuts())
        self.draw_skin = gui.Checkbox("Draw cut on skin")
        self.draw_cart = gui.Checkbox("Draw cut on cartilage")
        self.draw_skin.set_on_checked(lambda on: self._cut_mode("skin", on))
        self.draw_cart.set_on_checked(lambda on: self._cut_mode("cartilage", on))
        cuts.add_child(self.draw_skin)
        cuts.add_child(self.draw_cart)
        self.lasso = gui.Checkbox("Lasso: drag a loop to cut (camera drag off)")
        cuts.add_child(self.lasso)
        self.sl_depth = self._slider(cuts, "depth mm", 0.0, 15.0, 0.0, lambda v: None)
        row = gui.Horiz(4)
        self._button(row, "Undo point", lambda: (self.cut_points and self.cut_points.pop(), self._draw_cut_outline()))
        self._button(row, "Clear points", lambda: (self.cut_points.clear(), self._draw_cut_outline()))
        cuts.add_child(row)
        row = gui.Horiz(4)
        self.cut_btn = self._button(row, "Cut inside outline", self._make_cut)
        self._button(row, "Undo last cut", self._undo_cut)
        cuts.add_child(row)
        self.cut_label = gui.Label(self._cuts_text())
        cuts.add_child(self.cut_label)
        mea = section("Automatic meatus (ear.obj)", parent=old)
        self.meatus_on = gui.Checkbox("automatic meatus")
        self.meatus_on.checked = self.meatus.enabled
        self.meatus_on.set_on_checked(lambda on: (setattr(self.meatus, "enabled", on), self._apply_cuts()))
        mea.add_child(self.meatus_on)
        for field, label, lo, hi in (("flare_posterior_deg", "posterior flare deg", 0, 70), ("flare_anterior_deg", "anterior flare deg", 0, 45),
                                     ("flare_len_mm", "flare length mm", 1, 15), ("widen_mm", "widen mm", -2, 4),
                                     ("drop_inferior_mm", "meatus below canal mm", -3, 8), ("drop_anterior_mm", "meatus anterior of canal mm", -4, 6),
                                     ("depth_mm", "reach mm (0 = to concha floor)", 0, 25)):
            self._slider(mea, label, lo, hi, getattr(self.meatus, field),
                         lambda v, fl=field: (setattr(self.meatus, fl, v), self._cuts_soon()))
        flat = section("Flatten lasso (CT ear)", parent=old)
        self.flat_lasso = gui.Checkbox("Flatten: drag a loop (camera drag off)")
        flat.add_child(self.flat_lasso)
        self.sl_flat_depth = self._slider(flat, "flatten depth mm (0 = all through)", 0.0, 30.0, cm.FLATTEN_DEPTH_MM, lambda v: None)
        row = gui.Horiz(4)
        self._button(row, "Undo last flatten", self._undo_flatten)
        self._button(row, "Clear flattens", self._clear_flattens)
        flat.add_child(row)

        # -- always there ----------------------------------------------------------------------
        self._button(p, "Save", self._save)
        self.status = gui.Label("")
        p.add_child(self.status)
        self.win.add_child(self.panel)

    def _on_layout(self, ctx):
        r = self.win.content_rect
        w = 25 * ctx.theme.font_size
        self.scene.frame = self.gui.Rect(r.x, r.y, r.width - w, r.height)
        self.panel.frame = self.gui.Rect(r.get_right() - w, r.y, w, r.height)

    # -- ear controls ---------------------------------------------------------------
    def _move(self, i, v):
        self.place.move_mm[i] = v
        self._update_ear()

    def _turn(self, i, v):
        self.place.turn_deg[i] = v
        self._update_ear()

    def _scale(self, v):
        self.place.scale = v
        self._update_ear()

    def _quarter(self, axis):
        self.place.base = rot(self.place.axes()[axis], 90.0) @ self.place.base
        self._update_ear()

    def _mirror(self, on):
        self.place.mirror = on
        # a mirrored transform turns triangles inside out; flip the display mesh's winding
        t = np.asarray(self.ear_display.triangles)
        self.ear_display.triangles = self.o3d.utility.Vector3iVector(t[:, [0, 2, 1]])
        self.ear_display.compute_vertex_normals()
        self.scene.scene.remove_geometry("ear")
        self._color_cut()

    # -- canal alignment (CT ear) ---------------------------------------------------------
    def _canal_marks(self):
        """Meatus and a deeper canal point (ear model mm): clicked ones first, else ct_ear.py's."""
        m = dict(self.ct_marks or {})
        m.pop("hole", None)
        if "meatus" in self.ear_marks:
            m["meatus"] = self.ear_marks["meatus"]
        if "canal" in self.ear_marks:
            m["hole"] = self.ear_marks["canal"]
        return m

    def _align_pose(self, first=False):
        import ct_ear

        m = self._canal_marks()
        if "meatus" not in m:
            return "Pick the meatus first"
        if not first:  # keep how the ear is turned about the canal now
            L = self.place.matrix()[:3, :3]
            m["superior"] = np.linalg.solve(L, self.vol.frame.superior)
        elif "superior" not in m:
            m["superior"] = np.array([0.0, 0.0, 1.0])
        lateral_mm = self.sl_meatus.double_value if hasattr(self, "sl_meatus") else 14.0
        target = ct_ear.meatus_target(self.vol, lateral_mm, self.meatus.drop_inferior_mm, self.meatus.drop_anterior_mm)
        T, info = ct_ear.register(self.vol, m, target)
        self.place.mirror = False
        self.place.scale = 1.0
        self.place.adopt(T)
        how = info["by"] + ("" if "meatus" in self.ear_marks else "; meatus is a guess: click it")
        return f"Aligned by {how}"

    def _align_ui(self):
        self.status.text = self._align_pose()
        for i, k in enumerate(("anterior", "superior", "lateral")):
            self.sl[k].double_value = self.place.move_mm[i]
        for i, k in enumerate(("turn about anterior", "turn about superior", "turn about lateral")):
            self.sl[k].double_value = 0.0
        self._update_ear()

    def _pick_mark(self, which, on):
        if on:
            other = self.pick_canal if which == "meatus" else self.pick_meatus
            other.checked = False
            self.picking.checked = False
            self.status.text = {"meatus": "Click the centre of the canal's opening in the concha",
                                "canal": "Click the canal's deepest point (where its cartilaginous part ends)"}[which]

    def _draw_marks(self):
        sc = self.scene.scene
        T = self.place.matrix()
        for name, col in (("meatus", (60, 255, 120)), ("canal", (255, 80, 80))):
            g = "mark_" + name
            if sc.has_geometry(g):
                sc.remove_geometry(g)
            if name in self.ear_marks:
                s = self.o3d.geometry.TriangleMesh.create_sphere(0.8)
                s.translate(apply(T, self.ear_marks[name][None])[0])
                s.compute_vertex_normals()
                sc.add_geometry(g, s, self._mat(col))
        set_ = [k for k in ("meatus", "canal") if k in self.ear_marks]
        self.marks_label.text = ("picked: " + ", ".join(set_)) if set_ else "none picked (using ct_ear.py's guesses)"

    def _auto_place(self):
        if self.ct_marks is not None:  # a CT ear: line its canal up instead of guessing from shape
            self._align_ui()
            return
        if getattr(self, "_marks", None) is None:
            self._marks = au.ear_landmarks(self.ear_v, self.ear_t)
        was = self.place.mirror
        self.place.auto(self._marks, self.sl_tilt.double_value, self.sl_flare.double_value, self.sl_meatus.double_value)
        now = self.place.mirror
        if now != was:  # flip the display mesh's winding to match
            self.place.mirror = was
            self.mirror.checked = now
            self._mirror(now)
        for i, k in enumerate(("anterior", "superior", "lateral")):
            self.sl[k].double_value = self.place.move_mm[i]
        for i, k in enumerate(("turn about anterior", "turn about superior", "turn about lateral")):
            self.sl[k].double_value = 0.0
        self.cut.enabled = False  # treat ear.obj as the whole ear
        self.cut_on.checked = False
        self._color_cut()
        self.status.text = "Auto-placed" + (" (ear.obj is a left ear: mirrored)" if now else "")

    def _on_key(self, e):
        gui = self.gui
        if e.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        step = {gui.KeyName.LEFT: (0, -0.5), gui.KeyName.RIGHT: (0, 0.5), gui.KeyName.UP: (1, 0.5),
                gui.KeyName.DOWN: (1, -0.5), gui.KeyName.PAGE_UP: (2, 0.5), gui.KeyName.PAGE_DOWN: (2, -0.5)}.get(e.key)
        if step is None:
            return gui.Widget.EventCallbackResult.IGNORED
        i, d = step
        self.place.move_mm[i] += d
        self.sl[("anterior", "superior", "lateral")[i]].double_value = self.place.move_mm[i]
        self._update_ear()
        return gui.Widget.EventCallbackResult.HANDLED

    def _set_cut(self, field, value):
        setattr(self.cut, field, value)
        self._color_cut()

    def _auricle_mesh(self):
        if not self.cut.enabled:
            return self.ear_v, self.ear_t, None
        r = au.cut(self.ear_v, self.ear_t, self.cut, axes=self.ear_axes)
        return r.verts, r.tris, r

    def _build_framework(self):
        av, at, r = self._auricle_mesh()
        self.auricle_base = (av, at)
        self.auricle = (av, at)
        fw = au.framework(av, at)
        self.framework = fw
        self.cartilage = (fw.verts, fw.tris)
        w = fw.thickness_mm[np.isfinite(fw.thickness_mm)]
        cut_note = "no bulge trim" if r is None else f"bulge trim: {r.loops} outline(s), {r.removed_share * 100:.0f}% of surface removed"
        self.fw_label.text = (f"{cut_note}\ncover {au.COVER_MM:.2f} mm; thinned at {fw.capped_share * 100:.1f}% of points\n"
                              f"auricle thickness median {np.median(w):.1f} mm (5-95%: {np.percentile(w, 5):.1f}-{np.percentile(w, 95):.1f})\n"
                              "skin shown translucent over the blue framework")
        self.layer_box["ear"].checked = False
        self.scene.scene.show_geometry("ear", False)
        self._apply_cuts()

    def _show_obj_mesh(self, name, v, t, rgb, alpha):
        sc = self.scene.scene
        if sc.has_geometry(name):
            sc.remove_geometry(name)
        m = self._mesh(v, t)
        if len(t) > DISPLAY_EAR_TRIS:
            m = m.simplify_quadric_decimation(DISPLAY_EAR_TRIS)
            m.compute_vertex_normals()
        sc.add_geometry(name, m, self._mat(rgb, alpha))
        sc.show_geometry(name, self.layer_box[name].checked)
        sc.set_geometry_transform(name, self.place.matrix())

    def _open_meatus(self):
        """Open the CT ear's meatus onto the canal and trim it (ct_meatus), in the background."""
        if "skin" not in self.layers:
            self.ct_label.text = "No scalp skin: rebuild the envelope first"
            return
        if getattr(self, "_ct_busy", False):  # one at a time: run again with the latest settings when done
            self._ct_again = True
            return
        self.ct_btn.enabled = False
        self._ct_busy = True
        self.ct_label.text = "Opening the meatus..."
        T = self.place.matrix()
        flip = (lambda t: t[:, [0, 2, 1]]) if np.linalg.det(T[:3, :3]) < 0 else (lambda t: t)
        ear = (apply(T, self.ear_v), flip(np.asarray(self.ear_t)))
        scalp0, thickness = self.layers["skin"], dict(self.thickness)
        others = {k: m for k, m in self.layers.items() if k != "skin"}
        flatten, muscle = self._flatten_regions(T), self.muscle
        aw, fmax = self.ear_aw, self.framework_max_mm
        cart_in = (apply(T, self.cart_edit[0]), np.asarray(self.cart_edit[1])) if self.cart_edit is not None and self.use_blender else None

        def work():
            t0 = time.time()
            new_layers = None
            try:
                if self._eac_field is None:
                    self._eac_field = cm.EacField(self.vol)
                sc_, ot_ = scalp0, others
                if aw is not None and cm._is_open(ear[1]) and self.ramp_on:
                    # a skin sheet: the envelope thins from the auricle's base, not the canal
                    import sheet_skin

                    base = sheet_skin.base_outline(*ear, self._eac_field)
                    face_moved = self._face_envelope(ear)  # the face's tissue follows the auricle's front edge
                    if face_moved or base is not None and (self.ramp.base_outline is None or len(self.ramp.base_outline) != len(base)
                                                           or not np.allclose(self.ramp.base_outline, base)):
                        self.ramp.base_outline = base
                        new_layers = self.env.layers(muscle, thickness)
                        sc_, ot_ = new_layers["skin"], {k: m for k, m in new_layers.items() if k != "skin"}
                out = cm.open_meatus(self.vol, self.env, ear, sc_, thickness, eac=self._eac_field, others=ot_,
                                     flatten=flatten, muscle=muscle, auricle=aw, framework_max_mm=fmax,
                                     cartilage=cart_in if aw is not None else None)
                err = None
            except Exception as e:  # noqa: BLE001  (shown in the panel)
                out, err = None, e

            def done():
                self.ct_btn.enabled = True
                self._ct_busy = False
                if getattr(self, "_ct_again", False):
                    self._ct_again = False
                    self._open_meatus()
                    return
                if err is not None:
                    self.ct_label.text = f"Failed: {err}"
                    return
                if new_layers is not None:
                    self.layers = new_layers
                self.ct_open, self.ct_open_T = out, T
                self.scalp_joined = out["scalp"]
                self.layers_joined = out["layers"]
                self._add_layers()
                sc = self.scene.scene
                if sc.has_geometry("auricle cartilage"):
                    sc.remove_geometry("auricle cartilage")
                if self.cart_edit is not None and self.use_blender and not out.get("cartilage_done"):  # the cartilage as edited in Blender
                    fv_ = apply(T, self.cart_edit[0])
                    if out.get("sheet") is not None:  # never poking out of the skin
                        import cartilage

                        fv_, n_ = cartilage.keep_under_skin(fv_, self.cart_edit[1], *out["ear"])
                        if n_:
                            print(f"cartilage: {n_} vertices pulled back under the skin")
                    ft_ = np.asarray(self.cart_edit[1])
                    if out.get("trim_cartilage") is not None:  # not past the auricle's base edge
                        fv_, ft_ = out["trim_cartilage"](fv_, ft_)
                    out["framework"] = (fv_, ft_)
                if out.get("framework") is not None:
                    sc.add_geometry("auricle cartilage", self._mesh(*out["framework"]), self._mat((90, 170, 255)))
                    sc.show_geometry("auricle cartilage", self.layer_box["auricle cartilage"].checked)
                for name, mesh in (("opened ear", out["ear"]), ("canal skin", out["canal"])):
                    if sc.has_geometry(name):
                        sc.remove_geometry(name)
                    sc.add_geometry(name, self._mesh(*mesh), self._mat(ev.LAYER_RGB["skin"]))
                    sc.show_geometry(name, self.layer_box[name].checked)
                for name in ("ear", "eac"):  # the opened ear and the canal skin replace them
                    self.layer_box[name].checked = False
                    sc.show_geometry(name, False)
                self.ct_label.text = (f"Opened in {time.time() - t0:.1f} s ({len(self.flattens)} flatten loop(s)): ear {len(out['ear'][1])} tris,\n"
                                      f"canal skin {len(out['canal'][1])} tris (skin over bone at the mouth {out['skin_mm']:.2f} mm)")

            self.app.post_to_main_thread(self.win, done)

        threading.Thread(target=work, daemon=True).start()

    def _flatten_event(self, e):
        """Left-drag draws a loop; letting go flattens the CT tissue inside it, along the view."""
        gui = self.gui
        x, y = e.x - self.scene.frame.x, e.y - self.scene.frame.y
        if e.type == gui.MouseEvent.Type.BUTTON_DOWN and e.is_button_down(gui.MouseButton.LEFT):
            self._lasso_px = [(x, y)]
            _eye, _d, fwd = self._view_ray(x, y)
            self._lasso_plane = (self._ear_centre(), fwd)
            self.cut_points = []
            return gui.Widget.EventCallbackResult.HANDLED
        if getattr(self, "_lasso_px", None) is None:
            return gui.Widget.EventCallbackResult.IGNORED
        if e.type == gui.MouseEvent.Type.DRAG:
            px0 = self._lasso_px[-1]
            if abs(x - px0[0]) + abs(y - px0[1]) >= 4:
                self._lasso_px.append((x, y))
                eye, d, _f = self._view_ray(x, y)
                c, n = self._lasso_plane
                self.cut_points.append((eye + d * ((c - eye) @ n) / (d @ n)).tolist())
                if len(self.cut_points) % 3 == 0:
                    self._draw_cut_outline()
            return gui.Widget.EventCallbackResult.HANDLED
        if e.type == gui.MouseEvent.Type.BUTTON_UP:
            self._lasso_px = None
            if len(self.cut_points) >= 3:
                T = self.place.matrix()
                Tinv = np.linalg.inv(T)
                _c, n = self._lasso_plane
                d_obj = Tinv[:3, :3] @ n
                self.flattens.append(ec.MeshCut("flatten", apply(Tinv, np.asarray(self.cut_points)).tolist(),
                                                (d_obj / np.linalg.norm(d_obj)).tolist(), float(self.sl_flat_depth.double_value)))
                self.status.text = f"{len(self.flattens)} flatten loop(s): trimming again..."
                self._open_meatus()
            self.cut_points = []
            self._draw_cut_outline()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.HANDLED

    def _flatten_regions(self, T):
        """The flatten loops in volume mm, for the pose T."""
        out = []
        for c in self.flattens:
            d = T[:3, :3] @ np.asarray(c.direction_obj, float)
            out.append(ec.MeshCut("flatten", apply(T, np.asarray(c.outline_obj, float)).tolist(), (d / np.linalg.norm(d)).tolist(), c.depth_mm))
        return out

    def _undo_flatten(self):
        if self.flattens and not getattr(self, "_ct_busy", False):
            self.flattens.pop()
            self.status.text = f"{len(self.flattens)} flatten loop(s): trimming again..."
            self._open_meatus()

    def _clear_flattens(self):
        if self.flattens and not getattr(self, "_ct_busy", False):
            self.flattens = []
            self.status.text = "Flatten loops cleared: trimming again..."
            self._open_meatus()

    # -- Blender round trip (blender_edit) ------------------------------------------------
    def _edited_path(self):
        p = pathlib.Path(self.ear_path)
        return p.with_name(p.stem + "_edited.ply")

    def _bundle(self):
        return be.bundle_dir(pathlib.Path(self.adf).stem, pathlib.Path(self.ear_path).stem)

    def _edit_in_blender(self):
        folder = self._bundle()
        T = self.place.matrix()
        flip = (lambda t: t[:, [0, 2, 1]]) if np.linalg.det(T[:3, :3]) < 0 else (lambda t: t)
        blend = folder / "ct_ear.blend"
        f = self.vol.frame
        info = dict(ear_to_volume_mm=T.tolist(), ear=str(self.ear_path), volume=str(self.adf),
                    lateral=np.asarray(f.lateral).tolist(), superior=np.asarray(f.superior).tolist())
        if blend.exists():  # carry on with the saved scene (its pose); only the script is refreshed
            folder.joinpath("fivrs_scene.py").write_text(be.BLENDER_SCRIPT)
            old = json.loads((folder / "bundle.json").read_text()) if (folder / "bundle.json").exists() else {}
            if old.get("ear_to_volume_mm") is not None and not np.allclose(old["ear_to_volume_mm"], T, atol=1e-3):
                self.status.text = "Blender scene is for an earlier ear pose (delete ct_ear.blend to start over)"
        else:
            if "skin" not in self.layers:
                self.status.text = "No scalp skin: rebuild the envelope first"
                return
            be.export_bundle(folder, (apply(T, self.ear_v), flip(np.asarray(self.ear_t))), self.vol.bone_mesh(),
                             self.layers["skin"], self.vol.eac_mesh(), weight=self.ear_aw, info=info)
        self._open_blender(folder, "paint the auricle ('auricle' group); Ctrl+S sends it here")

    def _open_blender(self, folder, what):
        if be.running(folder):
            self.status.text = f"Blender is already open on this ear: switch to it ({what})"
            return
        try:
            be.launch(folder)
        except FileNotFoundError as e:
            self.status.text = f"{e} (files written to {folder})"
            return
        self.status.text = f"Opening Blender: {what}"

    def _edit_cartilage_in_blender(self):
        """Hand the cartilage to Blender and open it there (the skin scene, cartilage selected)."""
        folder = self._bundle()
        if not (folder / "ct_ear.blend").exists():
            self.status.text = "Edit the skin in Blender first (paint the auricle): the cartilage is built from it"
            return
        if self.ct_open is None or self.ct_open.get("framework") is None:
            self.status.text = "No cartilage yet: press Build ear (with the Blender skin) first"
            return
        self._cartilage_to_blender()
        (folder / "load_cartilage.request").write_text("")
        if be.running(folder):
            self.status.text = "Blender is open: FIVRS tab -> Show cartilage (Ctrl+S sends it here)"
            return
        folder.joinpath("fivrs_scene.py").write_text(be.BLENDER_SCRIPT)
        self._open_blender(folder, "cut and model the cartilage; Ctrl+S sends it here")

    def _watch_blender(self):
        """Every 2 s: did Blender save new edits for this ear? Then load them."""
        while True:
            time.sleep(2.0)
            try:
                folder = self._bundle()
                stamps = [p.stat().st_mtime for p in (folder / be.EDITED, folder / be.CARTILAGE_EDITED) if p.exists()]
            except OSError:
                continue
            if stamps and max(stamps) > self._blender_seen + 0.5 and not getattr(self, "_ct_busy", False):
                self._blender_seen = max(stamps)
                time.sleep(1.0)  # let Blender finish writing
                self.app.post_to_main_thread(self.win, self._load_blender_edit)

    def _load_blender_edit(self):
        folder = self._bundle()
        got = be.read_edit(folder)
        if got is None:
            self.status.text = "Nothing saved from Blender yet (Ctrl+S there)"
            return
        self._blender_seen = max([p.stat().st_mtime for p in (folder / be.EDITED, folder / be.CARTILAGE_EDITED) if p.exists()] or [0.0])
        v, t, w = got
        T = np.asarray(json.loads((folder / "bundle.json").read_text())["ear_to_volume_mm"], float)
        if np.linalg.det(T[:3, :3]) < 0:
            t = t[:, [0, 2, 1]]
        v = apply(np.linalg.inv(T), v)  # back to the ear's own frame: the placement still applies
        be.write_ply(self._edited_path(), v, t, w)
        self.use_blender = self.blender_box.checked = True
        cart = be.read_cartilage_edit(folder)
        if cart is not None:
            cv_, ct_ = cart
            if np.linalg.det(T[:3, :3]) < 0:
                ct_ = ct_[:, [0, 2, 1]]
            cv_ = apply(np.linalg.inv(T), cv_)
            be.write_ply(self._cart_edited_path(), cv_, ct_)
            self.cart_edit = (cv_, ct_)
        self._set_ear(v, t, w if w is not None else np.zeros(len(v)))
        self.status.text = (f"Blender edit loaded: {len(t)} tris, {int((self.ear_aw > 0.5).sum())} auricle vertices"
                            + (f", cartilage as edited ({len(self.cart_edit[1])} tris)" if self.cart_edit is not None else ""))

    def _cart_edited_path(self):
        p = pathlib.Path(self.ear_path)
        return p.with_name(p.stem + "_cartilage_edited.ply")

    def _cartilage_to_blender(self):
        """Write the cartilage shown now into the Blender bundle (Load cartilage there picks it up)."""
        if self.ct_open is None or self.ct_open.get("framework") is None:
            self.status.text = "No cartilage yet: edit the skin in Blender (paint the auricle), then Build ear"
            return
        folder = self._bundle()
        info = json.loads((folder / "bundle.json").read_text()) if (folder / "bundle.json").exists() else {}
        T = self.place.matrix()
        if info.get("ear_to_volume_mm") is not None and not np.allclose(info["ear_to_volume_mm"], T, atol=1e-3):
            self.status.text = "The ear moved since the Blender scene was made: the cartilage would not line up"
            return
        v, t = self.ct_open["framework"]
        import open3d as o3d

        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(v)), o3d.utility.Vector3iVector(np.asarray(t)))
        if len(t) > 200_000:  # lighter to edit
            m = m.simplify_quadric_decimation(200_000)
        be.write_ply(folder / be.CARTILAGE, np.asarray(m.vertices), np.asarray(m.triangles))
        self.status.text = f"Cartilage written for Blender ({len(m.triangles)} tris)"

    def _drop_cartilage_edit(self):
        self.cart_edit = None
        if self._cart_edited_path().exists():
            self._cart_edited_path().rename(self._cart_edited_path().with_suffix(".ply.bak"))
        self.status.text = "Cartilage edit set aside (.ply.bak): rebuilding from the sheet"
        if self.ct_open is not None:
            self._open_meatus()

    def _toggle_blender(self, on):
        self.use_blender = bool(on)
        if on and self._edited_path().exists():
            self._set_ear(*be.read_ply(self._edited_path()))
        elif not on:
            self._set_ear(*load_ear(self.ear_path), None)

    def _face_envelope(self, ear=None):
        """With the Blender skin sheet: the envelope rebuilt with the face's tissue in front of the
        ear, coming up to the auricle's front edge (``envelope.Envelope(face=...)``). ``ear``: the
        sheet placed in the volume (default: as placed now). True when it changed."""
        import sheet_skin

        if self.ear_aw is None or not cm._is_open(self.ear_t):
            return False
        if ear is None:
            T = self.place.matrix()
            flip = (lambda t: t[:, [0, 2, 1]]) if np.linalg.det(T[:3, :3]) < 0 else (lambda t: t)
            ear = (apply(T, self.ear_v), flip(np.asarray(self.ear_t)))
        if self._eac_field is None:
            self._eac_field = cm.EacField(self.vol)
        face = sheet_skin.front_edge(*ear, self.vol.frame, self._eac_field)
        key = None if face is None else np.round(face[0], 2)
        old = getattr(self, "_face_key", None)
        if (key is None and old is None) or (key is not None and old is not None and key.shape == old.shape and np.allclose(key, old)):
            return False
        ramp = self.env.auricle_ramp
        self.env = ev.Envelope(self.vol, face=face)
        self.env.auricle_ramp = ramp
        self._face_key = key
        print(f"envelope: rebuilt {'with the face in front of the ear' if face is not None else 'without the face'}")
        return True

    def _set_ear(self, v, t, w):
        self.ear_v, self.ear_t, self.ear_aw = np.asarray(v, float), np.asarray(t, np.int32), w
        sc = self.scene.scene
        if sc.has_geometry("ear"):
            sc.remove_geometry("ear")
        self._add_ear()
        sc.set_geometry_transform("ear", self.place.matrix())
        sc.show_geometry("ear", self.layer_box["ear"].checked)
        if self.ct_open is not None:
            self._open_meatus()

    def _revert_open(self):
        if getattr(self, "_ct_busy", False):
            return
        self._clear_open("Back to the untrimmed ear (Save keeps it that way)" if self.ct_open is not None else "")

    def _clear_open(self, why=""):
        self.ct_open = self.ct_open_T = None
        self.scalp_joined = None
        self.layers_joined = {}
        sc = self.scene.scene
        for name in ("opened ear", "canal skin", "auricle cartilage"):
            if sc.has_geometry(name):
                sc.remove_geometry(name)
        for name in ("ear", "eac"):
            self.layer_box[name].checked = True
            sc.show_geometry(name, True)
        self._add_layers()
        self.ct_label.text = why

    def _apply_cuts(self):
        """Rebuild skin and cartilage from their bases with all cuts (in the background)."""
        if self.framework is None:
            return
        skin_cuts = [c for c in self.cuts if c.target == "skin"]
        cart_cuts = [c for c in self.cuts if c.target == "cartilage"]
        self.cut_btn.enabled = False
        self.cut_label.text = "Cutting..." if self.cuts else self._cuts_text()

        join = self.join_on.checked and "skin" in self.layers
        T = self.place.matrix()
        scalp0 = self.layers.get("skin")

        def work():
            out = ec.assemble(self.vol, self.env, self.auricle_base, (self.framework.verts, self.framework.tris),
                              self.cuts, T, scalp0, join=join, make_funnel=join, meatus=self.meatus)
            skin, cart, scalp, tube = out["skin"], out["cartilage"], (out["scalp"] if join else None), out["funnel"]
            sleeve_m = out["sleeve"]

            def done():
                self.auricle, self.cartilage, self.scalp_joined, self.funnel = skin, cart, scalp, tube
                sc = self.scene.scene
                self.sleeve = sleeve_m
                for name in ("funnel", "canal cartilage"):
                    if sc.has_geometry(name):
                        sc.remove_geometry(name)
                if sleeve_m is not None:
                    sc.add_geometry("canal cartilage", self._mesh(*sleeve_m), self._mat((90, 170, 255)))
                    sc.show_geometry("canal cartilage", self.layer_box["canal cartilage"].checked)
                if tube is not None and len(tube[1]):
                    sc.add_geometry("funnel", self._mesh(*tube), self._mat(ev.LAYER_RGB["skin"], self.alpha.double_value))
                    sc.show_geometry("funnel", self.layer_box["funnel"].checked)
                self._show_obj_mesh("auricle", *skin, ev.LAYER_RGB["skin"], self.alpha.double_value)
                self._show_obj_mesh("framework", *cart, (90, 170, 255), 1.0)
                self._add_layers()
                self.cut_btn.enabled = True
                self.cut_label.text = self._cuts_text() + ("\njoined to the scalp skin (rejoin after moving the ear)" if join else "")
                self._update_ear()

            self.app.post_to_main_thread(self.win, done)

        threading.Thread(target=work, daemon=True).start()

    def _cuts_soon(self, delay=0.8):
        """Rebuild the ear a moment after a meatus slider settles."""
        if self.framework is None:
            return
        if getattr(self, "_ctimer", None) is not None:
            self._ctimer.cancel()
        self._ctimer = threading.Timer(delay, lambda: self.app.post_to_main_thread(self.win, self._apply_cuts))
        self._ctimer.daemon = True
        self._ctimer.start()

    def _cuts_text(self):
        n_s = sum(c.target == "skin" for c in self.cuts)
        n_c = sum(c.target == "cartilage" for c in self.cuts)
        return f"{n_s} skin cut(s), {n_c} cartilage cut(s)"

    def _cut_mode(self, target, on):
        if not on:
            return
        other = self.draw_cart if target == "skin" else self.draw_skin
        other.checked = False
        self.picking.checked = False
        if self.framework is None:
            self._build_framework()
        self.status.text = f"Click around the area on the {target}; then Cut"

    def _make_cut(self):
        target = "skin" if self.draw_skin.checked else "cartilage" if self.draw_cart.checked else None
        if target is None or len(self.cut_points) < 3:
            self.status.text = "Tick skin or cartilage and click 3+ points first"
            return
        T = self.place.matrix()
        Tinv = np.linalg.inv(T)
        V = np.asarray(self.scene.scene.camera.get_view_matrix())
        forward = -V[2, :3]  # camera looks along -z
        d_obj = Tinv[:3, :3] @ forward
        d_obj /= np.linalg.norm(d_obj)
        outline = apply(Tinv, np.asarray(self.cut_points))
        self.cuts.append(ec.MeshCut(target, outline.tolist(), d_obj.tolist(), float(self.sl_depth.double_value)))
        self.cut_points = []
        self._draw_cut_outline()
        self._apply_cuts()

    def _undo_cut(self):
        if self.cuts:
            self.cuts.pop()
            self._apply_cuts()

    # -- lasso ----------------------------------------------------------------------------
    def _view_ray(self, x, y):
        cam = self.scene.scene.camera
        V = np.asarray(cam.get_view_matrix())
        eye = -V[:3, :3].T @ V[:3, 3]
        through = np.asarray(cam.unproject(x, y, 0.5, self.scene.frame.width, self.scene.frame.height), float)
        d = through - eye
        return eye, d / np.linalg.norm(d), -V[2, :3]

    def _lasso_event(self, e):
        """Left-drag draws a loop on screen; letting go cuts everything inside it along the view."""
        gui = self.gui
        x, y = e.x - self.scene.frame.x, e.y - self.scene.frame.y
        if e.type == gui.MouseEvent.Type.BUTTON_DOWN and e.is_button_down(gui.MouseButton.LEFT):
            self._lasso_px = [(x, y)]
            # the loop is drawn on the plane across the view through the ear's centre
            eye, _d, fwd = self._view_ray(x, y)
            self._lasso_plane = (self._ear_centre(), fwd)
            self.cut_points = []
            return gui.Widget.EventCallbackResult.HANDLED
        if getattr(self, "_lasso_px", None) is None:
            return gui.Widget.EventCallbackResult.IGNORED
        if e.type == gui.MouseEvent.Type.DRAG:
            px0 = self._lasso_px[-1]
            if abs(x - px0[0]) + abs(y - px0[1]) >= 4:
                self._lasso_px.append((x, y))
                eye, d, _f = self._view_ray(x, y)
                c, n = self._lasso_plane
                self.cut_points.append((eye + d * ((c - eye) @ n) / (d @ n)).tolist())
                if len(self.cut_points) % 3 == 0:
                    self._draw_cut_outline()
            return gui.Widget.EventCallbackResult.HANDLED
        if e.type == gui.MouseEvent.Type.BUTTON_UP:
            self._lasso_px = None
            if len(self.cut_points) >= 3:
                pts = np.asarray(self.cut_points)
                c, n = self._lasso_plane
                if self.sl_depth.double_value > 0:  # a depth cut starts at the surface under the loop
                    target = "skin" if self.draw_skin.checked else "cartilage"
                    centre = pts.mean(axis=0)
                    ray = self.o3d.core.Tensor(np.concatenate([centre - n * 200, n])[None].astype(np.float32))
                    t = float(self._target_pick_scene(target).cast_rays(ray)["t_hit"].numpy()[0])
                    if np.isfinite(t):
                        pts = pts + n * ((centre - n * 200 + n * t - c) @ n)
                self.cut_points = pts.tolist()
                self._draw_cut_outline()
                self._make_cut()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.HANDLED

    def _target_pick_scene(self, target):
        """Ray scene of the skin or cartilage as placed now."""
        v, t = self.auricle if target == "skin" else self.cartilage
        sc = self.o3d.t.geometry.RaycastingScene()
        sc.add_triangles(self.o3d.core.Tensor(apply(self.place.matrix(), v).astype(np.float32)),
                         self.o3d.core.Tensor(np.asarray(t, np.uint32)))
        return sc

    def _draw_cut_outline(self):
        sc = self.scene.scene
        for name in getattr(self, "_cut_names", []):
            sc.remove_geometry(name)
        self._cut_names = []
        pts = np.asarray(self.cut_points, float).reshape(-1, 3)
        for i, p in enumerate(pts):
            s = self.o3d.geometry.TriangleMesh.create_sphere(0.4)
            s.translate(p)
            s.compute_vertex_normals()
            sc.add_geometry(f"cutpt{i}", s, self._mat((255, 220, 40)))
            self._cut_names.append(f"cutpt{i}")
        if len(pts) >= 2:
            ls = self.o3d.geometry.LineSet(self.o3d.utility.Vector3dVector(pts),
                                            self.o3d.utility.Vector2iVector([[i, (i + 1) % len(pts)] for i in range(len(pts))]))
            m = self.rendering.MaterialRecord()
            m.shader = "unlitLine"
            m.line_width = 3.0
            m.base_color = [1.0, 0.85, 0.15, 1.0]
            sc.add_geometry("cut_outline", ls, m)
            self._cut_names.append("cut_outline")

    # -- muscle picking -------------------------------------------------------------
    def _ear_centre(self):
        return apply(self.place.matrix(), self.place.pivot[None])[0]

    def _focus_ear(self):
        """Point the camera at the ear's centre so mouse orbiting turns around it."""
        cam = self.scene.scene.camera
        V = np.asarray(cam.get_view_matrix())
        eye = -V[:3, :3].T @ V[:3, 3]  # camera position in world
        c = self._ear_centre()
        dist = max(np.linalg.norm(eye - c), 60.0)
        self.scene.look_at(c, c + (eye - c) / max(np.linalg.norm(eye - c), 1e-9) * dist, V[1, :3])

    def _view(self):
        """Camera right, up, forward and eye position in volume mm."""
        V = np.asarray(self.scene.scene.camera.get_view_matrix())
        return V[0, :3], V[1, :3], -V[2, :3], -V[:3, :3].T @ V[:3, 3]

    def _shift_ear(self, d):
        """Move the ear by d (volume mm), keeping the sliders in step."""
        self.place.move_mm += np.array([d @ ax for ax in self.place.axes()])
        for i, k in enumerate(("anterior", "superior", "lateral")):
            self.sl[k].double_value = self.place.move_mm[i]
        self._update_ear()

    def _drag_ear(self, e):
        """A drag that started on the ear: slide it in the view plane, or turn it about its centre."""
        dx, dy = e.x - self._drag_from[0], e.y - self._drag_from[1]
        self._drag_from = (e.x, e.y)
        right, up, fwd, eye = self._view()
        if self._ear_drag == "slide":
            depth = max((self._ear_centre() - eye) @ fwd, 1.0)
            fov = np.radians(self.scene.scene.camera.get_field_of_view())
            mm_per_px = 2 * depth * np.tan(fov / 2) / max(self.scene.frame.height, 1)
            self._shift_ear((dx * right - dy * up) * mm_per_px)
            return
        R = rot(up, 0.4 * dx) @ rot(right, 0.4 * dy)
        # fold the turn into the base orientation so the slider angles stay as they are
        U = self.place.user_rotation()
        self.place.base = U.T @ R @ U @ self.place.base
        self._update_ear()

    def _on_mouse(self, e):
        gui = self.gui
        T_ = gui.MouseEvent.Type
        # a drag on the ear is decided when the button goes down and owns every event until it comes
        # up, so the camera never sees half of it (Open3D's own Shift+drag is a zoom)
        if e.type == T_.BUTTON_DOWN and e.is_button_down(gui.MouseButton.LEFT):
            mode = None
            if e.is_modifier_down(gui.KeyModifier.CTRL):
                mode = "slide"
            elif e.is_modifier_down(gui.KeyModifier.SHIFT):
                mode = "turn"
            elif self.drag_mode.selected_index == 1:
                mode = "slide"
            elif self.drag_mode.selected_index == 2:
                mode = "turn"
            if mode is not None:
                self._ear_drag, self._drag_from = mode, (e.x, e.y)
                return gui.Widget.EventCallbackResult.HANDLED
        if getattr(self, "_ear_drag", None) is not None:
            if e.type == T_.DRAG:
                self._drag_ear(e)
            elif e.type == T_.BUTTON_UP:
                self._ear_drag = self._drag_from = None
            return gui.Widget.EventCallbackResult.HANDLED
        if e.type == T_.WHEEL and (self.drag_mode.selected_index == 1 or e.is_modifier_down(gui.KeyModifier.CTRL)):
            _r, _u, fwd, _eye = self._view()
            self._shift_ear(fwd * 0.5 * np.sign(e.wheel_dy))
            return gui.Widget.EventCallbackResult.HANDLED
        if self.flat_lasso.checked:
            return self._flatten_event(e)
        cutting = self.draw_skin.checked or self.draw_cart.checked
        if cutting and self.lasso.checked and self.framework is not None:
            return self._lasso_event(e)
        marking = self.pick_meatus.checked or self.pick_canal.checked
        if marking:
            if e.type == gui.MouseEvent.Type.BUTTON_DOWN and e.is_button_down(gui.MouseButton.LEFT):
                self._click_from = (e.x, e.y)
            elif e.type == gui.MouseEvent.Type.BUTTON_UP and getattr(self, "_click_from", None) is not None:
                x0, y0 = self._click_from
                self._click_from = None
                if abs(e.x - x0) + abs(e.y - y0) <= 4:
                    T = self.place.matrix()
                    sc = self.o3d.t.geometry.RaycastingScene()
                    sc.add_triangles(self.o3d.core.Tensor(apply(T, np.asarray(self.ear_display.vertices)).astype(np.float32)),
                                     self.o3d.core.Tensor(np.asarray(self.ear_display.triangles, np.uint32)))
                    p = self._pick(e.x - self.scene.frame.x, e.y - self.scene.frame.y, sc)
                    if p is not None:
                        which = "meatus" if self.pick_meatus.checked else "canal"
                        self.ear_marks[which] = apply(np.linalg.inv(T), p[None])[0]
                        self._draw_marks()
                    return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED
        if not (self.picking.checked or cutting):
            return gui.Widget.EventCallbackResult.IGNORED
        # a click (press and release without moving) adds an outline point; a drag still turns the camera
        if e.type == gui.MouseEvent.Type.BUTTON_DOWN and e.is_button_down(gui.MouseButton.LEFT):
            self._click_from = (e.x, e.y)
        elif e.type == gui.MouseEvent.Type.BUTTON_UP and getattr(self, "_click_from", None) is not None:
            x0, y0 = self._click_from
            self._click_from = None
            if abs(e.x - x0) + abs(e.y - y0) <= 4:
                scene = None
                if cutting and self.framework is not None:
                    scene = self._target_pick_scene("skin" if self.draw_skin.checked else "cartilage")
                p = self._pick(e.x - self.scene.frame.x, e.y - self.scene.frame.y, scene)
                if p is None:
                    self.status.text = "Nothing under that click"
                elif cutting:
                    self.cut_points.append([float(c) for c in p])
                    self._draw_cut_outline()
                    self.status.text = f"{len(self.cut_points)} cut points"
                else:
                    self._add_point(p)
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _pick(self, x, y, scene=None):
        """First hit under widget pixel (x, y) on the bone (or the given scene), or None."""
        cam = self.scene.scene.camera
        w, h = self.scene.frame.width, self.scene.frame.height
        V = np.asarray(cam.get_view_matrix())
        eye = -V[:3, :3].T @ V[:3, 3]
        through = np.asarray(cam.unproject(x, y, 0.5, w, h), float)
        d = through - eye
        d /= max(np.linalg.norm(d), 1e-12)
        ray = self.o3d.core.Tensor(np.concatenate([eye, d])[None].astype(np.float32))
        t = float((scene or self.pick_scene).cast_rays(ray)["t_hit"].numpy()[0])
        return None if not np.isfinite(t) else eye + d * t

    # -- layer solids ------------------------------------------------------------------
    def _solid_names(self):
        return [n for n in getattr(self, "_solids", {})]

    def _show_solids(self, on):
        sc = self.scene.scene
        for name in ["bone", *ev.LAYER_NAMES]:
            if sc.has_geometry(name):
                sc.show_geometry(name, (not on) and self.layer_box[name].checked)
        for name in self._solid_names():
            sc.show_geometry("solid_" + name, on)
        if on and not self._solid_names():
            self._build_solids()

    def _rebuild_solids_soon(self):
        if not self.solids_on.checked:
            return
        if getattr(self, "_stimer", None) is not None:
            self._stimer.cancel()
        self._stimer = threading.Timer(0.6, lambda: self.app.post_to_main_thread(self.win, self._build_solids))
        self._stimer.daemon = True
        self._stimer.start()

    def _build_solids(self):
        if getattr(self, "_solid_busy", False):
            self._solid_again = True
            return
        self._solid_busy, self._solid_again = True, False
        f = self.vol.frame
        axis = {"anterior": f.anterior, "superior": f.superior, "lateral": f.lateral}[self.solid_axis]
        cut = (f.canal_center + axis * self.sl_solid.double_value, axis) if self.cut_on_solids.checked else None
        self.solid_label.text = "Building layer solids (about 10 s)..."
        self.solid_btn.enabled = False

        def work():
            solids = self.env.layer_solids(self.muscle, self.thickness, cut)
            keep = (self.ct_open or {}).get("skin_keep")
            if keep is not None and "skin" in solids:  # the auricle's skin, not the envelope's, under the auricle
                sv_, st_ = solids["skin"]
                solids["skin"] = au.clip_mesh(sv_, st_, keep(np.asarray(sv_)))

            def done():
                sc = self.scene.scene
                for name in self._solid_names():
                    if sc.has_geometry("solid_" + name):
                        sc.remove_geometry("solid_" + name)
                self._solids = solids
                for name, (v, t) in solids.items():
                    rgb = (242, 238, 226) if name == "bone" else ev.LAYER_RGB[name]
                    sc.add_geometry("solid_" + name, self._mesh(v, t), self._mat(rgb))
                    sc.show_geometry("solid_" + name, self.solids_on.checked)
                self.solids_on.checked = True
                self._show_solids(True)
                where = f"cut across {self.solid_axis} at {self.sl_solid.double_value:+.0f} mm" if cut else "no cutaway"
                self.solid_label.text = f"{where}\nthin layers drawn at least {ev.MIN_SOLID_MM} mm thick (display only)"
                self.solid_btn.enabled = True
                self._solid_busy = False
                if self._solid_again:
                    self._build_solids()

            self.app.post_to_main_thread(self.win, done)

        threading.Thread(target=work, daemon=True).start()

    def _ramp_toggled(self, on):
        self.ramp_on = bool(on)
        self.env.auricle_ramp = self.ramp if self.ramp_on else None
        self._rebuild_soon(0.0)

    def _rebuild_soon(self, delay=0.6):
        """Rebuild once the slider has been still for a moment."""
        if getattr(self, "_timer", None) is not None:
            self._timer.cancel()
        self._timer = threading.Timer(delay, lambda: self.app.post_to_main_thread(self.win, self._rebuild))
        self._timer.daemon = True
        self._timer.start()

    def _add_point(self, p):
        self.muscle.outline_mm.append([float(c) for c in p])
        self._draw_outline()
        self.status.text = f"{len(self.muscle.outline_mm)} outline points"
        if len(self.muscle.outline_mm) >= 3:
            self._rebuild_soon()

    def _undo_point(self):
        if self.muscle.outline_mm:
            self.muscle.outline_mm.pop()
        self._draw_outline()
        self._rebuild_soon()

    def _clear_points(self):
        self.muscle.outline_mm = []
        self._draw_outline()
        self._rebuild_soon()

    # -- rebuild / save -------------------------------------------------------------
    def _rebuild(self):
        if getattr(self, "_busy", False):
            self._pending = True
            return
        self._busy, self._pending = True, False
        self.status.text = "Rebuilding envelope..."
        self.rebuild_btn.enabled = False

        def work():
            t0 = time.time()
            layers = self.env.layers(self.muscle, self.thickness)

            def done():
                self.layers = layers
                self.scalp_joined = None
                self.layers_joined = {}
                self._add_layers()
                self._update_ear()
                if self.framework is not None and self.join_on.checked:
                    self._apply_cuts()  # rejoin the ear to the new scalp skin
                if self.ct_open is not None:
                    self._open_meatus()  # open and trim against the new scalp skin
                self.rebuild_btn.enabled = True
                self.status.text = f"Envelope rebuilt in {time.time() - t0:.1f} s"
                self._busy = False
                if self._pending:
                    self._rebuild()

            self.app.post_to_main_thread(self.win, done)

        threading.Thread(target=work, daemon=True).start()

    def _save(self):
        path = ev.placement_path(self.adf, self.ear_path)
        folder = path.parent / path.name.removesuffix("_placement.json")  # ear.obj: <volume>/, others: <volume>_<ear>/
        T = self.place.matrix()
        W = self.vol.mm_to_world_matrix()
        doc = dict(
            volume=str(self.adf.relative_to(ev.ROOT)) if self.adf.is_relative_to(ev.ROOT) else str(self.adf),
            ear_obj=str(self.ear_path),
            ear=self.place.to_dict() | dict(ear_to_volume_mm=T.tolist(), ear_to_world_au=(W @ T).tolist()),
            volume_mm_to_world_au=W.tolist(),
            frame=dict(anterior=self.vol.frame.anterior.tolist(), superior=self.vol.frame.superior.tolist(),
                       lateral=self.vol.frame.lateral.tolist(), canal_lateral_end_mm=self.vol.frame.canal_lateral_end.tolist()),
            muscle=dict(outline_mm=self.muscle.outline_mm, max_mm=self.muscle.max_mm, taper_mm=self.muscle.taper_mm),
            thickness_mm=self.thickness,
            auricle_ramp=dict(on=self.ramp_on, near_mm=self.ramp.near_mm, far_mm=self.ramp.far_mm, near=dict(self.ramp.near),
                              base_near_mm=self.ramp.base_near_mm, base_far_mm=self.ramp.base_far_mm,
                              sources={n: src for n, (_t, src) in ev.AURICLE_NEAR.items()}),
            cut=dict(vars(self.cut)),
            mesh_cuts=[dict(vars(c)) for c in self.cuts],
            ct_flatten=[dict(vars(c)) for c in self.flattens],
            blender=dict(use=self.use_blender, framework_max_mm=self.framework_max_mm,
                         edited=str(self._edited_path()) if self.ear_aw is not None else None),
            ear_marks={k: np.asarray(x).tolist() for k, x in self.ear_marks.items()},
            meatus=dict(vars(self.meatus)),
            ct_meatus=dict(opened=self.ct_open is not None, skin_mm=self.ct_open["skin_mm"] if self.ct_open else None,
                           blend_mm=cm.BLEND_MM, flare_len_mm=cm.FLARE_LEN_MM, flare_tan=cm.FLARE_TAN, lining_mm=cm.LINING_MM,
                           mouth_mm=cm.MOUTH_MM, meshes=["ct_ear_opened.ply", "canal_skin.ply"] if self.ct_open else []),
            layer_sources={n: s for n, _t, s in ev.LAYERS},
            framework_cover_mm=au.COVER_MM,
            saved=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n")
        for name, (v, t) in self.layers.items():
            if name == "skin" and getattr(self, "scalp_joined", None) is not None:
                v, t = self.scalp_joined
            v, t = self.layers_joined.get(name, (v, t))
            ev.write_ply(folder / f"{name}.ply", v, t, ev.LAYER_RGB[name])
        flip = (lambda t: t[:, [0, 2, 1]]) if np.linalg.det(T[:3, :3]) < 0 else (lambda t: t)
        ev.write_ply(folder / "ear_obj_placed.ply", apply(T, self.ear_v), flip(self.ear_t), (228, 170, 150))
        av, at, _r = self._auricle_mesh() if self.auricle is None else (*self.auricle, None)
        ev.write_ply(folder / "auricle_placed.ply", apply(T, av), flip(at), (228, 170, 150))
        if self.funnel is not None:
            ev.write_ply(folder / "eac_funnel.ply", *self.funnel, ev.LAYER_RGB["skin"])
        if self.sleeve is not None:
            ev.write_ply(folder / "canal_cartilage.ply", *self.sleeve, (90, 170, 255))
        if self.ct_open is not None:
            ev.write_ply(folder / "ct_ear_opened.ply", *self.ct_open["ear"], ev.LAYER_RGB["skin"])
            ev.write_ply(folder / "canal_skin.ply", *self.ct_open["canal"], ev.LAYER_RGB["skin"])
            if self.ct_open.get("framework") is not None:
                ev.write_ply(folder / "auricle_cartilage.ply", *self.ct_open["framework"], (90, 170, 255))
        if self.cartilage is not None:
            ev.write_ply(folder / "framework_placed.ply", apply(T, self.cartilage[0]), flip(self.cartilage[1]), (150, 190, 225))
        self.status.text = f"Saved {path.name} and meshes in {folder.relative_to(ev.ROOT)}"
        print(self.status.text)

    # -- snapshot -------------------------------------------------------------------
    def _snap_later(self):
        time.sleep(3.0)
        for _ in range(600):  # let a meatus opening started at launch finish first
            if not getattr(self, "_ct_busy", False):
                break
            time.sleep(0.2)
        time.sleep(1.0)

        def shoot():
            def got(img):
                self.o3d.io.write_image(str(self.snapshot), img)
                print(f"wrote {self.snapshot}")
                self.app.quit()

            self.scene.scene.scene.render_to_image(got)

        self.app.post_to_main_thread(self.win, shoot)

    def run(self):
        self.app.run()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adf", nargs="?", default=str(ev.ROOT / "ADF" / "volume_256_RT143.yaml"))
    ap.add_argument("--ear", default=str(DEFAULT_EAR))
    ap.add_argument("--snapshot", help="render once to this PNG and exit")
    ap.add_argument("--web", action="store_true", help="stream the window to a browser (WebRTC, port 8888)")
    ap.add_argument("--port", type=int, default=8888, help="browser port with --web")
    ap.add_argument("--open-meatus", action="store_true", help="open the CT ear's meatus and trim it at launch")
    ap.add_argument("--size", default=None, help="window size WxH (default 1700x1050, or 1180x760 with --web)")
    args = ap.parse_args()
    if args.web:
        import os

        import open3d as o3d

        os.environ["WEBRTC_IP"] = "localhost"
        os.environ["WEBRTC_PORT"] = str(args.port)
        o3d.visualization.webrtc_server.enable_webrtc()
        print(f"Open http://localhost:{args.port} in a browser", flush=True)
    size = args.size or ("1180x760" if args.web else "1700x1050")
    App(pathlib.Path(args.adf).resolve(), pathlib.Path(args.ear), args.snapshot, tuple(int(x) for x in size.lower().split("x")),
        open_meatus=args.open_meatus).run()


if __name__ == "__main__":
    main()
