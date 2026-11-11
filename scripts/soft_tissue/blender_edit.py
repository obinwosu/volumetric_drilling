"""Edit the CT ear in Blender: mark the auricle, flatten or delete what is not auricle.

The placement viewer's own drawing tools fight Open3D's camera. Blender is a real editor, so
the viewer hands the ear over and reads it back:

1. ``export_bundle`` writes a folder (``bundle_dir``) with the placed CT ear, bone, the
   envelope's skin and the EAC (volume mm, 1 Blender unit = 1 mm) and ``fivrs_scene.py``.
2. ``launch`` starts Blender on it. The script builds the scene the first time (references
   locked, the ear in Weight Paint with an empty "auricle" group) and saves ``ct_ear.blend``;
   later launches open that file, so the work carries on.
3. In Blender: paint the auricle into the "auricle" vertex group (weight 1 = auricle), cut the
   rest away; model the cartilage (object 'cartilage', from ``cartilage.ply``). Every save
   (Ctrl+S) writes ``ct_ear_edited.ply`` (the ear, triangulated, with each vertex's auricle
   weight) and ``cartilage_edited.ply`` (modifiers applied).
4. The viewer notices those files and loads them (``read_edit``, ``read_cartilage_edit``).

PLY here: binary little endian, float x y z, float ``auricle`` per vertex, triangles.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLES = ROOT / "resources" / "soft_tissue" / "blender"
EDITED = "ct_ear_edited.ply"
CARTILAGE = "cartilage.ply"  # the framework built from the sheet, for editing in Blender
CARTILAGE_EDITED = "cartilage_edited.ply"  # ... as edited (Save cartilage for FIVRS)


def bundle_dir(volume_name, ear_name="ct_right_ear"):
    return BUNDLES / f"{volume_name}_{ear_name}"


def write_ply(path, verts, tris, weight=None):
    """Binary PLY with an optional per-vertex float ``auricle``."""
    v = np.asarray(verts, np.float32)
    t = np.asarray(tris, np.int32)
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(v)}",
            "property float x", "property float y", "property float z"]
    if weight is not None:
        head.append("property float auricle")
    head += [f"element face {len(t)}", "property list uchar int vertex_indices", "end_header"]
    vdt = [("p", "<f4", 3)] + ([("w", "<f4")] if weight is not None else [])
    va = np.zeros(len(v), dtype=vdt)
    va["p"] = v
    if weight is not None:
        va["w"] = np.asarray(weight, np.float32)
    fa = np.zeros(len(t), dtype=[("n", "u1"), ("i", "<i4", 3)])
    fa["n"] = 3
    fa["i"] = t
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode())
        f.write(va.tobytes())
        f.write(fa.tobytes())


def read_ply(path):
    """(verts, tris, auricle weight or None) from ``write_ply`` (or Blender's ``Save for FIVRS``)."""
    data = pathlib.Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode().splitlines()
    nv = nf = 0
    props = []
    cur = None
    for line in head:
        w = line.split()
        if w[:2] == ["element", "vertex"]:
            nv, cur = int(w[2]), "v"
        elif w[:2] == ["element", "face"]:
            nf, cur = int(w[2]), "f"
        elif w[0] == "property" and cur == "v":
            props.append(w[2])
    vdt = [(p, "<f4") for p in props]
    va = np.frombuffer(data, dtype=vdt, count=nv, offset=end)
    fa = np.frombuffer(data, dtype=[("n", "u1"), ("i", "<i4", 3)], count=nf, offset=end + va.nbytes)
    verts = np.stack([va["x"], va["y"], va["z"]], axis=1).astype(np.float64)
    weight = va["auricle"].astype(np.float64) if "auricle" in props else None
    return verts, fa["i"].astype(np.int32), weight


def export_bundle(folder, ear, bone, skin, eac, weight=None, info=None):
    """Write the meshes (volume mm) and the Blender script. ``ear`` keeps any earlier paint."""
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_ply(folder / "ct_ear.ply", *ear, weight)
    write_ply(folder / "bone.ply", *bone)
    write_ply(folder / "envelope_skin.ply", *skin)
    write_ply(folder / "eac.ply", *eac)
    (folder / "fivrs_scene.py").write_text(BLENDER_SCRIPT)
    (folder / "bundle.json").write_text(json.dumps(info or {}, indent=2) + "\n")
    return folder


def find_blender():
    """The Blender executable: on PATH, the snap, or a build unpacked under ~/apps or ~/blender*."""
    for name in ("blender",):
        p = shutil.which(name)
        if p:
            return p
    for p in [pathlib.Path("/snap/bin/blender"), *sorted(pathlib.Path.home().glob("apps/blender*/blender")),
              *sorted(pathlib.Path.home().glob("blender*/blender"))]:
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def running(folder):
    """True if a Blender is open on this bundle."""
    try:
        out = subprocess.run(["pgrep", "-af", "blender"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return str(pathlib.Path(folder).resolve()) in out


def launch(folder, blender=None):
    """Start Blender on the bundle (the saved .blend if there is one). Returns the Popen."""
    folder = pathlib.Path(folder)
    blender = blender or find_blender()
    if blender is None:
        raise FileNotFoundError("Blender not found: sudo snap install blender --classic")
    blend = folder / "ct_ear.blend"
    args = [blender] + ([str(blend)] if blend.exists() else []) + ["--python", str(folder / "fivrs_scene.py")]
    log = open(folder / "blender.log", "w")
    return subprocess.Popen(args, cwd=str(folder), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def read_cartilage_edit(folder):
    """The cartilage as edited in Blender (volume mm), or None."""
    p = pathlib.Path(folder) / CARTILAGE_EDITED
    return read_ply(p)[:2] if p.exists() else None


def read_edit(folder):
    """The ear as edited in Blender (volume mm) and its auricle weights, or None if not saved yet."""
    p = pathlib.Path(folder) / EDITED
    return read_ply(p) if p.exists() else None


# -- runs inside Blender ----------------------------------------------------------------
BLENDER_SCRIPT = r'''
"""FIVRS: CT ear editing scene. Run by Blender (blender_edit.launch)."""
import os
import struct

import json

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

FOLDER = os.path.dirname(os.path.abspath(__file__))
REFS = {  # name: (file, RGBA)
    "bone": ("bone.ply", (0.62, 0.62, 0.60, 1.0)),
    "envelope_skin": ("envelope_skin.ply", (0.35, 0.55, 0.95, 0.35)),
    "eac": ("eac.ply", (1.0, 0.45, 0.25, 1.0)),
}


def read_ply(path):
    data = open(path, "rb").read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    nv = nf = 0
    props, cur = [], None
    for line in data[:end].decode().splitlines():
        w = line.split()
        if w[:2] == ["element", "vertex"]:
            nv, cur = int(w[2]), "v"
        elif w[:2] == ["element", "face"]:
            nf, cur = int(w[2]), "f"
        elif w and w[0] == "property" and cur == "v":
            props.append(w[2])
    va = np.frombuffer(data, dtype=[(p, "<f4") for p in props], count=nv, offset=end)
    fa = np.frombuffer(data, dtype=[("n", "u1"), ("i", "<i4", 3)], count=nf, offset=end + va.nbytes)
    v = np.stack([va["x"], va["y"], va["z"]], axis=1)
    return v, fa["i"], (va["auricle"] if "auricle" in props else None)


def make_object(name, path):
    v, t, w = read_ply(path)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.astype(np.float32).ravel())
    me.loops.add(3 * len(t))
    me.loops.foreach_set("vertex_index", t.astype(np.int32).ravel())
    me.polygons.add(len(t))
    me.polygons.foreach_set("loop_start", np.arange(0, 3 * len(t), 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob, w


def build_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.001  # 1 unit = 1 mm
    sc.unit_settings.length_unit = "MILLIMETERS"
    for name, (fname, rgba) in REFS.items():
        ob, _w = make_object(name, os.path.join(FOLDER, fname))
        ob.color = rgba
        ob.show_transparent = rgba[3] < 1.0
        ob.hide_select = True  # references: seen, never edited
    ear, w = make_object("ct_ear", os.path.join(FOLDER, "ct_ear.ply"))
    ear.color = (0.95, 0.78, 0.68, 1.0)
    vg = ear.vertex_groups.new(name="auricle")
    vg.add(list(range(len(ear.data.vertices))), 0.0, "REPLACE")  # unpainted shows blue, not black
    if w is not None:
        idx = np.nonzero(w > 0.5)[0].tolist()
        if idx:
            vg.add(idx, 1.0, "REPLACE")
    info = {}
    p = os.path.join(FOLDER, "bundle.json")
    if os.path.exists(p):
        info = json.load(open(p))
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_start = 0.1
                    space.clip_end = 5000.0
                    space.shading.type = "SOLID"
                    space.shading.color_type = "OBJECT"
                    if info.get("lateral") and info.get("superior"):  # look in from the side, superior up
                        z = np.asarray(info["lateral"], float)
                        y = np.asarray(info["superior"], float)
                        x = np.cross(y, z)
                        space.region_3d.view_rotation = Matrix((x, y, z)).transposed().to_quaternion()
                        space.region_3d.view_perspective = "PERSP"
    bpy.context.view_layer.objects.active = ear
    ear.select_set(True)
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            with bpy.context.temp_override(area=area, region=[r for r in area.regions if r.type == "WINDOW"][0]):
                bpy.ops.view3d.view_selected()
    bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(FOLDER, "ct_ear.blend"))


def write_object(ob, path, group=None):
    """Triangulated mesh of ``ob`` (world mm) as binary PLY, with ``group`` weights if given."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.ensure_lookup_table()
    mw = np.array(ob.matrix_world)
    v = np.array([x.co[:] for x in bm.verts], np.float64) @ mw[:3, :3].T + mw[:3, 3]
    t = np.array([[l.vert.index for l in f.loops] for f in bm.faces], np.int32).reshape(-1, 3)
    w = None
    if group is not None and group in ob.vertex_groups:
        deform = bm.verts.layers.deform.active
        gi = ob.vertex_groups[group].index
        w = np.array([(x[deform].get(gi, 0.0) if deform is not None else 0.0) for x in bm.verts], np.float32)
    bm.free()
    props = "property float x\nproperty float y\nproperty float z\n" + ("property float auricle\n" if w is not None else "")
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n%selement face %d\n"
            "property list uchar int vertex_indices\nend_header\n" % (len(v), props, len(t)))
    va = np.zeros(len(v), dtype=[("p", "<f4", 3)] + ([("w", "<f4")] if w is not None else []))
    va["p"] = v
    if w is not None:
        va["w"] = w
    fa = np.zeros(len(t), dtype=[("n", "u1"), ("i", "<i4", 3)])
    fa["n"] = 3
    fa["i"] = t
    with open(path, "wb") as f:
        f.write(head.encode())
        f.write(va.tobytes())
        f.write(fa.tobytes())
    return len(v), len(t)


def load_cartilage(report=print):
    """The cartilage the viewer built (cartilage.ply) as object 'cartilage', selected; the ear hidden."""
    path = os.path.join(FOLDER, "cartilage.ply")
    ob = bpy.data.objects.get("cartilage")
    if ob is None:
        if not os.path.exists(path):
            report("No cartilage.ply yet: press 'Edit cartilage' in the viewer")
            return None
        try:
            if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
        ob, _w = make_object("cartilage", path)
        ob.color = (0.35, 0.6, 1.0, 1.0)
    ear = bpy.data.objects.get("ct_ear")
    if ear is not None:
        ear.hide_set(True)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    return ob


def export_for_fivrs():
    """Write what the viewer reads: the ear with its auricle paint, and the cartilage (modifiers,
    such as a Boolean cut, applied). Runs on every save (Ctrl+S)."""
    out = []
    ear = bpy.data.objects.get("ct_ear")
    if ear is not None:
        n = write_object(ear, os.path.join(FOLDER, "ct_ear_edited.ply"), group="auricle")
        out.append("ear %d vertices" % n[0])
    cart = bpy.data.objects.get("cartilage")
    if cart is not None:
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(cart.evaluated_get(dg))
        tmp = bpy.data.objects.new("_cart_eval", me)
        tmp.matrix_world = cart.matrix_world
        n = write_object(tmp, os.path.join(FOLDER, "cartilage_edited.ply"))
        bpy.data.objects.remove(tmp)
        bpy.data.meshes.remove(me)
        out.append("cartilage %d vertices" % n[0])
    return ", ".join(out)


@bpy.app.handlers.persistent
def fivrs_on_save(_dummy=None):
    try:
        msg = export_for_fivrs()
        print("FIVRS: sent to the viewer:", msg)
    except Exception as e:  # noqa: BLE001  (never block a save)
        print("FIVRS: export failed:", e)


class FIVRS_OT_send(bpy.types.Operator):
    """Save, and send the ear and the cartilage to the FIVRS viewer (same as Ctrl+S)"""

    bl_idname = "fivrs.send"
    bl_label = "Send to viewer (Ctrl+S)"

    def execute(self, context):
        bpy.ops.wm.save_mainfile(filepath=os.path.join(FOLDER, "ct_ear.blend"))
        self.report({"INFO"}, "Sent to the viewer. It picks the edits up by itself.")
        return {"FINISHED"}


class FIVRS_OT_load_cartilage(bpy.types.Operator):
    """Bring in the cartilage framework the viewer built (as object 'cartilage')"""

    bl_idname = "fivrs.load_cartilage"
    bl_label = "Show cartilage"

    def execute(self, context):
        ob = load_cartilage(lambda m: self.report({"ERROR"}, m))
        return {"FINISHED"} if ob is not None else {"CANCELLED"}


class FIVRS_PT_panel(bpy.types.Panel):
    bl_label = "FIVRS ear"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FIVRS"

    def draw(self, context):
        col = self.layout.column()
        col.label(text="Skin (ct_ear):")
        col.label(text="  paint the auricle into the")
        col.label(text="  'auricle' group (weight 1);")
        col.label(text="  cut the rest away, or leave it.")
        col.label(text="Cartilage: cut and model it.")
        col.label(text="Ctrl+S sends both to the viewer.")
        col.operator("fivrs.send", icon="EXPORT")
        col.operator("fivrs.load_cartilage", icon="IMPORT")
        col.separator()
        for name in ("ct_ear", "cartilage", "envelope_skin", "bone", "eac"):
            o = bpy.data.objects.get(name)
            if o is not None:
                col.prop(o, "hide_viewport", text="Hide " + name)


def view_settings():
    """Every launch: paint shown over the shading (so the ear's shape reads), long clip range."""
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.overlay.weight_paint_mode_opacity = 0.6
                        space.clip_start = 0.1
                        space.clip_end = 5000.0


for cls in (FIVRS_OT_send, FIVRS_OT_load_cartilage, FIVRS_PT_panel):
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        pass
for h in list(bpy.app.handlers.save_post):  # one handler, even when the script runs again
    if getattr(h, "__name__", "") == "fivrs_on_save":
        bpy.app.handlers.save_post.remove(h)
bpy.app.handlers.save_post.append(fivrs_on_save)

if bpy.data.objects.get("ct_ear") is None:
    build_scene()
view_settings()
_req = os.path.join(FOLDER, "load_cartilage.request")
if os.path.exists(_req):  # the viewer's 'Edit cartilage': open on the cartilage
    os.remove(_req)
    load_cartilage()
'''
