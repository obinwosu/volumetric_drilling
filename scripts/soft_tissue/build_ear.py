"""Build the finished ear composite in one step, save its meshes, and show them.

Uses what the placement viewer saved (the ear's placement, layer thicknesses, the thinning
around the ear) and the Blender edits (the auricle's skin sheet and its cartilage). Nothing to
click: it builds the same composite the viewer does and writes every mesh to

    resources/soft_tissue/composite/<volume>_<ear>/

as PLY files in volume mm (``composite.json`` lists them, with their colours), ready for the
solvers. Then it opens a plain 3D window (drag to orbit, scroll to zoom), or with ``--png``
renders a sheet of views instead.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/build_ear.py [ADF] [--png out.png] [--no-show]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import blender_edit as be  # noqa: E402
import ct_meatus as cm  # noqa: E402
import envelope as ev  # noqa: E402
import sheet_skin as ss  # noqa: E402

AURICLE = ev.ROOT / "resources" / "soft_tissue" / "auricle"
PINHOLE_RIM_MM = 20.0  # envelope holes with a rim shorter than this are closed
UNDER_AURICLE = ()  # solids built untrimmed under the auricle: none (they poked through the concha under traction)
SOLID_SMOOTH = 8  # smoothing passes on the layer solids (thin layers are 1-2 grid cells: scaly otherwise)
EAR = AURICLE / "ct_right_ear.ply"

# what each mesh is, its colour (RGB 0-255), drawn in this order
PARTS = {
    "bone": ("bone", (225, 222, 212)),
    "periosteum": ("periosteum (envelope)", None),
    "loose_areolar": ("loose areolar tissue / fascia (envelope)", None),
    "subcutaneous_fat": ("subcutaneous fat (envelope)", None),
    "temporalis": ("temporalis (envelope)", None),
    "envelope_skin": ("periauricular skin (envelope)", (214, 170, 150)),
    "cartilage": ("auricular cartilage", (90, 150, 255)),
    "auricle_skin": ("auricle skin (open ear, as edited in Blender, closed underneath)", (240, 205, 190)),
    "canal_skin": ("ear canal skin, open at the eardrum", (230, 120, 130)),
}


def _apply(T, v):
    v = np.asarray(v, float)
    return v @ T[:3, :3].T + T[:3, 3]


def COMPOSITE_OF(adf, ear=EAR):
    return ev.ROOT / "resources" / "soft_tissue" / "composite" / f"{pathlib.Path(adf).stem}_{pathlib.Path(ear).stem}"


def build(adf, ear=EAR, log=print):
    """The composite: dict name -> (verts, tris), volume mm."""
    import placement_viewer as pv

    t0 = time.time()
    vol = ev.Volume(adf)
    saved = ev.ROOT / "resources" / "soft_tissue" / "envelope" / f"{pathlib.Path(adf).stem}_{pathlib.Path(ear).stem}_placement.json"
    if not saved.exists():
        raise SystemExit(f"No saved placement: {saved}\nPlace the ear in the viewer and press Save first.")
    doc = json.loads(saved.read_text())
    place = pv.Placement(vol.frame, np.zeros(3))
    place.load(doc["ear"], doc.get("frame"))
    T = place.matrix()
    flip = (lambda t: t[:, [0, 2, 1]]) if np.linalg.det(T[:3, :3]) < 0 else (lambda t: t)
    thickness = {n: x for n, x, _ in ev.LAYERS}
    thickness.update(ev.upgrade_thickness(doc.get("thickness_mm", {})))
    muscle = ev.Muscle(**doc.get("muscle", {}))
    placed = COMPOSITE_OF(adf, ear) / "atlas_temporalis_muscle.ply"
    import os

    if placed.exists() and not os.environ.get("FIVRS_OUTLINE_TEMPORALIS"):  # the temporalis as placed from the atlas (atlas_placement.py), not the outline
        muscle = ev.Muscle(mesh_ply=str(placed))
        log(f"temporalis: the placed atlas muscle ({placed.name})")
    r = doc.get("auricle_ramp", {})
    ramp = ev.AuricleRamp(base_near_mm=r.get("base_near_mm", 2.0), base_far_mm=r.get("base_far_mm", 12.0))
    edited = pathlib.Path(ear).with_name(pathlib.Path(ear).stem + "_edited.ply")
    cart_edited = pathlib.Path(ear).with_name(pathlib.Path(ear).stem + "_cartilage_edited.ply")
    if not edited.exists():
        raise SystemExit(f"No Blender skin edit: {edited}\nEdit the skin in Blender (viewer: Edit skin) and press Ctrl+S.")
    v, t, w = be.read_ply(edited)
    v, t = _apply(T, v), flip(np.asarray(t))
    cart = None
    if cart_edited.exists():
        cv, ct = be.read_ply(cart_edited)[:2]
        cart = (_apply(T, cv), flip(np.asarray(ct)))
    log(f"loaded: placement, skin sheet ({len(t)} tris), cartilage ({0 if cart is None else len(cart[1])} tris)")
    eac = cm.EacField(vol)
    # the face's tissue in front of the ear comes up to the auricle's front edge
    face = ss.front_edge(v, t, vol.frame, eac)
    env = ev.Envelope(vol, face=face)
    log(f"envelope: {'with the face in front of the ear' if face is not None else 'no front edge found'} ({env.build_seconds:.1f} s)")
    if r.get("on", True):
        ramp.base_outline = ss.base_outline(v, t, eac)
        env.auricle_ramp = ramp
    base = ss.base_outline(v, t, eac)
    if base is not None:  # the auricular muscles in layer 3, from the physics fans to the auricle's base
        import auricular_muscles as am

        env.auricular_map = am.muscle_map(env, base, log=log)
    layers = env.layers(muscle, thickness)
    out = ss.join_sheet(vol, env, (v, t, w), layers["skin"], thickness, muscle=muscle, eac=eac,
                        others={k: m for k, m in layers.items() if k != "skin"}, cartilage=cart, log=log)
    leaks = ss.meatus_leaks(out, eac)
    log(f"built in {time.time() - t0:.1f} s; pour test: {'holds water' if not leaks else f'LEAKS {leaks}'}")
    parts = {"bone": vol.bone_mesh(), "envelope_skin": out["scalp"]}
    parts.update(out["layers"])
    # pinholes the layer tracing leaves (more where the temporalis lifts the layers): closed; the
    # canal opening and the ear's footprint are far longer than this and stay open
    import auricle_sim as asim

    for name in ("envelope_skin", *ev.LAYER_NAMES):
        if name in parts and name != "skin":
            v_, t_ = parts[name]
            n0 = len(t_)
            parts[name] = asim._close_small_holes(*asim.split_bowties(v_, t_), PINHOLE_RIM_MM)
            if len(parts[name][1]) > n0:
                log(f"{name}: closed pinholes ({len(parts[name][1]) - n0} triangles added)")
    if out.get("framework") is not None:
        parts["cartilage"] = out["framework"]
    parts["auricle_skin"] = out["ear"]
    parts["canal_skin"] = out["canal"]
    # the layers as solids, stacked tight on the bone and cut flat at the scene's box (a cutaway
    # block); the skin's solid stops where the auricle's own skin takes over
    import auricle as au

    box = env.scene_box(out["scalp"][0])
    solids = {"bone": env.layer_solids(muscle, thickness, box=box, surfaces=layers, smooth=SOLID_SMOOTH)["bone"]}
    face = env.fill_solid(box=box)
    if face is not None:  # the face's deep tissue the layers lie on in front of the ear
        solids["face_deep"] = face
    t_fields = env.thickness_fields(muscle, thickness)
    for name in ev.LAYER_NAMES:  # each layer's (pinhole-free) outer surface, pushed in by its thickness
        # the superficial layers (skin, fat, TPF) from the untrimmed envelope: they run on under the
        # auricle's base, out of sight at rest, so a pull that lifts the ear shows skin, not a hole
        if name in UNDER_AURICLE and name in layers:
            v_, t_ = layers[name]
            surf = asim._close_small_holes(*asim.split_bowties(v_, t_), PINHOLE_RIM_MM)
        else:
            surf = parts.get("envelope_skin" if name == "skin" else name)
        if surf is not None and len(surf[1]):
            sv_, st_ = env.extruded_solid(*surf, t_fields[name], box=box)
            if len(st_):
                solids[name] = (sv_, st_)
    for name, m in solids.items():
        parts[f"solid_{name}"] = m
    log(f"solids: {', '.join(f'{k} {len(m[1])}' for k, m in solids.items())} triangles; box {np.round(box[0], 1)} .. {np.round(box[1], 1)}")
    write_obstacle(env, t_fields, COMPOSITE_OF(adf, ear) / "head_obstacle.npz", log)
    return {k: (np.asarray(m[0], float), np.asarray(m[1], np.int32)) for k, m in parts.items() if m is not None and len(m[1])}, vol


def write_obstacle(env, t_fields, path, log=print):
    """The head as an obstacle for the soft bodies (nothing passes through anything; user): a
    signed distance grid, phi > 0 outside the head's tissue (bone, the face's deep tissue, and every
    envelope layer up to the skin), negative inside, on the envelope's grid. Saved for the solver's
    contact (fast_solver.set_obstacle)."""
    from scipy import ndimage

    total = np.zeros(env.shape, np.float32)
    for t_ in t_fields.values():
        total += np.broadcast_to(np.asarray(t_, np.float32), env.shape)
    solid = env.dist <= 0
    inside = ndimage.distance_transform_edt(solid, sampling=env.h).astype(np.float32)
    phi = np.where(solid, -inside - total, env.dist - total).astype(np.float32)
    phi = np.where(env.exterior | solid, phi, np.minimum(phi, -0.5))  # enclosed air counts as inside
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, phi=phi.astype(np.float16), origin=np.asarray(env.origin, float), h=np.asarray(env.h, float))
    log(f"obstacle: {phi.shape} grid, {float(env.h.max()):.2f} mm cells -> {path.name}")


def colour(name):
    c = PARTS.get(name, (name, None))[1]
    return c if c is not None else tuple(int(x) for x in ev.LAYER_RGB.get(name, (160, 160, 160)))


def save(parts, folder):
    folder.mkdir(parents=True, exist_ok=True)
    listing = {}
    for name, (v, t) in parts.items():
        ev.write_ply(folder / f"{name}.ply", v, t, colour(name))
        listing[name] = {"file": f"{name}.ply", "what": PARTS.get(name, (name,))[0], "rgb": list(colour(name)),
                         "vertices": len(v), "triangles": len(t)}
    (folder / "composite.json").write_text(json.dumps({"units": "volume mm", "meshes": listing}, indent=2))
    return folder


def _o3d_meshes(parts, hide=()):
    import open3d as o3d

    out = []
    for name, (v, t) in parts.items():
        if name in hide:
            continue
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(t))
        m.compute_vertex_normals()
        m.paint_uniform_color([c / 255 for c in colour(name)])
        out.append((name, m))
    return out


def show(parts, vol):
    import open3d as o3d

    meshes = [m for _n, m in _o3d_meshes(parts)]
    E = parts["auricle_skin"][0].mean(axis=0)
    L, S = np.asarray(vol.frame.lateral, float), np.asarray(vol.frame.superior, float)
    o3d.visualization.draw_geometries(meshes, window_name="Ear composite (drag: orbit, scroll: zoom, ctrl+drag: pan)",
                                      width=1600, height=1000, lookat=E, front=L, up=S, zoom=0.35)


def render(parts, vol, path, size=(800, 650)):
    """A sheet of views: lateral, the concha, from behind, from above, and a cut through the layers."""
    from open3d.visualization import rendering
    from PIL import Image, ImageDraw

    L, A, S = (np.asarray(x, float) for x in (vol.frame.lateral, vol.frame.anterior, vol.frame.superior))
    E = parts["auricle_skin"][0].mean(axis=0)
    views = [("lateral", L, S), ("into the concha", L + 0.3 * A + 0.2 * S, S), ("from behind", -A + 0.4 * L, S),
             ("from above", S + 0.3 * L, L), ("from below", -S + 0.5 * L, L), ("from in front", A + 0.6 * L, S)]
    ims = []
    for title, e, up in views:
        e = e / np.linalg.norm(e)
        r = rendering.OffscreenRenderer(*size)
        for name, m in _o3d_meshes(parts):
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            mat.base_color = [1, 1, 1, 1]
            r.scene.add_geometry(name, m, mat)
        r.scene.set_background([0.05, 0.05, 0.07, 1])
        r.setup_camera(36, E, E + 115 * e, up)
        for _ in range(3):
            img = np.asarray(r.render_to_image())
        del r
        im = Image.fromarray(img).convert("RGB")
        ImageDraw.Draw(im).text((10, 10), title, fill=(255, 255, 255))
        ims.append(im)
    W, H = ims[0].size
    sheet = Image.new("RGB", (3 * W, 2 * H + 40), (10, 10, 12))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % 3) * W, (i // 3) * H))
    d = ImageDraw.Draw(sheet)
    x = 10
    for name in parts:
        d.rectangle([x, 2 * H + 14, x + 12, 2 * H + 26], fill=colour(name))
        d.text((x + 18, 2 * H + 14), PARTS.get(name, (name,))[0].split(" (")[0], fill=(230, 230, 230))
        x += 30 + 7 * len(PARTS.get(name, (name,))[0].split(" (")[0])
    sheet.save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adf", nargs="?", default=str(ev.ROOT / "ADF" / "volume_256_RT143.yaml"))
    ap.add_argument("--ear", default=str(EAR))
    ap.add_argument("--png", help="render a sheet of views to this PNG")
    ap.add_argument("--no-show", action="store_true", help="do not open the 3D window")
    args = ap.parse_args()
    parts, vol = build(pathlib.Path(args.adf).resolve(), pathlib.Path(args.ear))
    folder = save(parts, ev.ROOT / "resources" / "soft_tissue" / "composite" / f"{pathlib.Path(args.adf).stem}_{pathlib.Path(args.ear).stem}")
    print(f"saved {len(parts)} meshes to {folder}")
    for name, (v, t) in parts.items():
        print(f"  {name:18s} {len(t):8d} tris  {PARTS.get(name, (name,))[0]}")
    if args.png:
        print(f"wrote {render(parts, vol, args.png)}")
    if not args.no_show:
        show(parts, vol)


if __name__ == "__main__":
    main()
