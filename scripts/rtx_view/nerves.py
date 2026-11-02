"""Nerve shader for the facial nerve and chorda tympani.

Each nerve vertex carries (along_mm, across_a_mm, across_b_mm, valid): how
far along the nerve's centreline it is, and its offset from that line in a
frame carried smoothly along it. Those values change smoothly, so the
renderer can blend them across triangles without losing anything. The
vessel pattern itself is drawn in ``mdl/Nerve.mdl`` per shading point, so it
stays sharp below the voxel size. ``vessel_amounts`` mirrors that pattern in
numpy for tests.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

import rtx_materials

HERE = pathlib.Path(__file__).resolve().parent
MDL_PATH = HERE / "mdl" / "Nerve.mdl"
CONFIG_PATH = HERE / "tuner_nerves.json"
# Same millimetre scale as markings.MM_TO_SIM and the burr mesh.
MM_TO_SIM = 0.01007
PRIMVAR = "nerve"
NERVE_NAMES = ("facial_nerve", "chorda_tympani")

DEFAULTS = {
    "use_shader": True,
    "nerve_color": [1.0, 0.90, 0.92],
    "nerve_roughness": 0.38,
    "fiber_strength": 0.10,
    "fiber_width_mm": 0.06,
    "subsurface_weight": 0.45,
    "subsurface_color": [1.0, 0.72, 0.68],
    "subsurface_mm": 0.4,
    "sheath_coat": 0.5,
    "coat_roughness": 0.08,
    "coat_ior": 1.40,
    "vessels_enabled": True,
    "vessel_strength": 1.0,
    "vessel_color": [0.62, 0.06, 0.06],
    "vessel_core_color": [0.32, 0.01, 0.02],
    "vessel_roughness": 0.22,
    "seed": 3.0,
    "main_count": 4.0,
    "main_length_mm": 16.0,
    "main_width_mm": 0.07,
    "main_continuity": 0.8,
    "waviness": 0.35,
    "branch_count": 16.0,
    "branch_length_mm": 3.0,
    "branch_width_mm": 0.025,
    "branch_density": 0.55,
    "branch_strength": 0.6,
}
BOOL_KEYS = ("use_shader", "vessels_enabled")
# Saved only; not a shader input.
LOCAL_KEYS = ("use_shader", "subsurface_mm")


def load_parameters() -> dict:
    params = dict(DEFAULTS)
    if CONFIG_PATH.is_file():
        try:
            stored = json.loads(CONFIG_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            stored = {}
        for key, value in stored.get("parameters", stored).items():
            if key in BOOL_KEYS:
                params[key] = bool(value)
            elif key in params:
                params[key] = value
    return params


def save_parameters(params: dict):
    payload = {"version": 1, "parameters": {key: params[key] for key in DEFAULTS}}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def is_nerve(name: str) -> bool:
    leaf = str(name or "").rsplit("/", 1)[-1]
    return leaf in NERVE_NAMES


def shader_inputs(params: dict, scene_per_mm: float) -> dict:
    """Shader inputs. Subsurface depth is in mm here and scene units in the shader."""
    inputs = {key: params[key] for key in DEFAULTS if key not in LOCAL_KEYS}
    inputs["subsurface_scale"] = float(params["subsurface_mm"]) * float(scene_per_mm)
    return inputs


# ---------------------------------------------------------------- coordinates


def world_to_mm(world, rotation, translation, mm_per_local) -> np.ndarray:
    """Volume-local millimetres. World is ``rotation @ local + translation``."""
    local = (np.asarray(world, np.float64) - np.asarray(translation, np.float64)) @ np.asarray(
        rotation, np.float64
    )
    return local * np.asarray(mm_per_local, np.float64)


def _weld(points_mm, tris, tolerance_mm=1e-3):
    """Block meshes repeat seam vertices. Merge them so the surface is one graph."""
    keys = np.round(np.asarray(points_mm, np.float64) / tolerance_mm).astype(np.int64)
    _unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    return np.asarray(points_mm, np.float64)[first], inverse[np.asarray(tris, np.int64)], inverse


def _surface_graph(points, tris):
    edges = np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
    edges = edges[edges[:, 0] != edges[:, 1]]
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    lengths = np.maximum(lengths, 1e-6)
    count = len(points)
    graph = sparse.coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(count, count)).tocsr()
    return graph.maximum(graph.T)


def _smooth_rows(rows, passes=3):
    """Light 1-2-1 smoothing down a list of points, ends held."""
    rows = np.asarray(rows, np.float64)
    if len(rows) < 3:
        return rows
    for _ in range(passes):
        inner = (rows[:-2] + 2.0 * rows[1:-1] + rows[2:]) * 0.25
        rows = np.concatenate((rows[:1], inner, rows[-1:]))
    return rows


def _transport_frames(tangents):
    """Normals carried along the line without twisting."""
    first = tangents[0]
    helper = np.array([0.0, 0.0, 1.0]) if abs(first[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    normal = np.cross(first, helper)
    normal /= np.linalg.norm(normal)
    normals = np.zeros_like(tangents)
    for index, tangent in enumerate(tangents):
        normal = normal - np.dot(normal, tangent) * tangent
        length = np.linalg.norm(normal)
        if length < 1e-9:
            normal = normals[index - 1] if index else np.cross(tangent, helper)
            length = np.linalg.norm(normal)
        normal = normal / length
        normals[index] = normal
    binormals = np.cross(tangents, normals)
    return normals, binormals


def _farthest(graph, start):
    distance = dijkstra(graph, directed=False, indices=start)
    distance = np.where(np.isfinite(distance), distance, -1.0)
    return int(np.argmax(distance))


def _distance_from(graph, sources):
    distance = dijkstra(graph, directed=False, indices=np.asarray(sources), min_only=True)
    return np.where(np.isfinite(distance), distance, 0.0)


def _along_from_ends(graph, bin_mm):
    """Rough order along the piece, from surface distance.

    The two vertices farthest apart over the surface are the nerve's ends.
    Distance is then re-measured from the whole end slice, not one rim point,
    and averaged from both ends. It only orders the cross-sections; the final
    along value comes from the centreline.
    """
    end_a = _farthest(graph, 0)
    end_b = _farthest(graph, end_a)
    from_a = _distance_from(graph, [end_a])
    from_b = _distance_from(graph, [end_b])
    for _ in range(2):
        length = float(max(from_a.max(), from_b.max()))
        s = 0.5 * (from_a + (length - from_b))
        lo, hi = float(s.min()), float(s.max())
        first = np.nonzero(s <= lo + bin_mm)[0]
        last = np.nonzero(s >= hi - bin_mm)[0]
        if len(first) == 0 or len(last) == 0:
            break
        from_a = _distance_from(graph, first)
        from_b = _distance_from(graph, last)
    length = float(max(from_a.max(), from_b.max()))
    s = 0.5 * (from_a + (length - from_b))
    adjacency = (graph > 0).astype(np.float64)
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).reshape(-1), 1.0)
    for _ in range(8):
        s = 0.5 * s + 0.5 * (adjacency @ s) / degree
    return s


def _resample(path, step_mm):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cumulative[-1])
    samples = np.arange(0.0, total + 1e-9, step_mm) if total > step_mm else np.array([0.0])
    if samples[-1] < total:
        samples = np.append(samples, total)
    line = np.column_stack([np.interp(samples, cumulative, path[:, axis]) for axis in range(3)])
    return line, samples, cumulative


def _piece_coordinates(points, graph, bin_mm, step_mm=0.05, order_weight=0.5, smooth_mm=0.9):
    """Along, across a, across b for one connected piece, in mm.

    Slices ordered by surface distance give the centreline. Each vertex is
    then projected onto the nearest part of that line whose surface order is
    close to its own, so a tight bend cannot send it to another limb.
    """
    s = _along_from_ends(graph, bin_mm)
    lo, hi = float(s.min()), float(s.max())
    bins = max(2, int(np.ceil((hi - lo) / max(bin_mm, 1e-3))))
    slot = np.clip(((s - lo) / max(hi - lo, 1e-9) * bins).astype(np.int64), 0, bins - 1)
    counts = np.bincount(slot, minlength=bins).astype(np.float64)
    keep = counts > 0
    centre_s = np.bincount(slot, weights=s, minlength=bins)[keep] / counts[keep]
    centre = np.stack(
        [np.bincount(slot, weights=points[:, axis], minlength=bins)[keep] / counts[keep] for axis in range(3)],
        axis=1,
    )
    out = np.zeros((len(points), 4), np.float64)
    out[:, 0] = s - lo
    out[:, 3] = 1.0
    if len(centre) < 4:
        return out
    # The end slices are cut short by the nerve's end, so their centroids lean.
    centre, centre_s = _smooth_rows(centre[1:-1]), centre_s[1:-1]
    line, arc, cumulative = _resample(centre, step_mm)
    line_s = np.interp(arc, cumulative, centre_s)
    tangents = _smooth_rows(np.gradient(line, axis=0), passes=4)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    normals, binormals = _transport_frames(tangents)
    tree = cKDTree(np.column_stack((line, order_weight * line_s)))
    _distance, nearest = tree.query(np.column_stack((points, order_weight * s)), k=1)
    offset = points - line[nearest]
    slide = np.einsum("ij,ij->i", offset, tangents[nearest])
    offset -= slide[:, None] * tangents[nearest]
    out[:, 0] = arc[nearest] + slide
    out[:, 1] = np.einsum("ij,ij->i", offset, normals[nearest])
    out[:, 2] = np.einsum("ij,ij->i", offset, binormals[nearest])
    # Where two parts of the line are about equally near (bulges, tight
    # bends), neighbouring vertices can pick different parts. Averaging with
    # neighbours turns those small breaks into gentle stretches.
    # Passes scale with edge length so the blend reaches about ``smooth_mm``
    # on any volume. The distance from the line is kept; only the angle blends.
    radius = np.hypot(out[:, 1], out[:, 2])
    edge_mm = float(np.median(graph.data)) if graph.nnz else 0.3
    passes = int(np.clip(round((smooth_mm / max(edge_mm, 1e-3)) ** 2), 4, 60))
    adjacency = (graph > 0).astype(np.float64)
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).reshape(-1), 1.0)
    for _ in range(passes):
        out[:, :3] = 0.5 * out[:, :3] + 0.5 * (adjacency @ out[:, :3]) / degree[:, None]
    blended = np.maximum(np.hypot(out[:, 1], out[:, 2]), 1e-9)
    out[:, 1:3] *= (radius / blended)[:, None]
    return out


def nerve_coordinates(points_mm, tris, bin_mm=0.5, min_vertices=24) -> np.ndarray:
    """Float4 per vertex: along, across a, across b (all mm), valid.

    Each connected piece gets its own line. Pieces under ``min_vertices``
    stay invalid and show the plain nerve look.
    """
    points_mm = np.asarray(points_mm, np.float64)
    out = np.zeros((len(points_mm), 4), np.float32)
    if len(points_mm) == 0 or len(tris) == 0:
        return out
    points, welded_tris, inverse = _weld(points_mm, np.asarray(tris).reshape(-1, 3))
    graph = _surface_graph(points, welded_tris)
    _count, labels = connected_components(graph, directed=False)
    welded = np.zeros((len(points), 4), np.float64)
    for label in np.unique(labels):
        members = np.nonzero(labels == label)[0]
        if len(members) < min_vertices:
            continue
        welded[members] = _piece_coordinates(points[members], graph[members][:, members], bin_mm)
    out[:] = welded[inverse].astype(np.float32)
    return out


# ------------------------------------------------- numpy mirror of Nerve.mdl


def _mod289(x):
    return x - np.floor(x * np.float32(1.0 / 289.0)) * np.float32(289.0)


def _permute(x):
    return _mod289((np.float32(34.0) * x + np.float32(1.0)) * x)


def _corner_gradient(cell):
    h = _permute(_permute(_permute(_mod289(cell[:, 0])) + _mod289(cell[:, 1])) + _mod289(cell[:, 2]))
    z = np.modf(h * np.float32(7.0 / 289.0))[0] * np.float32(2.0) - np.float32(1.0)
    phi = h * np.float32(6.2831853 * 13.0 / 289.0)
    r = np.sqrt(np.maximum(np.float32(1.0) - z * z, np.float32(0.0)))
    return np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1)


def gnoise(p) -> np.ndarray:
    p = np.asarray(p, np.float32)
    i = np.floor(p)
    f = p - i
    u = f * f * f * (f * (f * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0))
    values = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = np.array([dx, dy, dz], np.float32)
                values[(dx, dy, dz)] = np.einsum("ij,ij->i", _corner_gradient(i + offset), f - offset)

    def lerp(a, b, t):
        return a + (b - a) * t

    x00 = lerp(values[(0, 0, 0)], values[(1, 0, 0)], u[:, 0])
    x10 = lerp(values[(0, 1, 0)], values[(1, 1, 0)], u[:, 0])
    x01 = lerp(values[(0, 0, 1)], values[(1, 0, 1)], u[:, 0])
    x11 = lerp(values[(0, 1, 1)], values[(1, 1, 1)], u[:, 0])
    return lerp(lerp(x00, x10, u[:, 1]), lerp(x01, x11, u[:, 1]), u[:, 2])


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fiber_space(c, width_mm, offset):
    w = max(float(width_mm), 0.005)
    q = np.stack((c[:, 0] / (w * 18.0), c[:, 1] / w, c[:, 2] / w), axis=1)
    return (q + np.asarray(offset, np.float32)).astype(np.float32)


def vessel_noise(s, theta, count, length_mm, waviness, offset):
    rc = max(float(count), 0.5) / 6.2831853
    length = max(float(length_mm), 0.02)
    offset = np.asarray(offset, np.float32)
    slow = np.stack((s / length * 0.5, np.cos(theta) * rc * 0.5, np.sin(theta) * rc * 0.5), axis=1)
    slow = (slow + offset + np.array([5.0, 23.0, 61.0], np.float32)).astype(np.float32)
    turned = theta + float(waviness) * gnoise(slow) * 6.2831853 / max(float(count), 0.5)
    q = np.stack((s / length, np.cos(turned) * rc, np.sin(turned) * rc), axis=1) + offset
    return gnoise(q.astype(np.float32))


def vessel_distance(s, theta, radius_mm, count, length_mm, waviness, offset):
    """Noise value and distance in mm to its zero line, as Nerve.mdl computes it."""
    n0 = vessel_noise(s, theta, count, length_mm, waviness, offset)
    ds = 0.05 * max(float(length_mm), 0.02)
    dtheta = 0.05 * 6.2831853 / max(float(count), 0.5)
    along = (vessel_noise(s + ds, theta, count, length_mm, waviness, offset) - n0) / ds
    around = (vessel_noise(s, theta + dtheta, count, length_mm, waviness, offset) - n0) / (
        dtheta * np.maximum(radius_mm, 0.05)
    )
    slope = np.sqrt(along * along + around * around)
    return n0, np.abs(n0) / np.maximum(slope, 1e-4)


def _line_of(distance_mm, width_mm):
    half = 0.5 * max(float(width_mm), 0.001)
    return 1.0 - _smoothstep(half * 0.55, half, distance_mm)


def vessel_amounts(coords, params: dict):
    """Main-vessel and branch amounts per point, as Nerve.mdl computes them."""
    c = np.asarray(coords, np.float32).reshape(-1, 4)
    s = c[:, 0].astype(np.float64)
    radius = np.maximum(np.hypot(c[:, 1], c[:, 2]), 0.05)
    theta = np.arctan2(c[:, 2], c[:, 1]).astype(np.float64)
    seed = float(params["seed"])
    seed_offset = np.array([seed * 13.17, seed * 7.31, seed * 3.73], np.float32)
    count, length = float(params["main_count"]), float(params["main_length_mm"])
    _n, main_d = vessel_distance(s, theta, radius, count, length, float(params["waviness"]), seed_offset)
    gate_threshold = 0.45 - 0.9 * float(params["main_continuity"])
    main_gate = _smoothstep(
        gate_threshold - 0.15,
        gate_threshold + 0.15,
        vessel_noise(s * 0.35, theta, count * 0.35, length, 0.0, seed_offset + np.array([17.0, 3.0, 29.0], np.float32)),
    )
    main = _line_of(main_d, params["main_width_mm"]) * main_gate
    branch_offset = seed_offset + np.array([41.0, 7.0, 13.0], np.float32)
    count, length = float(params["branch_count"]), float(params["branch_length_mm"])
    _n, branch_d = vessel_distance(s, theta, radius, count, length, float(params["waviness"]) * 1.6, branch_offset)
    density_threshold = 0.5 - float(params["branch_density"])
    branch_gate = _smoothstep(
        density_threshold - 0.12,
        density_threshold + 0.12,
        vessel_noise(s * 0.4, theta, count * 0.4, length, 0.0, branch_offset + np.array([53.0, 29.0, 11.0], np.float32)),
    )
    branch = _line_of(branch_d, params["branch_width_mm"]) * branch_gate
    on = (1.0 if params["vessels_enabled"] else 0.0) * float(params["vessel_strength"]) * np.clip(c[:, 3], 0.0, 1.0)
    vm = np.clip(main * on, 0.0, 1.0)
    vb = np.clip(branch * float(params["branch_strength"]) * on, 0.0, 1.0)
    return vm, vb


def base_color(coords, params: dict) -> np.ndarray:
    """Diffuse color per point, as Nerve.mdl mixes it. For previews and tests."""
    c = np.asarray(coords, np.float32).reshape(-1, 4)
    seed = float(params["seed"])
    seed_offset = np.array([seed * 13.17, seed * 7.31, seed * 3.73], np.float32)
    fiber = 0.5 + 0.5 * gnoise(_fiber_space(c, params["fiber_width_mm"], seed_offset + np.array([91.0, 17.0, 43.0], np.float32)))
    vm, vb = vessel_amounts(c, params)
    s = c[:, 0].astype(np.float64)
    radius = np.maximum(np.hypot(c[:, 1], c[:, 2]), 0.05)
    theta = np.arctan2(c[:, 2], c[:, 1]).astype(np.float64)
    _n, main_d = vessel_distance(
        s, theta, radius, float(params["main_count"]), float(params["main_length_mm"]), float(params["waviness"]), seed_offset
    )
    core = 1.0 - _smoothstep(0.0, 0.3 * max(float(params["main_width_mm"]), 0.001), main_d)
    nerve = np.asarray(params["nerve_color"], np.float64)
    vessel = np.asarray(params["vessel_color"], np.float64)
    core_color = np.asarray(params["vessel_core_color"], np.float64)
    sheath = nerve[None, :] * (1.0 - float(params["fiber_strength"]) * fiber[:, None])
    branch_tint = vessel * 0.75 + nerve * 0.25
    with_branches = sheath * (1.0 - vb[:, None]) + branch_tint[None, :] * vb[:, None]
    main_tint = vessel[None, :] * (1.0 - core[:, None]) + core_color[None, :] * core[:, None]
    return with_branches * (1.0 - vm[:, None]) + main_tint * vm[:, None]


# ------------------------------------------------------------ USD and runtime


def ensure_shader(viewer):
    from pxr import Sdf, UsdShade

    path = f"{viewer._root_path}/Looks/nerve_shader"
    stage = viewer.stage
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    prim = shader.GetPrim()
    prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(MDL_PATH)))
    prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("Nerve")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for name in ("surface", "displacement", "volume"):
        material.CreateOutput(f"mdl:{name}", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader.ConnectableAPI(), "out"
        )
    return material, shader


def push_look(viewer, params: dict):
    """Author the shader inputs. The camera update sends them to the renderer."""
    _material, shader = ensure_shader(viewer)
    scene_per_mm = float(getattr(viewer, "_nerve_scene_per_mm", MM_TO_SIM))
    stored = shader_inputs(params, scene_per_mm)
    for key, value in stored.items():
        rtx_materials._author_input(shader, key, value)
    viewer._nerve_inputs = stored
    viewer._nerve_look_dirty = True


def bind_shader(viewer, prim):
    from pxr import UsdShade

    material, _shader = ensure_shader(viewer)
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def write_coordinates(viewer, prim, coords: np.ndarray, name: str):
    from pxr import Gf, Sdf, UsdGeom, Vt

    data = np.ascontiguousarray(coords, np.float32).reshape(-1, 4)
    api = UsdGeom.PrimvarsAPI(UsdGeom.Mesh(prim))
    primvar = api.GetPrimvar(PRIMVAR)
    if not primvar or primvar.GetTypeName() != Sdf.ValueTypeNames.Float4Array:
        primvar = api.CreatePrimvar(PRIMVAR, Sdf.ValueTypeNames.Float4Array, UsdGeom.Tokens.vertex)
    try:
        primvar.Set(Vt.Vec4fArray.FromNumpy(data))
    except Exception:
        primvar.Set([Gf.Vec4f(*(float(value) for value in row)) for row in data])
    pending = getattr(viewer, "_nerve_pending", None)
    if pending is None:
        pending = {}
        viewer._nerve_pending = pending
    pending[name] = data


def flush_runtime(viewer):
    """Run from the camera update, where the other tuners write."""
    if getattr(viewer, "_rtx", None) is None:
        return
    if getattr(viewer, "_nerve_look_dirty", False):
        _material, shader = ensure_shader(viewer)
        failed = False
        for key, value in getattr(viewer, "_nerve_inputs", {}).items():
            rtx_materials._author_input(shader, key, value)
            if not rtx_materials._runtime_input(viewer, shader, key, value):
                failed = True
        if not failed:
            viewer._nerve_look_dirty = False
    pending = getattr(viewer, "_nerve_pending", None) or {}
    for name, data in list(pending.items()):
        path = viewer._mesh_prim_paths.get(viewer._qualify(name))
        if path is None:
            continue
        try:
            viewer._write_runtime_array_attribute(path, f"primvars:{PRIMVAR}", data)
        except Exception as exc:
            if not getattr(viewer, "_nerve_primvar_warned", False):
                print(f"nerve coordinate write failed: {exc}", flush=True)
                viewer._nerve_primvar_warned = True
            return
        del pending[name]


def attach(viewer, prim, name: str, points_mm, tris, params: dict | None = None) -> bool:
    """Give one nerve mesh its coordinates and the nerve shader.

    ``points_mm`` are the mesh vertices in volume millimetres, same order.
    Returns False when the shader is switched off in tuner_nerves.json.
    """
    params = params if params is not None else load_parameters()
    if not params.get("use_shader", True):
        return False
    coords = nerve_coordinates(points_mm, np.asarray(tris).reshape(-1, 3))
    write_coordinates(viewer, prim, coords, name)
    bind_shader(viewer, prim)
    surfaces = getattr(viewer, "_nerve_surfaces", None)
    if surfaces is None:
        surfaces = {}
        viewer._nerve_surfaces = surfaces
    surfaces[name] = coords
    if len(coords):
        print(
            f"Nerve shader on {name}: {float(coords[:, 0].max()):.1f} mm along, "
            f"median radius {float(np.median(np.hypot(coords[:, 1], coords[:, 2]))):.2f} mm",
            flush=True,
        )
    return True


def sync_live(viewer):
    """Live window: pick up tuner_nerves.json when the tuner saves it."""
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = None
    if mtime != getattr(viewer, "_nerve_mtime", "unset"):
        viewer._nerve_mtime = mtime
        push_look(viewer, load_parameters())
    flush_runtime(viewer)


# ---------------------------------------------------------------- tuner panel


class NervePanel:
    """Nerve controls. Drawn inside the Shaders window."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.params = load_parameters()
        self.status = "Looks save as you drag."
        push_look(viewer, self.params)

    def _save(self):
        save_parameters(self.params)
        self.status = f"Saved {CONFIG_PATH.name}"

    def _toggle(self, imgui, key, label) -> bool:
        changed, value = imgui.checkbox(label, bool(self.params[key]))
        if changed:
            self.params[key] = bool(value)
        return bool(changed)

    def _slider(self, imgui, key, label, low, high, fmt) -> bool:
        changed, value = imgui.slider_float(label, float(self.params[key]), float(low), float(high), fmt)
        if changed:
            self.params[key] = float(value)
        return bool(changed)

    def _color(self, imgui, key, label) -> bool:
        changed, value = imgui.color_edit3(label, [float(channel) for channel in self.params[key]])
        if changed:
            self.params[key] = [float(channel) for channel in value]
        return bool(changed)

    def draw_controls(self, imgui):
        imgui.text("Facial nerve and chorda tympani")
        imgui.text_disabled("Vessels are drawn per pixel along each nerve.")
        surfaces = getattr(self.viewer, "_nerve_surfaces", {}) or {}
        if surfaces:
            imgui.text_wrapped("On: " + ", ".join(sorted(name.rsplit("/", 1)[-1] for name in surfaces)))
        else:
            imgui.text_wrapped("No nerve surface in this volume, or the shader is off.")
        changed = False
        bound = self._toggle(imgui, "use_shader", "Use nerve shader (next launch)")
        imgui.text_disabled("Their Structures entries do not show while this is on.")
        imgui.separator()
        imgui.text("Nerve")
        changed |= self._color(imgui, "nerve_color", "Nerve color")
        changed |= self._slider(imgui, "nerve_roughness", "Roughness", 0.0, 1.0, "%.2f")
        changed |= self._slider(imgui, "fiber_strength", "Fiber strength", 0.0, 0.6, "%.2f")
        changed |= self._slider(imgui, "fiber_width_mm", "Fiber width", 0.01, 0.3, "%.3f mm")
        changed |= self._slider(imgui, "subsurface_weight", "Subsurface", 0.0, 1.0, "%.2f")
        changed |= self._color(imgui, "subsurface_color", "Subsurface color")
        changed |= self._slider(imgui, "subsurface_mm", "Subsurface depth", 0.0, 3.0, "%.2f mm")
        changed |= self._slider(imgui, "sheath_coat", "Sheath coat", 0.0, 1.0, "%.2f")
        changed |= self._slider(imgui, "coat_roughness", "Coat roughness", 0.0, 1.0, "%.2f")
        changed |= self._slider(imgui, "coat_ior", "Coat IOR", 1.1, 1.8, "%.2f")
        imgui.separator()
        imgui.text("Vessels")
        changed |= self._toggle(imgui, "vessels_enabled", "Vessels")
        changed |= self._slider(imgui, "vessel_strength", "Vessel strength", 0.0, 1.0, "%.2f")
        changed |= self._slider(imgui, "seed", "Seed", 0.0, 40.0, "%.0f")
        changed |= self._color(imgui, "vessel_color", "Vessel color")
        changed |= self._color(imgui, "vessel_core_color", "Vessel core")
        changed |= self._slider(imgui, "vessel_roughness", "Vessel roughness", 0.0, 1.0, "%.2f")
        changed |= self._slider(imgui, "waviness", "Waviness", 0.0, 2.0, "%.2f")
        imgui.text("Main vessels")
        changed |= self._slider(imgui, "main_count", "Main vessels around", 1.0, 8.0, "%.1f")
        changed |= self._slider(imgui, "main_width_mm", "Main width", 0.01, 0.4, "%.3f mm")
        changed |= self._slider(imgui, "main_length_mm", "Main run", 0.5, 20.0, "%.1f mm")
        changed |= self._slider(imgui, "main_continuity", "Main continuity", 0.0, 1.0, "%.2f")
        imgui.text("Branches")
        changed |= self._slider(imgui, "branch_count", "Branches around", 2.0, 40.0, "%.1f")
        changed |= self._slider(imgui, "branch_width_mm", "Branch width", 0.005, 0.2, "%.3f mm")
        changed |= self._slider(imgui, "branch_length_mm", "Branch run", 0.1, 5.0, "%.2f mm")
        changed |= self._slider(imgui, "branch_density", "Branch density", 0.0, 1.0, "%.2f")
        changed |= self._slider(imgui, "branch_strength", "Branch strength", 0.0, 1.0, "%.2f")
        if changed:
            self.params["seed"] = float(int(round(self.params["seed"])))
            push_look(self.viewer, self.params)
            self._save()
        elif bound:
            self._save()
        imgui.separator()
        imgui.text_disabled(self.status)
        imgui.text_disabled(str(CONFIG_PATH))


# ----------------------------------------------------------------- self check


def _tube(radius_mm=0.6, length_mm=12.0, bend_mm=6.0, rings=160, around=20):
    """A bent closed-ish tube standing in for a nerve, with known centreline."""
    t = np.linspace(0.0, 1.0, rings)
    angle = t * (np.pi / 2.0)
    centre = np.stack((bend_mm * np.sin(angle), bend_mm * (1.0 - np.cos(angle)), t * length_mm * 0.3), axis=1)
    tangent = np.gradient(centre, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal, binormal = _transport_frames(tangent)
    theta = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    ring = (np.cos(theta)[None, :, None] * normal[:, None, :] + np.sin(theta)[None, :, None] * binormal[:, None, :])
    points = (centre[:, None, :] + radius_mm * ring).reshape(-1, 3)
    tris = []
    for i in range(rings - 1):
        for j in range(around):
            a = i * around + j
            b = i * around + (j + 1) % around
            tris.append((a, b, a + around))
            tris.append((b, b + around, a + around))
    return points, np.asarray(tris, np.int64), centre


def _self_check():
    points, tris, centre = _tube()
    # Split into two "blocks" with duplicated seam vertices, like the block mesher.
    half = len(tris) // 2
    second = tris[half:] + len(points)
    doubled = np.concatenate((points, points))
    coords = nerve_coordinates(doubled, np.concatenate((tris[:half], second)))
    true_length = float(np.sum(np.linalg.norm(np.diff(centre, axis=0), axis=1)))
    along = coords[:, 0]
    span = float(along.max() - along.min())
    if abs(span - true_length) > 0.15 * true_length:
        raise SystemExit(f"self-check along span {span:.2f} mm, tube is {true_length:.2f} mm")
    radius = np.hypot(coords[:, 1], coords[:, 2])
    if abs(float(np.median(radius)) - 0.6) > 0.12:
        raise SystemExit(f"self-check radius {float(np.median(radius)):.2f} mm, tube is 0.60 mm")
    ring_along = along[: len(points)].reshape(160, 20)
    # The test tube is open-ended; real nerve meshes are closed, so skip its last 1.5 mm.
    if float(np.ptp(ring_along, axis=1)[24:-24].max()) > 0.1:
        raise SystemExit("self-check one ring spreads too far along the nerve")
    ring_median = np.median(ring_along, axis=1)
    direction = np.sign(ring_median[-1] - ring_median[0])
    if not np.all(np.diff(ring_median)[10:-10] * direction > -1e-3):
        raise SystemExit("self-check along is not monotonic down the tube")
    # Pattern: a dense grid on the tube surface.
    rings, around = 600, 120
    s = np.linspace(0.0, 10.0, rings)
    theta = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    ss, tt = np.meshgrid(s, theta, indexing="ij")
    grid = np.stack((ss, 0.6 * np.cos(tt), 0.6 * np.sin(tt), np.ones_like(ss)), axis=-1).reshape(-1, 4)
    params = dict(DEFAULTS)
    vm, vb = vessel_amounts(grid, params)
    main_cover = float((vm > 0.5).mean())
    branch_cover = float((vb > 0.3).mean())
    if not 0.01 < main_cover < 0.25:
        raise SystemExit(f"self-check main vessel coverage {main_cover:.3f}")
    if not 0.002 < branch_cover < 0.25:
        raise SystemExit(f"self-check branch coverage {branch_cover:.3f}")
    field = vm.reshape(rings, around)
    lengthwise = float(np.mean(np.abs(np.diff(field, axis=0))))
    around_ds = float(np.mean(np.abs(np.diff(field, axis=1))))
    # Per-mm change: ring step is 10/599 mm, around step is 2*pi*0.6/120 mm.
    if lengthwise / (10.0 / (rings - 1)) >= around_ds / (2.0 * np.pi * 0.6 / around):
        raise SystemExit("self-check main vessels do not run along the nerve")
    off = dict(params, vessels_enabled=False)
    if float(np.max(vessel_amounts(grid, off)[0])) != 0.0:
        raise SystemExit("self-check vessels still show when switched off")
    print(
        "nerves self-check ok  "
        f"along {span:.2f}/{true_length:.2f} mm  radius {float(np.median(radius)):.2f} mm  "
        f"main {100 * main_cover:.1f}%  branches {100 * branch_cover:.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    _self_check()
