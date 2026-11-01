"""Operating room around the scope light, shared by the tuner and the live window.

Water only reads as water when it has something to mirror. With a uniform fill
dome the film reflects one flat color plus the scope disk. This adds:

- a room on the fill dome: dark microscope body overhead, a lit ceiling and
  walls, a dark gown at one side, and lit drapes from the horizon down;
- two overhead surgical lights, fixed in the room, that the wet bone mirrors;
- bloom on the brightest highlights, the way a surgical camera shows glints.

Everything is exposed for the 2e7-nit scope disk, so a room at fill level
(tens of nits) is black. The room and the overhead lights are therefore bright
in reflections only: their diffuse multiplier keeps the light they put on the
bone at the old fill level, or at zero for the overhead lights.

The room texture and bloom are written when the renderer starts. Brightness
changes live.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

DEFAULTS = {
    # 0 is the old uniform fill. 1 is the full room. Restart to apply.
    "room": 1.0,
    # How bright the room is in reflections, nits. The bone is lit at "fill".
    "room_nits": 3000.0,
    "drape_color": [0.16, 0.42, 0.52],
    # Overhead surgical lights, nits, seen only in reflections. 0 turns them off.
    "or_lights": 0.0,
    "or_light_color": [1.0, 0.97, 0.92],
    # Bloom brightness. 0 turns it off. Restart to apply.
    "bloom": 0.5,
    "bloom_cutoff": 2.0,
}

OR_LIGHT_PATHS = ("/root/_ORLight0", "/root/_ORLight1")
# Scene units are about 99 mm. The lamps hang about 1.5 m from the field and
# are about 50 cm across.
OR_LIGHT_DISTANCE = 15.0
OR_LIGHT_SIZE = 5.0
# (elevation, azimuth) in degrees, world +Z up, azimuth from world +X.
OR_LIGHT_PLACES = ((62.0, 35.0), (55.0, 205.0))

TEXTURE_WIDTH = 512
TEXTURE_HEIGHT = 256


def settings(lights: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update({key: lights[key] for key in DEFAULTS if key in lights})
    return merged


def dome_inputs(lights: dict) -> dict:
    """Fill dome intensity and diffuse multiplier. Bone fill stays at "fill"."""
    look = settings(lights)
    fill = float(lights.get("fill", 0.0))
    if float(look["room"]) <= 0.0:
        return {"inputs:intensity": fill, "inputs:diffuse": 1.0}
    nits = max(float(look["room_nits"]), fill, 1e-3)
    return {"inputs:intensity": nits, "inputs:diffuse": fill / nits}


def dome_color(lights: dict):
    """The room carries its own colors, so the fill tint only applies without it."""
    if float(settings(lights)["room"]) > 0.0:
        return (1.0, 1.0, 1.0)
    return tuple(float(c) for c in lights.get("fill_color", (1.0, 1.0, 1.0)))


def or_light_inputs(lights: dict) -> dict:
    return {"inputs:intensity": float(settings(lights)["or_lights"]), "inputs:diffuse": 0.0}


def render_settings(lights: dict) -> dict:
    """Startup ``omni:rtx`` settings for bloom."""
    from pxr import Gf

    look = settings(lights)
    scale = float(look["bloom"])
    cutoff = float(look["bloom_cutoff"])
    return {
        "omni:rtx:post:bloom:enabled": ("bool", scale > 0.0),
        "omni:rtx:post:bloom:scale": ("float", max(scale, 0.0)),
        "omni:rtx:post:bloom:cutoff": ("float3", Gf.Vec3f(cutoff, cutoff, cutoff)),
    }


def room_radiance(drape_color, room: float) -> np.ndarray:
    """Lat-long radiance, row 0 at the zenith. Mean over the sphere is 1."""
    height, width = TEXTURE_HEIGHT, TEXTURE_WIDTH
    elevation = (0.5 - (np.arange(height) + 0.5) / height) * math.pi
    azimuth = (np.arange(width) + 0.5) / width * 2.0 * math.pi
    elev = np.repeat(elevation[:, None], width, axis=1)
    azim = np.repeat(azimuth[None, :], height, axis=0)
    deg = np.degrees(elev)

    image = np.empty((height, width, 3), dtype=np.float64)
    drape = np.asarray(drape_color, dtype=np.float64).reshape(3)
    image[:] = drape * 0.6
    walls = deg > 4.0
    image[walls] = (0.42, 0.50, 0.47)
    ceiling = deg > 32.0
    image[ceiling] = (0.62, 0.60, 0.56)
    # Microscope head and arm straight overhead.
    image[deg > 72.0] = (0.03, 0.03, 0.035)
    # Surgeon in a dark gown on one side.
    gown = (deg > 4.0) & (deg < 48.0) & (np.abs(np.angle(np.exp(1j * (azim - math.pi)))) < 0.45)
    image[gown] = (0.04, 0.07, 0.12)
    # Soft edges, so the reflections do not show the texel steps.
    for _ in range(3):
        image = (image + np.roll(image, 1, axis=1) + np.roll(image, -1, axis=1)) / 3.0
        image[1:-1] = (image[1:-1] + image[:-2] + image[2:]) / 3.0

    weight = np.cos(elev)[..., None]
    image /= float((image * weight).sum() / (weight.sum() * 3.0))
    mix = float(min(max(room, 0.0), 1.0))
    return (1.0 - mix) + mix * image


def _write_hdr(path: Path, rgb: np.ndarray) -> None:
    """Flat Radiance RGBE."""
    rgb = np.ascontiguousarray(np.clip(rgb, 0.0, None), dtype=np.float64)
    peak = rgb.max(axis=2)
    mantissa, exponent = np.frexp(peak)
    scale = np.where(peak > 1e-32, mantissa * 256.0 / np.maximum(peak, 1e-32), 0.0)
    rgbe = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
    rgbe[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(peak > 1e-32, exponent + 128, 0).astype(np.uint8)
    height, width = rgb.shape[:2]
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode()
    path.write_bytes(header + rgbe.tobytes())


def write_room(lights: dict) -> Path | None:
    look = settings(lights)
    if float(look["room"]) <= 0.0:
        return None
    folder = Path(tempfile.gettempdir()) / "fivrs_rtx"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "or_room.hdr"
    _write_hdr(path, room_radiance(look["drape_color"], float(look["room"])))
    return path


def author_room(stage, dome_path: str, lights: dict) -> bool:
    """Put the room on the fill dome, bright in reflections and at fill level on the bone."""
    from pxr import Gf, Sdf

    prim = stage.GetPrimAtPath(dome_path)
    if not prim:
        return False
    for name, value in dome_inputs(lights).items():
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(float(value))
    prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*dome_color(lights)))
    path = write_room(lights)
    if path is None:
        return False
    # DomeLight_1 has poleAxis, so the room's ceiling is world +Z.
    prim.SetTypeName("DomeLight_1")
    prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
    prim.CreateAttribute("inputs:texture:format", Sdf.ValueTypeNames.Token).Set("latlong")
    prim.CreateAttribute("poleAxis", Sdf.ValueTypeNames.Token, variability=Sdf.VariabilityUniform).Set("Z")
    return True


def or_light_matrices(center) -> list[np.ndarray]:
    """Row-vector xforms. A rect light emits along its local -Z, here toward the field."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    world_up = np.array((0.0, 0.0, 1.0))
    out = []
    for elevation, azimuth in OR_LIGHT_PLACES:
        el, az = math.radians(elevation), math.radians(azimuth)
        offset = np.array((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        position = center + offset * OR_LIGHT_DISTANCE
        back = offset
        right = np.cross(world_up, back)
        right /= max(np.linalg.norm(right), 1e-8)
        up = np.cross(back, right)
        mat = np.eye(4)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = back
        mat[3, :3] = position
        out.append(mat)
    return out


def author_or_lights(stage, center, lights: dict) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    look = settings(lights)
    color = Gf.Vec3f(*[float(c) for c in look["or_light_color"]])
    for path, mat in zip(OR_LIGHT_PATHS, or_light_matrices(center)):
        rect = UsdLux.RectLight.Define(stage, path)
        rect.CreateWidthAttr(OR_LIGHT_SIZE)
        rect.CreateHeightAttr(OR_LIGHT_SIZE)
        rect.CreateIntensityAttr(float(look["or_lights"]))
        rect.CreateColorAttr(color)
        # Seen in the wet bone, not lighting it. The scope light does that.
        rect.CreateDiffuseAttr(0.0)
        xform = UsdGeom.Xformable(rect.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(mat.tolist()))
