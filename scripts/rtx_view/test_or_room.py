"""Operating room dome, overhead lights, and the film glint boost."""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import or_room
import rtx_materials


def _sphere_mean(image):
    height = image.shape[0]
    elevation = (0.5 - (np.arange(height) + 0.5) / height) * math.pi
    weight = np.cos(elevation)[:, None, None]
    return float((image * weight).sum() / (weight.sum() * image.shape[1] * 3))


def test_room_keeps_the_fill_level():
    image = or_room.room_radiance(or_room.DEFAULTS["drape_color"], 1.0)
    assert abs(_sphere_mean(image) - 1.0) < 1e-6
    uniform = or_room.room_radiance(or_room.DEFAULTS["drape_color"], 0.0)
    assert np.allclose(uniform, 1.0)


def test_microscope_overhead_and_drapes_below():
    image = or_room.room_radiance((0.16, 0.42, 0.52), 1.0)
    zenith = image[0].mean(axis=0)
    ceiling = image[image.shape[0] // 4].mean(axis=0)
    floor = image[-1].mean(axis=0)
    assert zenith.sum() < 0.2 * ceiling.sum()
    # Drapes are blue-green, not the neutral ceiling.
    assert floor[2] > floor[0] and floor[1] > floor[0]


def test_hdr_file_has_the_lat_long_size():
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "room.hdr"
        image = or_room.room_radiance(or_room.DEFAULTS["drape_color"], 1.0)
        or_room._write_hdr(path, image)
        data = path.read_bytes()
        header = f"-Y {or_room.TEXTURE_HEIGHT} +X {or_room.TEXTURE_WIDTH}\n".encode()
        start = data.index(header) + len(header)
        assert data.startswith(b"#?RADIANCE")
        assert len(data) - start == or_room.TEXTURE_WIDTH * or_room.TEXTURE_HEIGHT * 4


def test_overhead_lights_face_the_field_from_above():
    center = np.array((-5.4, 0.1, -0.2))
    for mat in or_room.or_light_matrices(center):
        position = mat[3, :3]
        emits = -mat[2, :3]
        toward = (center - position) / np.linalg.norm(center - position)
        assert np.dot(emits, toward) > 0.999
        assert position[2] > center[2] + 0.8 * or_room.OR_LIGHT_DISTANCE * 0.8
        assert np.allclose(np.linalg.det(mat[:3, :3]), 1.0)


def test_room_is_bright_in_reflections_but_lights_bone_at_fill():
    lights = {"fill": 80.0, "fill_color": [0.45, 0.58, 0.82], "room": 1.0, "room_nits": 3000.0}
    dome = or_room.dome_inputs(lights)
    assert dome["inputs:intensity"] == 3000.0
    assert abs(dome["inputs:intensity"] * dome["inputs:diffuse"] - 80.0) < 1e-6
    assert or_room.dome_color(lights) == (1.0, 1.0, 1.0)
    flat = or_room.dome_inputs(dict(lights, room=0.0))
    assert flat == {"inputs:intensity": 80.0, "inputs:diffuse": 1.0}
    assert or_room.dome_color(dict(lights, room=0.0)) == (0.45, 0.58, 0.82)
    assert or_room.or_light_inputs({"or_lights": 5e5})["inputs:diffuse"] == 0.0


def test_glint_boost_scales_only_the_specular_color():
    look = {"specular_reflection_color": [1.0, 0.5, 0.25], "geometry_opacity": 0.25, "glint_boost": 4.0}
    inputs = rtx_materials._shader_inputs(look)
    assert "glint_boost" not in inputs
    assert inputs["specular_reflection_color"] == [4.0, 2.0, 1.0]
    assert inputs["geometry_opacity"] == 0.25
    assert rtx_materials._look_signature(look) != rtx_materials._look_signature(dict(look, glint_boost=1.0))


def test_glints_hold_as_opacity_drops():
    look = {"specular_reflection_color": [1.0, 1.0, 1.0], "enable_opacity": True, "glint_boost": 2.0}
    opaque = rtx_materials._shader_inputs(dict(look, geometry_opacity=1.0))
    clear = rtx_materials._shader_inputs(dict(look, geometry_opacity=0.25))
    # Cutout reflection is opacity times color, so the product stays the same.
    assert opaque["specular_reflection_color"][0] == 2.0
    assert abs(clear["specular_reflection_color"][0] * 0.25 - 2.0) < 1e-9
    bone = rtx_materials._shader_inputs({"enable_opacity": True, "geometry_opacity": 0.5, "specular_reflection_color": [1, 1, 1]})
    assert bone["specular_reflection_color"] == [1, 1, 1]


if __name__ == "__main__":
    test_room_keeps_the_fill_level()
    test_microscope_overhead_and_drapes_below()
    test_hdr_file_has_the_lat_long_size()
    test_overhead_lights_face_the_field_from_above()
    test_room_is_bright_in_reflections_but_lights_bone_at_fill()
    test_glint_boost_scales_only_the_specular_color()
    test_glints_hold_as_opacity_drops()
    print("or room ok")
