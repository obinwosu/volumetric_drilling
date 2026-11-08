"""EAC voxels load as air in RTX, and the plugin clears the same colors."""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import air_labels

PLUGIN = pathlib.Path(__file__).resolve().parents[2] / "plugin" / "volumetric_drilling" / "volumetric_drilling.cpp"


def test_eac_voxels_become_air_and_bone_stays():
    occupied = np.ones((4, 4, 4), dtype=bool)
    colors = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    colors[...] = (0xFF, 0xF9, 0xDB)  # bone
    colors[1:3, 1:3, :] = (0xFF, 0xE1, 0xD6)  # eac core
    result, cleared = air_labels.drop_air_labels(occupied, colors)
    assert cleared == 16
    assert not result[1:3, 1:3, :].any()
    assert int(result.sum()) == 64 - 16
    assert occupied.all(), "the input occupancy is not changed"


def test_empty_eac_colored_voxels_are_not_counted():
    occupied = np.zeros((2, 2, 2), dtype=bool)
    colors = np.full((2, 2, 2, 3), (0xFF, 0xE1, 0xD6), dtype=np.uint8)
    result, cleared = air_labels.drop_air_labels(occupied, colors)
    assert cleared == 0
    assert result is occupied


def test_plugin_clears_the_same_colors():
    text = PLUGIN.read_text()
    block = re.search(r"kAirLabelRgb\[\]\[3\]\s*=\s*\{(.*?)\n\};", text, re.S)
    assert block, "kAirLabelRgb not found in the plugin"
    found = re.findall(r"\{\s*0x([0-9a-fA-F]{2}),\s*0x([0-9a-fA-F]{2}),\s*0x([0-9a-fA-F]{2})\s*\}", block.group(1))
    plugin_hexes = sorted("".join(parts).lower() for parts in found)
    assert plugin_hexes == sorted(air_labels.AIR_LABEL_HEXES)


if __name__ == "__main__":
    test_eac_voxels_become_air_and_bone_stays()
    test_empty_eac_colored_voxels_are_not_counted()
    test_plugin_clears_the_same_colors()
    print("air label tests passed")
