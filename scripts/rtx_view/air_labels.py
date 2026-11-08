"""Segment colors that load as air.

The EAC label is a solid plug filling the ear canal. The real canal is air
lined by a sheet of skin, so its voxels are cleared at load: no drilling, no
haptic contact, no rendering. AMBF clears the same colors in
``plugin/volumetric_drilling/volumetric_drilling.cpp`` (``kAirLabelRgb``).
Keep the two lists the same, or AMBF and RTX occupancy disagree.

Colors stay in the color array. The mesher takes a vertex color from the
occupied side of each edge, so a cleared voxel's color is never drawn, and the
cortex-mask frame can still find the canal shape from the intact labels.
"""

from __future__ import annotations

import numpy as np

AIR_LABEL_HEXES = ("ffe1d6",)  # eac


def _rgb(hex_id: str) -> tuple[int, int, int]:
    return tuple(int(hex_id[offset : offset + 2], 16) for offset in (0, 2, 4))


def air_label_mask(occupied: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Occupied voxels whose RGB is one of the air labels."""
    mask = np.zeros(occupied.shape, dtype=bool)
    for hex_id in AIR_LABEL_HEXES:
        red, green, blue = _rgb(hex_id)
        mask |= (colors[..., 0] == red) & (colors[..., 1] == green) & (colors[..., 2] == blue)
    return mask & np.asarray(occupied, dtype=bool)


def drop_air_labels(occupied: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, int]:
    """Copy of ``occupied`` with the air labels cleared, and how many were cleared."""
    mask = air_label_mask(occupied, colors)
    cleared = int(mask.sum())
    if cleared == 0:
        return occupied, 0
    result = np.asarray(occupied, dtype=bool).copy()
    result[mask] = False
    return result, cleared
