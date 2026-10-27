"""Fluid collision proxy derived from the authoritative AMBF occupancy.

This package does not draw bone and does not run in the haptic loop.
"""

from .maelstrom_boundary import (
    FILTER_HALO,
    KERNEL_SIZE,
    TEXTURE_RES,
    influence_radius,
    recommended_band,
)
from .frozen_bone import FrozenBone, bake_frozen_bone
from .sdf_proxy import FluidCollisionProxy, TexturePatch

__all__ = [
    "FILTER_HALO",
    "FluidCollisionProxy",
    "FrozenBone",
    "KERNEL_SIZE",
    "TEXTURE_RES",
    "TexturePatch",
    "bake_frozen_bone",
    "influence_radius",
    "recommended_band",
]
