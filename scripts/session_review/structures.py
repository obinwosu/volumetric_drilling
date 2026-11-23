"""Names and display colors of the labelled structures in a recorded volume. No GUI here.

A structure's identity is its original segmentation color (see ADF/color/volume_labels.yaml and
LABELS in scripts/segmentation/tbone_worker.py). The order below is the order in the review menu.
"""
from __future__ import annotations

BONE_DISPLAY = (0.93, 0.89, 0.76)
INNER_EAR_DISPLAY = (0.30, 0.78, 0.62)  # the label's own gray-green is hard to see

NAMES = {
    "fff9db": "Bone",
    # Ossicles
    "e900ff": "Malleus", "fff5c5": "Malleus",
    "00ff95": "Incus", "e1ffd8": "Incus",
    "3f00ff": "Stapes", "d0e4ff": "Stapes",
    "ffe4c4": "Tympanic membrane",
    # Inner ear
    "5b7b5b": "Inner ear",
    "4184ff": "Scala tympani",
    "ffdebe": "Scala vestibuli",
    "5b627b": "Vestibular aqueduct",
    # Nerves
    "f4d631": "Facial nerve", "ff0004": "Facial nerve",
    "97831d": "Chorda tympani", "ff0104": "Chorda tympani",
    "f48e34": "Internal auditory canal",
    "ffbf87": "Superior vestibular nerve",
    "794618": "Inferior vestibular nerve",
    "dbf434": "Cochlear nerve",
    "ff8aab": "Cochleovestibular nerve",
    # Vessels and dura
    "d8644f": "Internal carotid artery", "bdffa9": "Internal carotid artery",
    "2864a0": "Sigmoid sinus",
    "aad7e8": "Dura",
    "6eb8d1": "Sigmoid sinus + dura", "ffea87": "Sigmoid sinus + dura",
    # Other
    "be6eaa": "Vestibular schwannoma",
    "640000": "Mandible",
    "ffe1d6": "Ear canal",
}
_ORDER = {key: k for k, key in enumerate(NAMES)}


def describe(hex_color: str, bone_hex: str = "fff9db") -> dict:
    """{hex, name, color (0-1 RGB)} for one label color."""
    key = hex_color.lower()
    rgb = tuple(int(key[k:k + 2], 16) / 255.0 for k in (0, 2, 4))
    if key == bone_hex.lower():
        return {"hex": key, "name": "Bone", "color": BONE_DISPLAY}
    if key == "5b7b5b":
        rgb = INNER_EAR_DISPLAY
    return {"hex": key, "name": NAMES.get(key, f"Structure #{key}"), "color": rgb}


def menu_order(hex_color: str, bone_hex: str = "fff9db"):
    key = hex_color.lower()
    if key == bone_hex.lower():
        return (-1, key)
    return (_ORDER.get(key, len(_ORDER)), key)
