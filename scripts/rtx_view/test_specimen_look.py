"""The UR5 look applies to UR5 specimens only.

    middle_ear_newton/.venv/bin/python scripts/rtx_view/test_specimen_look.py
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import markings  # noqa: E402
import rtx_materials  # noqa: E402
import specimen_look  # noqa: E402
import thin_bone  # noqa: E402

UR5 = REPO / "ADF" / "volume_ur5_specimen1_0p350mm.yaml"
OTHER = REPO / "ADF" / "volume_256_RT143.yaml"


def test_other_anatomies_are_unchanged():
    specimen_look.deactivate()
    before = (rtx_materials.load_document(), markings.load_parameters(), thin_bone.load_parameters())
    assert specimen_look.activate(OTHER) is False
    after = (rtx_materials.load_document(), markings.load_parameters(), thin_bone.load_parameters())
    assert before == after
    assert thin_bone.load_capsule(OTHER) is None


def test_ur5_specimen_gets_the_look():
    look = json.loads(specimen_look.LOOK_PATH.read_text())
    assert specimen_look.activate(UR5) is True
    try:
        document = rtx_materials.load_document()
        for key in specimen_look.BONE_SEGMENTS:
            assert document["segments"][key]["diffuse_reflection_color"] == look["bone_color"]
        assert markings.load_parameters()["bone_color"] == look["bone_color"]
        params = thin_bone.load_parameters()
        assert params["otic_band_mm"] == 2.5 and params["otic_strength"] == 1.0
        landed = np.array(params["otic_color"]) * np.array(look["bone_color"])
        assert np.allclose(landed, look["capsule_color"])
        with np.load(UR5.parent / "../resources/volumes/ur5_specimen1_0p350mm/otic_capsule.npz") as data:
            dense = data["dense_bone"]
        field = thin_bone.load_capsule(UR5)
        assert int((field < thin_bone.FAR_VALUE).sum()) == int((dense != 65535).sum())
    finally:
        specimen_look.deactivate()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
