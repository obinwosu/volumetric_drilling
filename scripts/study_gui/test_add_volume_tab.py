"""Tests for the "Add a volume" tab helpers (no GPU, no model).

QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_add_volume_tab.py
"""
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "segmentation"))

import add_volume_tab as tab  # noqa: E402
import tbone_worker as worker  # noqa: E402

import yaml  # noqa: E402

SETUP = """ambf_executable: ambf_simulator
volume1:
  adf_path: ../../ADF/volume_256_RT143.yaml
  icon_path: ./assets/RT143.png
  name: Anatomy A
  description: 'Anatomy A'
volume7:
  adf_path: ../../ADF/volume_256_LT144.yaml
  icon_path: ./assets/LT144.png
  name: Anatomy G
volumes:
- volume1
- volume7
study_choice: resident
"""


def test_register_adds_then_updates():
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "gui_setup.yaml"
        path.write_text(SETUP)
        adf = tab.REPO / "ADF" / "volume_x_0p196mm.yaml"
        icon = tab.REPO / "scripts" / "study_gui" / "assets" / "x.png"
        key = tab.register_volume(path, "Patient's ear", adf, icon, "it's left")
        assert key == "volume8"
        data = yaml.safe_load(path.read_text())
        assert data["volumes"] == ["volume1", "volume7", "volume8"]
        assert data[key]["name"] == "Patient's ear" and data[key]["description"] == "it's left"
        assert data[key]["adf_path"] == "../../ADF/volume_x_0p196mm.yaml"
        assert data["study_choice"] == "resident"  # rest of the file untouched
        again = tab.register_volume(path, "Renamed", adf, icon, "second run")
        data = yaml.safe_load(path.read_text())
        assert again == key and data["volumes"].count(key) == 1 and data[key]["name"] == "Renamed"


def test_rename_changes_only_that_name():
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "gui_setup.yaml"
        path.write_text(SETUP)
        adf = path.parent / "../../ADF/volume_256_LT144.yaml"
        assert tab.rename_volume(path, adf, "Left: Anatomy G") == "volume7"
        data = yaml.safe_load(path.read_text())
        assert data["volume7"]["name"] == "Left: Anatomy G" and data["volume1"]["name"] == "Anatomy A"
        assert path.read_text().replace('"Left: Anatomy G"', "Anatomy G") == SETUP
        assert tab.rename_volume(path, path.parent / "nope.yaml", "x") is None


def test_laterality_from_series_name():
    cases = {
        "Left Dehiscence  0.4  Ur70  3": "left",
        "RT TEMPORAL BONE 0.6": "right",
        "Temporal bone L": "left",
        "R IAC": "right",
        "Head 0.6 Hr60": "",
        "Bilateral temporal bones": "",
        "Lateral": "",
        "Left and right": "",
    }
    for text, side in cases.items():
        assert worker.laterality_from_text(text) == side, (text, worker.laterality_from_text(text))


def test_labels_use_anatomy_a_colors():
    # Same hexes as the JHU anatomies' PNGs, so the tuner's materials apply.
    hexes = {name: "%02x%02x%02x" % rgb for name, rgb in worker.LABELS.values()}
    assert hexes["Bone"] == "fff9db" and hexes["EAC"] == "ffe1d6"
    assert hexes["Malleus"] == "e900ff" and hexes["Incus"] == "00ff95" and hexes["Stapes"] == "3f00ff"
    assert hexes["Sinus + Dura"] == "6eb8d1" and hexes["Facial Nerve"] == "f4d631"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
