#!/usr/bin/env python3
"""QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_haptics_preset.py

Opens no simulator and no haptic device.
"""
import json
import pathlib
import struct
import sys
import tempfile

from PyQt5 import QtCore, QtWidgets

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import haptics_preset as hp  # noqa: E402
import tuning_shm as shm  # noqa: E402


class Config:
    def __init__(self, saved=None):
        self.yaml_data = {} if saved is None else {hp.CONFIG_KEY: saved}
        self.writes = []

    def save_value(self, key, value):
        self.yaml_data[key] = value
        self.writes.append((key, value))


def main():
    app = QtWidgets.QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as name:
        folder = pathlib.Path(name)
        path = folder / "presets.json"
        mine = shm.clean({"stiffness": 1.49, "hum_amp": 0.5})
        path.write_text(json.dumps({"mine": mine}))
        links = []

        def factory():
            links.append(shm.Link(folder / "p", folder / "t"))
            return links[-1]

        messages = []
        config = Config()
        picker = hp.HapticsPresetPicker(config, messages.append, path, factory)
        box = picker.build_group(None)
        items = [picker.combo.itemData(i) for i in range(picker.combo.count())]
        assert items[0] == hp.STOCK and "mine" in items and "Today's haptics" not in items
        assert picker.combo.currentData() == hp.STOCK

        # Stock: the plugin's tuning stays off, even if the variable was inherited.
        env = QtCore.QProcessEnvironment()
        env.insert("FIVRS_HAPTIC_TUNER", "1")
        assert picker.prepare_launch(env) is None and not env.contains("FIVRS_HAPTIC_TUNER") and not links

        # A chosen preset is saved and reaches the shared file with force on.
        picker.combo.setCurrentIndex(picker.combo.findData("mine"))
        picker.combo.activated.emit(picker.combo.currentIndex())
        assert config.writes == [(hp.CONFIG_KEY, "mine")] and "1.49" in picker.summary.text()
        env = QtCore.QProcessEnvironment()
        assert picker.prepare_launch(env) == "mine" and env.value("FIVRS_HAPTIC_TUNER") == "1"
        raw = struct.unpack_from(f"<{len(shm.KEYS) + 1}d", links[-1]._params, shm.PARAM_HEADER)
        assert dict(zip(shm.KEYS, raw))["hum_amp"] == 0.5 and raw[-1] == 1.0

        # Studies and UR5 launches stay on stock with a preset selected.
        env = QtCore.QProcessEnvironment()
        assert picker.prepare_launch(env, allowed=False) is None and not env.contains("FIVRS_HAPTIC_TUNER")

        # A preset saved later appears when the list opens.
        path.write_text(json.dumps({"mine": mine, "new": shm.stock()}))
        picker.refresh()
        assert picker.combo.findData("new") >= 0 and picker.combo.currentData() == "mine"

        # A deleted preset is shown as missing and launches on stock.
        path.write_text("{}")
        picker.refresh()
        assert "missing" in picker.combo.currentText()
        env = QtCore.QProcessEnvironment()
        assert picker.prepare_launch(env) is None and not env.contains("FIVRS_HAPTIC_TUNER")
        assert any("not found" in message for message in messages)
        box.deleteLater()
    del app
    print("haptics preset: all checks passed")


if __name__ == "__main__":
    main()
