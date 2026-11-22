#!/usr/bin/env python3
"""QT_QPA_PLATFORM=offscreen python3 scripts/haptics_tuner/test_haptics_tuner.py

Checks the window, the cube, and that Python and the C++ plugin header agree on
the shared file layout. Opens no simulator and no haptic device.
"""
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from PyQt5 import QtWidgets

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import haptics_tuner as app  # noqa: E402
import make_bone_cube  # noqa: E402
import tuning_shm as shm  # noqa: E402

REPO = HERE.parents[1]

READER = r"""
#include "haptic_tuning.h"
#include <cstdio>
int main(int, char** argv) {
    haptic_tuning::Link link;
    if (!link.open(argv[1], argv[2])) { puts("no link"); return 1; }
    const auto& p = link.read();
    printf("%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", p.cutGate, p.cutStartN, p.cutFullN, p.cutCurve,
           p.rateMm3[0], p.rateMm3[1], p.rateMm3[2], p.rateMm3[3], p.stiffnessMult, p.humOn, p.humAmpN, p.humHz,
           p.humDirection, p.humSpinUpS, p.humSpinDownS, p.humLoadAmp, p.humLoadPitch);
    const float row[8] = {1.5f, 2.0f, 0.3f, 2.2f, 0.75f, 4.0f, 11.0f, 4.0f};
    link.log(row);
}
"""


def test_layout_matches_plugin(folder):
    params, telemetry = folder / "params", folder / "telemetry"
    link = shm.Link(params, telemetry)
    sent = {key: spec.low + (spec.high - spec.low) * (index + 1) / 40 for index, (key, spec)
            in enumerate(shm.BY_KEY.items())}
    sent["cut_gate"], sent["hum_on"], sent["hum_direction"] = 1, 1, 2
    sent = link.write(sent)
    source, binary = folder / "reader.cpp", folder / "reader"
    source.write_text(READER)
    subprocess.run(["g++", "-std=c++17", "-I", str(REPO / "plugin/volumetric_drilling"), str(source), "-o", str(binary)], check=True)
    output = subprocess.run([str(binary), str(params), str(telemetry)], check=True, capture_output=True, text=True).stdout
    received = [float(word) for word in output.split()]
    assert len(received) == len(shm.KEYS)
    for key, value in zip(shm.KEYS, received):
        assert abs(value - sent[key]) < 1e-3 * max(1.0, abs(value)), (key, value, sent[key])
    row = link.recent(1)
    assert link.count() == 1 and np.allclose(row[0], [1.5, 2.0, 0.3, 2.2, 0.75, 4.0, 11.0, 4.0])


def test_cube(folder):
    adf = folder / "cube.yaml"
    out = make_bone_cube.build(folder / "cube", adf, voxels=12, margin=2)
    assert len(list(out.glob("*.png"))) == 12
    assert np.asarray(Image.open(out / "plane000.png"))[..., 3].max() == 0
    middle = np.asarray(Image.open(out / "plane006.png").convert("RGBA")) if (out / "plane006.png").exists() else None
    middle = np.asarray(Image.open(out / "plane006.png")) if middle is None else middle
    assert tuple(middle[6, 6]) == make_bone_cube.BONE_RGBA and middle[0, 0, 3] == 0
    assert int((middle[..., 3] > 0).sum()) == 64
    assert "count: 12" in adf.read_text()


def test_motor_state(folder):
    path = folder / "selfcheck.json"
    write = lambda status, result, pid=7: path.write_text(json.dumps(
        {"pid": pid, "status": status, "devices": [{"index": 0, "result": result}]}))
    assert shm.motor_state(path, 7)[0] is False  # no file yet
    write("pending", "docked")
    assert shm.motor_state(path, 7)[0] is False and "inkwell" in shm.motor_state(path, 7)[1]
    write("recovering", "dead")
    assert shm.motor_state(path, 7)[0] is False
    write("dead", "dead")
    assert shm.motor_state(path, 7)[0] is False and "NO FORCE" in shm.motor_state(path, 7)[1]
    write("ok", "ok", pid=6)
    assert shm.motor_state(path, 7)[0] is False  # an earlier launch's pass does not count
    write("ok", "ok")
    assert shm.motor_state(path, 7) == (True, "Touch motors: OK")


def test_window(folder):
    link = shm.Link(folder / "p2", folder / "t2")
    app.PRESET_FILE = folder / "presets.json"
    window = app.Tuner(link)
    window.timer.stop()
    read = lambda: dict(zip(shm.KEYS, struct.unpack_from(f"<{len(shm.KEYS)}d", link._params, shm.PARAM_HEADER)))
    assert read() == shm.stock()

    # A slider reaches the shared file at once.
    window.controls["hum_hz"].slider.setValue(500)
    assert abs(read()["hum_hz"] - 210) < 0.1

    # The full-rate force can never fall below the start force.
    window.controls["cut_gate"].box.setChecked(True)
    window.controls["cut_start_n"].spin.setValue(2.0)
    window.controls["cut_full_n"].spin.setValue(1.0)
    assert read()["cut_full_n"] >= 2.05 - 1e-9

    # Presets save, reload, and leave the built-ins alone.
    window.save_preset(name="mine")
    window.load_preset("Today's haptics")
    assert read() == shm.stock()
    window.load_preset("mine")
    assert read()["cut_start_n"] == 2.0
    window.save_preset(name="Today's haptics")
    assert "Today's haptics" not in window.user_presets

    # A/B and a blind round: settings are hidden, and reveal names the real slots.
    window.show()
    window.slots["A"] = shm.clean({"hum_hz": 111})
    window.slots["B"] = shm.clean({"hum_hz": 222})
    window.play("A")
    assert read()["hum_hz"] == 111
    window.play("B")
    assert read()["hum_hz"] == 222
    window.toggle_blind()
    assert not window.settings_area.isVisible() and window.blind_note.isVisible()
    mapping = dict(window.blind)
    window.play("2")
    assert read()["hum_hz"] == window.slots[mapping["2"]]["hum_hz"]
    window.toggle_blind()
    assert window.settings_area.isVisible() and f"1 was {mapping['1']}" in window.status.text()

    # Live plots draw from telemetry without a simulator.
    rows = np.zeros((500, 8), dtype=np.float32)
    rows[:, 1] = 1.2
    rows[:, 2] = 0.2
    rows[:, 5] = 2
    rows[:, 6] = shm.FLAG_DRILL_ON | shm.FLAG_CONTACT
    rows[:, 7] = 4
    link._telemetry[shm.TELEMETRY_HEADER:shm.TELEMETRY_HEADER + rows.nbytes] = rows.tobytes()
    struct.pack_into("<Q", link._telemetry, 16, 500)
    window.refresh()
    assert "drill ON" in window.live_label.text() and "4 mm" in window.live_label.text()
    window.grab()
    window.hide()

    # The launch stays on this branch's plugin and turns the tuner on.
    command = app.ambf_command()
    assert "FIVRS_HAPTIC_TUNER=1" in command and "launch_haptics.yaml" in command and "--rtx_view 1" in command
    assert ".haptics/install/volumetric_plugin/lib" in (REPO / "launch_haptics.yaml").read_text()


def main():
    qt = QtWidgets.QApplication(sys.argv)
    with tempfile.TemporaryDirectory(dir=shm.PARAM_PATH.parent if False else None) as name:
        folder = pathlib.Path(name)
        test_layout_matches_plugin(folder)
        test_cube(folder)
        test_motor_state(folder)
        test_window(folder)
    del qt
    print("haptics tuner: all checks passed")


if __name__ == "__main__":
    main()
