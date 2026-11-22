"""Settings and telemetry shared with the plugin. Mirrors plugin/.../haptic_tuning.h."""
import dataclasses
import json
import mmap
import os
import pathlib
import struct

import numpy as np

PARAM_PATH = pathlib.Path("/dev/shm/fivrs_haptic_tuning")
TELEMETRY_PATH = pathlib.Path("/dev/shm/fivrs_haptic_telemetry")
PARAM_MAGIC = 0x4E555448
TELEMETRY_MAGIC = 0x4C455448
VERSION = 1
PARAM_HEADER = 16
TELEMETRY_HEADER = 24
TELEMETRY_ROWS = 4096
TELEMETRY_FIELDS = 8

FLAG_DRILL_ON = 1
FLAG_CONTACT = 2
FLAG_TRIPPED = 4
FLAG_GATED = 8
FLAG_FORCE_HELD = 16
FLAG_LIMITED = 32

# One newton-per-mm figure for the stiffness slider: 3 x 400 N/m / workspace scale,
# through the plugin's hand-to-burr motion, at a 1 kHz physics loop.
STIFFNESS_N_PER_MM = 0.216


@dataclasses.dataclass(frozen=True)
class Spec:
    key: str
    label: str
    unit: str
    low: float
    high: float
    stock: float
    decimals: int = 2
    group: str = ""
    kind: str = "slider"  # slider, toggle, choice
    choices: tuple = ()
    tip: str = ""


# File order. Stock values reproduce today's haptics.
SPECS = (
    Spec("cut_gate", "Force decides cutting", "", 0, 1, 0, 0, "Bone removal", "toggle",
         tip="Off: today's removal, any touch cuts at full speed."),
    Spec("cut_start_n", "Cutting starts at", "N", 0.0, 3.2, 0.4, 2, "Bone removal",
         tip="Below this push the spinning burr rubs without cutting."),
    Spec("cut_full_n", "Full speed at", "N", 0.05, 3.3, 1.6, 2, "Bone removal",
         tip="Pushing harder than this cuts no faster."),
    Spec("cut_curve", "Response curve", "", 0.25, 4.0, 1.0, 2, "Bone removal",
         tip="1 = linear. Above 1 = slow to bite, then fast."),
    Spec("rate_1", "1 mm burr, full rate", "mm³/s", 0, 400, 4, 1, "Bone removal"),
    Spec("rate_2", "2 mm burr, full rate", "mm³/s", 0, 400, 12, 1, "Bone removal"),
    Spec("rate_4", "4 mm burr, full rate", "mm³/s", 0, 400, 40, 1, "Bone removal"),
    Spec("rate_6", "6 mm burr, full rate", "mm³/s", 0, 400, 80, 1, "Bone removal"),
    Spec("stiffness", "Bone stiffness", "×", 0.1, 1.85, 1.0, 2, "Contact",
         tip="1 = today (0.22 N per mm of hand motion). 1.85 = the Touch's rated 400 N/m."),
    Spec("hum_on", "Hum on", "", 0, 1, 1, 0, "Drill hum", "toggle"),
    Spec("hum_amp", "Strength", "N", 0.0, 0.6, 0.1386, 3, "Drill hum",
         tip="Hard limit 0.6 N. Sustained strong hum warms the motors."),
    Spec("hum_hz", "Frequency", "Hz", 20, 400, 79.577, 1, "Drill hum",
         tip="Today: 80 Hz. The hand is most sensitive to vibration near 200-300 Hz."),
    Spec("hum_direction", "Direction", "", 0, 2, 0, 0, "Drill hum", "choice",
         ("Diagonal (today)", "Along the shaft", "Around the shaft"),
         tip="Around the shaft acts like a spinning imbalance."),
    Spec("hum_up", "Spin-up", "s", 0, 3, 0, 2, "Drill hum"),
    Spec("hum_down", "Spin-down", "s", 0, 3, 0, 2, "Drill hum"),
    Spec("hum_load_amp", "Stronger under load", "×", -1, 2, 0, 2, "Drill hum",
         tip="Added hum at full device force. 0.5 = 50% stronger."),
    Spec("hum_load_pitch", "Pitch drop under load", "", 0, 0.6, 0, 2, "Drill hum",
         tip="Frequency drop at full device force. 0.2 = 20% lower."),
)
KEYS = tuple(spec.key for spec in SPECS)
BY_KEY = {spec.key: spec for spec in SPECS}
PARAM_BYTES = PARAM_HEADER + (len(SPECS) + 1) * 8  # + force enable
TELEMETRY_BYTES = TELEMETRY_HEADER + TELEMETRY_ROWS * TELEMETRY_FIELDS * 4


def stock():
    return {spec.key: float(spec.stock) for spec in SPECS}


def clean(values):
    """A full, in-range settings dict from any saved one."""
    result = stock()
    for key, value in (values or {}).items():
        if key in BY_KEY:
            spec = BY_KEY[key]
            result[key] = min(max(float(value), spec.low), spec.high)
    result["cut_full_n"] = max(result["cut_full_n"], result["cut_start_n"] + 0.05)
    return result


BUILT_IN = {
    "Today's haptics": stock(),
    "Firm cut, strong hum (starting point)": clean({
        "cut_gate": 1, "cut_start_n": 0.6, "cut_full_n": 2.2, "cut_curve": 1.4,
        "rate_1": 3, "rate_2": 10, "rate_4": 30, "rate_6": 60, "stiffness": 1.5,
        "hum_amp": 0.3, "hum_hz": 200, "hum_direction": 2, "hum_up": 0.25, "hum_down": 0.45,
        "hum_load_amp": 0.3, "hum_load_pitch": 0.15}),
    "Light cut, soft hum (starting point)": clean({
        "cut_gate": 1, "cut_start_n": 0.2, "cut_full_n": 1.1, "cut_curve": 1.0,
        "rate_1": 5, "rate_2": 16, "rate_4": 50, "rate_6": 100, "stiffness": 1.0,
        "hum_amp": 0.14, "hum_hz": 250, "hum_direction": 2, "hum_up": 0.2, "hum_down": 0.35,
        "hum_load_amp": 0.2, "hum_load_pitch": 0.1}),
}


def motor_state(path, pid):
    """(motors confirmed, text) from libhdPhantom's self-check, for display only."""
    try:
        data = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return False, "Touch motors: waiting for the warm-up and self-check"
    if not pid or data.get("pid") != pid:
        return False, "Touch motors: waiting for the warm-up and self-check"
    drill = next((d for d in data.get("devices") or [] if d.get("index") == 0), None)
    result = (drill or {}).get("result")
    if data.get("status") == "dead" or (result == "dead" and data.get("status") != "recovering"):
        return False, "Touch motors: NO FORCE. They did not move after re-opening. Stop and start again."
    if data.get("status") == "recovering" or result == "dead":
        return False, "Touch motors: did not move, re-opening the Touch"
    if result == "docked":
        return False, "Touch motors: stylus in the inkwell, checked automatically when it is lifted"
    if result == "ok":
        return True, "Touch motors: OK"
    if result == "held":
        return True, "Touch motors: not checked, the stylus was never still"
    return False, "Touch motors: checking, hold the stylus still"


def _map(path, size, magic):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fresh = os.fstat(fd).st_size != size
        if fresh:
            os.ftruncate(fd, 0)
            os.ftruncate(fd, size)
        memory = mmap.mmap(fd, size)
    finally:
        os.close(fd)
    if fresh or struct.unpack_from("<II", memory, 0) != (magic, VERSION):
        memory[:size] = bytes(size)
        struct.pack_into("<II", memory, 0, magic, VERSION)
    return memory


class Link:
    def __init__(self, param_path=PARAM_PATH, telemetry_path=TELEMETRY_PATH):
        self._params = _map(param_path, PARAM_BYTES, PARAM_MAGIC)
        self._telemetry = _map(telemetry_path, TELEMETRY_BYTES, TELEMETRY_MAGIC)
        self.force_enable = False
        self._last = stock()

    def write(self, values):
        values = clean(values)
        self._last = values
        payload = struct.pack(f"<{len(KEYS) + 1}d", *(values[key] for key in KEYS),
                              1.0 if self.force_enable else 0.0)
        seq = struct.unpack_from("<I", self._params, 8)[0]
        seq = seq + 1 if seq % 2 == 0 else seq  # odd while writing
        struct.pack_into("<I", self._params, 8, seq & 0xFFFFFFFF)
        self._params[PARAM_HEADER:PARAM_BYTES] = payload
        struct.pack_into("<I", self._params, 8, (seq + 1) & 0xFFFFFFFF)
        return values

    def set_force_enable(self, enabled):
        """On while the tuner's simulator runs."""
        if bool(enabled) != self.force_enable:
            self.force_enable = bool(enabled)
            self.write(self._last)

    def count(self):
        return struct.unpack_from("<Q", self._telemetry, 16)[0]

    def recent(self, rows):
        """Newest samples, oldest first, shape (n, 8)."""
        count = self.count()
        # The row being written next may be half new, so it is left out.
        rows = min(rows, count, TELEMETRY_ROWS - 8)
        if rows <= 0:
            return np.zeros((0, TELEMETRY_FIELDS), dtype=np.float32)
        table = np.frombuffer(self._telemetry, dtype=np.float32, count=TELEMETRY_ROWS * TELEMETRY_FIELDS,
                              offset=TELEMETRY_HEADER).reshape(TELEMETRY_ROWS, TELEMETRY_FIELDS)
        index = np.arange(count - rows, count) % TELEMETRY_ROWS
        return table[index].copy()
