"""Advanced tab: start the simulator with a saved haptics preset.

Presets come from the haptics tuner (scripts/haptics_tuner/presets.json plus its
built-in ones). "Stock" leaves FIVRS_HAPTIC_TUNER unset, so the plugin uses its normal
forces. Studies and UR5 recordings always run on stock forces. The choice is saved as
`haptics_preset` in gui_setup.yaml and applies at the next Start Simulation.
"""
import json
import pathlib
import sys

from PyQt5 import QtWidgets

REPO = pathlib.Path(__file__).resolve().parents[2]
TUNER_DIR = REPO / "scripts" / "haptics_tuner"
if str(TUNER_DIR) not in sys.path:
    sys.path.insert(0, str(TUNER_DIR))
import tuning_shm as shm  # noqa: E402

CONFIG_KEY = "haptics_preset"
STOCK = "stock"
STOCK_LABEL = "Stock (today's haptics)"
TUNER_STOCK_NAME = "Today's haptics"  # the tuner's name for the same forces


def load_presets(path=TUNER_DIR / "presets.json"):
    """Name -> settings: the tuner's built-in presets, then the saved ones."""
    presets = {name: values for name, values in shm.BUILT_IN.items() if name != TUNER_STOCK_NAME}
    try:
        saved = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        saved = {}
    if isinstance(saved, dict):
        for name in sorted(saved):
            if isinstance(saved[name], dict) and name != STOCK:
                presets[name] = shm.clean(saved[name])
    return presets


def describe(values):
    direction = shm.BY_KEY["hum_direction"].choices[int(values["hum_direction"])].lower()
    hum = (f"hum {values['hum_amp']:.2f} N at {values['hum_hz']:.0f} Hz, {direction}"
           if values["hum_on"] >= 0.5 else "hum off")
    cutting = (f"cuts from {values['cut_start_n']:.2f} N, full at {values['cut_full_n']:.2f} N"
               if values["cut_gate"] >= 0.5 else "stock removal")
    return f"Stiffness {values['stiffness']:.2f}×, {hum}, {cutting}."


class _RefreshingCombo(QtWidgets.QComboBox):
    """Re-reads the presets when opened, so one saved in the tuner appears at once."""

    def __init__(self, refresh, parent=None):
        super().__init__(parent)
        self._refresh = refresh

    def showPopup(self):
        self._refresh()
        super().showPopup()


class HapticsPresetPicker:
    def __init__(self, configuration, print_info=print, presets_path=TUNER_DIR / "presets.json", link_factory=shm.Link):
        self.configuration = configuration
        self.print_info = print_info
        self.presets_path = presets_path
        self.link_factory = link_factory
        self.presets = {}
        self.combo = None
        self.summary = None

    def build_group(self, parent):
        box = QtWidgets.QGroupBox("Haptics", parent)
        grid = QtWidgets.QGridLayout(box)
        self.combo = _RefreshingCombo(self.refresh, box)
        self.combo.setMinimumWidth(320)
        self.combo.setToolTip("Forces for Main launches. Studies and UR5 recordings always use Stock.")
        self.summary = QtWidgets.QLabel(box)
        grid.addWidget(QtWidgets.QLabel("Preset:", box), 0, 0)
        grid.addWidget(self.combo, 0, 1)
        grid.addWidget(self.summary, 0, 2)
        grid.addWidget(QtWidgets.QLabel(
            "Applies at the next Start Simulation. Studies and UR5 recordings always use Stock.", box), 1, 0, 1, 3)
        grid.setColumnStretch(2, 1)
        self.refresh()
        self.combo.activated.connect(self._chosen)
        return box

    def saved_name(self):
        return str(self.configuration.yaml_data.get(CONFIG_KEY) or STOCK)

    def refresh(self):
        """Fill the list and show the saved choice."""
        self.presets = load_presets(self.presets_path)
        wanted = self.saved_name()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(STOCK_LABEL, STOCK)
        for name in self.presets:
            self.combo.addItem(name, name)
        if wanted != STOCK and wanted not in self.presets:
            self.combo.addItem(f"{wanted} (missing)", wanted)
        self.combo.setCurrentIndex(max(0, self.combo.findData(wanted)))
        self.combo.blockSignals(False)
        self._show_summary()

    def _show_summary(self):
        name = self.combo.currentData()
        if name == STOCK:
            self.summary.setText("The simulator's normal forces.")
        elif name in self.presets:
            self.summary.setText(describe(self.presets[name]))
        else:
            self.summary.setText("This preset no longer exists: the launch will use Stock.")

    def _chosen(self, *_):
        self.configuration.save_value(CONFIG_KEY, self.combo.currentData())
        self._show_summary()

    def prepare_launch(self, env, allowed=True):
        """Set up `env` for this launch. Returns the preset name, or None for stock."""
        env.remove("FIVRS_HAPTIC_TUNER")
        name = self.saved_name()
        if name == STOCK or not allowed:
            return None
        values = load_presets(self.presets_path).get(name)
        if values is None:
            self.print_info(f"WARNING! Haptics preset {name} not found: using Stock.")
            return None
        try:
            link = self.link_factory()
            link.force_enable = True
            link.write(values)
        except OSError as error:
            self.print_info(f"WARNING! Haptics preset could not be applied ({error}): using Stock.")
            return None
        env.insert("FIVRS_HAPTIC_TUNER", "1")
        self.print_info(f"Haptics preset: {name}. {describe(values)}")
        return name
