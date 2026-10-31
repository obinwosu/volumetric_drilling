from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment
from PyQt5.QtWidgets import QVBoxLayout, QButtonGroup, QFileDialog, QInputDialog, QMessageBox
import sys
from study_manager import StudyManager, RecordOptions
from gui_setup import GUIConfiguration
from user_profiles import UserProfileStore, DEFAULT_BURR_MM, VALID_BURR_SIZES
import datetime
from enum import Enum
import json
import os
import pathlib
import re
import yaml


class DialogType(Enum):
    FILE = 0
    FOLDER = 1


def wivrn_nvidia_extension_problem():
    """WiVRn Flatpak only uses NVIDIA when its GL extension matches the host driver.

    A mismatch makes the server start on the Intel GPU. AMBF still renders on
    NVIDIA, the OpenXR submit then crashes, and the Quest stays blank.
    """
    version_file = pathlib.Path("/proc/driver/nvidia/version")
    if not version_file.is_file():
        return None
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", version_file.read_text(errors="replace"))
    if not match:
        return None
    version = match.group(1)
    ext_id = "org.freedesktop.Platform.GL.nvidia-" + version.replace(".", "-")
    roots = (
        pathlib.Path("/var/lib/flatpak/runtime") / ext_id,
        pathlib.Path.home() / ".local/share/flatpak/runtime" / ext_id,
    )
    if any(path.exists() for path in roots):
        return None
    return (
        f"ERROR! WiVRn cannot use the NVIDIA GPU. Host driver is {version}, "
        f"but Flatpak runtime {ext_id} is not installed, so the server falls back "
        "to the Intel GPU, the sim crashes, and the Quest stays blank. "
        "Run this, then quit and reopen the WiVRn dashboard and reconnect the Quest:\n"
        f"  flatpak install -y flathub {ext_id}"
    )


def read_adf_laterality(adf_path):
    """Read laterality from anatomy ADF if present."""
    try:
        with open(adf_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    # Prefer explicit key on volume block(s).
    for key, val in data.items():
        if isinstance(val, dict) and "laterality" in val:
            lat = str(val["laterality"]).strip().lower()
            if lat in ("left", "right"):
                return lat
    if "laterality" in data:
        lat = str(data["laterality"]).strip().lower()
        if lat in ("left", "right"):
            return lat
    return None


def canonical_views_path():
    return pathlib.Path(__file__).resolve().parent / ".." / ".." / "ADF" / "world" / "canonical_views.yaml"


def load_canonical_views():
    """Return list of dicts with id, label, laterality, valid from canonical_views.yaml."""
    path = canonical_views_path()
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    views = data.get("views") or []
    out = []
    if not isinstance(views, list):
        return out
    for item in views:
        if not isinstance(item, dict):
            continue
        vid = str(item.get("id", "")).strip()
        if not vid:
            continue
        lat = str(item.get("laterality", "")).strip().lower()
        label = str(item.get("label", "")).strip() or vid
        out.append({
            "id": vid,
            "label": label,
            "laterality": lat,
            "valid": bool(item.get("valid", False)),
        })
    return out


MIDNIGHT_OCEAN_STYLE = """
QWidget {
    background-color: #0B1C2C;
    color: #E6EEF5;
    font-size: 18px;
}
QTabWidget::pane {
    border: 1px solid #1E3A4F;
    background-color: #0B1C2C;
}
QTabBar::tab {
    background-color: #12283A;
    color: #9BB0C3;
    padding: 12px 22px;
    font-size: 18px;
    border: 1px solid #1E3A4F;
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1E3A4F;
    color: #E6EEF5;
}
QGroupBox {
    background-color: #12283A;
    border: 1px solid #1E3A4F;
    border-radius: 6px;
    margin-top: 16px;
    padding-top: 14px;
    font-size: 18px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 8px;
    color: #9BB0C3;
    font-size: 18px;
}
QLabel {
    background-color: transparent;
    color: #E6EEF5;
}
QScrollArea, QAbstractScrollArea {
    background-color: #0F2433;
    border: 1px solid #1E3A4F;
}
QRadioButton, QCheckBox {
    background-color: transparent;
    color: #E6EEF5;
    spacing: 10px;
    font-size: 18px;
}
QRadioButton::indicator, QCheckBox::indicator {
    width: 22px;
    height: 22px;
    border: 1px solid #1E3A4F;
    border-radius: 3px;
    background-color: #0F2433;
}
QRadioButton::indicator:checked, QCheckBox::indicator:checked {
    background-color: #2F8FBF;
    border-color: #2F8FBF;
}
QCheckBox[loggingActive="true"] {
    color: #F0A8A8;
    font-weight: bold;
}
QCheckBox[loggingActive="true"]::indicator:checked {
    background-color: #C45C5C;
    border-color: #C45C5C;
}
QComboBox, QTextEdit, QPlainTextEdit, QLineEdit {
    background-color: #0F2433;
    color: #E6EEF5;
    border: 1px solid #1E3A4F;
    border-radius: 4px;
    padding: 8px;
    font-size: 18px;
    selection-background-color: #2F8FBF;
    min-height: 28px;
}
QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus, QLineEdit:focus {
    border: 1px solid #2F8FBF;
}
QComboBox::drop-down {
    border: none;
    width: 32px;
}
QComboBox QAbstractItemView {
    background-color: #0F2433;
    color: #E6EEF5;
    selection-background-color: #2F8FBF;
    border: 1px solid #1E3A4F;
}
QPushButton {
    background-color: #1E3A4F;
    color: #E6EEF5;
    border: 1px solid #2F8FBF;
    border-radius: 4px;
    padding: 10px 18px;
    font-size: 18px;
    min-height: 36px;
}
QPushButton:hover {
    background-color: #2A4F68;
}
QPushButton:pressed {
    background-color: #163042;
}
QPushButton#button_start_simulation {
    background-color: #1F6F8B;
    color: #E6EEF5;
    border: 1px solid #2F8FBF;
    font-size: 22px;
    font-weight: bold;
    min-height: 72px;
}
QPushButton#button_start_simulation:hover {
    background-color: #2589A8;
}
QPushButton#button_start_simulation[simRunning="true"] {
    background-color: #C45C5C;
    border-color: #E08080;
}
QPushButton#button_start_simulation[simRunning="true"]:hover {
    background-color: #D46A6A;
}
QMessageBox, QInputDialog {
    background-color: #12283A;
    color: #E6EEF5;
}
"""


class Ui(QtWidgets.QWidget):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi("layout.ui", self)
        self.setStyleSheet(MIDNIGHT_OCEAN_STYLE)

        self.gui_configuration = GUIConfiguration("gui_setup.yaml")
        self.study_manager = StudyManager(
            self.gui_configuration.ambf_executable.get(),
            self.gui_configuration.pupil_executable.get(),
            self.gui_configuration.recording_script.get(),
        )
        self.profile_store = UserProfileStore()
        self.active_volume_adf = ""
        self.active_volume_name = ""
        self.active_volume_info = None
        self.record_options = RecordOptions()
        self._loading_profile = False
        self._updating_log_checkbox = False

        # --- Anatomy list + preview ---
        self.scroll_area = self.findChild(QtWidgets.QScrollArea, "scrollArea")
        self.label_preview_icon = self.findChild(QtWidgets.QLabel, "label_preview_icon")
        self.label_preview_description = self.findChild(QtWidgets.QLabel, "label_preview_description")
        self.label_preview_laterality = self.findChild(QtWidgets.QLabel, "label_preview_laterality")
        self.combo_canonical_view = self.findChild(QtWidgets.QComboBox, "combo_canonical_view")
        self._canonical_views = load_canonical_views()

        self.anatomy_button_group = QButtonGroup(self)
        anatomy_container = QtWidgets.QWidget()
        anatomy_layout = QVBoxLayout(anatomy_container)
        anatomy_layout.setAlignment(Qt.AlignTop)
        for i, vinfo in enumerate(self.gui_configuration.volumes_info):
            radio = QtWidgets.QRadioButton(vinfo.name)
            radio.volume_name = vinfo.name
            radio.volume_adf = str(vinfo.adf_path)
            radio.volume_info = vinfo
            radio.toggled.connect(self.radio_button_volume_selection)
            self.anatomy_button_group.addButton(radio)
            anatomy_layout.addWidget(radio)
            if i == 0:
                radio.setChecked(True)
        anatomy_layout.addStretch(1)
        self.scroll_area.setWidget(anatomy_container)
        self.scroll_area.setWidgetResizable(True)

        # --- Burr radios ---
        self.radio_burr_1 = self.findChild(QtWidgets.QRadioButton, "radio_burr_1")
        self.radio_burr_2 = self.findChild(QtWidgets.QRadioButton, "radio_burr_2")
        self.radio_burr_4 = self.findChild(QtWidgets.QRadioButton, "radio_burr_4")
        self.radio_burr_6 = self.findChild(QtWidgets.QRadioButton, "radio_burr_6")
        self.burr_button_group = QButtonGroup(self)
        for btn in (self.radio_burr_1, self.radio_burr_2, self.radio_burr_4, self.radio_burr_6):
            self.burr_button_group.addButton(btn)
        self.radio_burr_1.burr_mm = 1
        self.radio_burr_2.burr_mm = 2
        self.radio_burr_4.burr_mm = 4
        self.radio_burr_6.burr_mm = 6
        self.set_starting_burr_mm(DEFAULT_BURR_MM)

        # --- Profiles ---
        self.combo_user_profile = self.findChild(QtWidgets.QComboBox, "combo_user_profile")
        self.button_new_profile = self.findChild(QtWidgets.QPushButton, "button_new_profile")
        self.button_save_profile = self.findChild(QtWidgets.QPushButton, "button_save_profile")
        self.button_new_profile.clicked.connect(self.pressed_new_profile)
        self.button_save_profile.clicked.connect(self.pressed_save_profile)
        self.combo_user_profile.currentIndexChanged.connect(self.profile_selection_changed)
        self.reload_profiles_combo()

        # --- Launch / recording ---
        self.button_start_simulation = self.findChild(QtWidgets.QPushButton, "button_start_simulation")
        self.button_start_simulation.setProperty("simRunning", False)
        self.button_start_simulation.clicked.connect(self.pressed_start_simulation)

        self.check_stream_stereo = self.findChild(QtWidgets.QCheckBox, "check_stream_stereo")
        self.check_stream_depth = self.findChild(QtWidgets.QCheckBox, "check_stream_depth")
        self.check_openvr_hmd = self.findChild(QtWidgets.QCheckBox, "check_openvr_hmd")
        self.check_openxr_eyepiece = self.findChild(QtWidgets.QCheckBox, "check_openxr_eyepiece")
        self.check_rtx_view = self.findChild(QtWidgets.QCheckBox, "check_rtx_view")
        self.check_video_record = self.findChild(QtWidgets.QCheckBox, "check_video_record")
        self.check_pupil_capture = self.findChild(QtWidgets.QCheckBox, "check_pupil_capture")
        self.check_pupil_capture.toggled.connect(self.pupil_capture_toggled)

        self.check_log_study_data = self.findChild(QtWidgets.QCheckBox, "check_log_study_data")
        self.check_log_study_data.setProperty("loggingActive", False)
        self.check_log_study_data.toggled.connect(self.log_study_data_toggled)

        # --- Advanced controls ---
        self.button_reset_drill = self.findChild(QtWidgets.QPushButton, "button_reset_drill")
        self.button_reset_drill.clicked.connect(self.study_manager.reset_drill)
        self.button_reset_volume = self.findChild(QtWidgets.QPushButton, "button_reset_volume")
        self.button_reset_volume.clicked.connect(self.study_manager.reset_volume)
        self.button_toggle_shadows = self.findChild(QtWidgets.QPushButton, "button_toggle_shadows")
        self.button_toggle_shadows.clicked.connect(self.study_manager.toggle_shadows)
        self.button_toggle_vol_smooth = self.findChild(QtWidgets.QPushButton, "button_toggle_vol_smooth")
        self.button_toggle_vol_smooth.clicked.connect(self.study_manager.toggle_volume_smoothening)

        self.textEdit_info = self.findChild(QtWidgets.QPlainTextEdit, "textEdit_info")
        self.textEdit_debug = self.findChild(QtWidgets.QPlainTextEdit, "textEdit_debug")

        self.gui_param_dialogs = {}
        self.connect_gui_param_to_dialog(
            "ambf_executable", self.gui_configuration.ambf_executable.get_id(), DialogType.FILE
        )
        self.connect_gui_param_to_dialog(
            "pupil_capture_executable", self.gui_configuration.pupil_executable.get_id(), DialogType.FILE
        )
        self.connect_gui_param_to_dialog(
            "launch_file", self.gui_configuration.launch_file.get_id(), DialogType.FILE
        )
        self.connect_gui_param_to_dialog(
            "recording_script_executable", self.gui_configuration.recording_script.get_id(), DialogType.FILE
        )
        self.connect_gui_param_to_dialog(
            "recording_base_path", self.gui_configuration.recording_base_path.get_id(), DialogType.FOLDER
        )
        self.connect_gui_param_to_dialog(
            "footpedal_device", self.gui_configuration.footpedal_device.get_id(), DialogType.FILE
        )

        self.button_save_configuration = self.findChild(QtWidgets.QPushButton, "pushButton_save_configuration")
        self.button_save_configuration.clicked.connect(self.save_configuration)
        self.button_reload_configuration = self.findChild(QtWidgets.QPushButton, "pushButton_reload_configuration")
        self.button_reload_configuration.clicked.connect(self.reload_configuration)

        self._recording_study = False
        self._ambf_process = QProcess()
        self._rtx_process = QProcess()
        self._fluid_process = QProcess()
        self._rtx_process.readyReadStandardOutput.connect(self._rtx_output)
        self._rtx_process.finished.connect(self._rtx_finished)
        self._fluid_process.readyReadStandardOutput.connect(self._fluid_output)
        self._fluid_process.finished.connect(self._fluid_finished)
        self._ambf_process.readyReadStandardOutput.connect(self.handle_stdout)
        self._ambf_process.readyReadStandardError.connect(self.handle_stderr)
        self._ambf_process.finished.connect(self._simulation_closed)
        self._ambf_process.errorOccurred.connect(self._ambf_process_error)
        self._pupil_process = QProcess()
        self._recording_process = QProcess()
        self._current_args = []

        self.place_on_pc_monitor(0)
        self.show()

    def place_on_pc_monitor(self, screen_index=0):
        """Move the study GUI onto a normal PC monitor, not the VR headset display."""
        screens = QtWidgets.QApplication.screens()
        if not screens:
            return
        idx = min(max(screen_index, 0), len(screens) - 1)
        geo = screens[idx].availableGeometry()
        width = min(1800, max(1200, geo.width() - 80))
        height = min(1200, max(900, geo.height() - 80))
        self.resize(width, height)
        x = geo.x() + max(20, (geo.width() - self.width()) // 2)
        y = geo.y() + max(20, (geo.height() - self.height()) // 2)
        self.move(x, y)
        print(
            f"Study GUI placed on screen {idx}: {screens[idx].name()} "
            f"at ({x}, {y}) within {geo.width()}x{geo.height()}"
        )

    # ---- Profiles ----
    def reload_profiles_combo(self, select_name=None):
        self._loading_profile = True
        self.combo_user_profile.clear()
        self.combo_user_profile.addItem("(select user)", None)
        profiles = self.profile_store.list_profiles()
        selected_idx = 0
        for i, profile in enumerate(profiles, start=1):
            self.combo_user_profile.addItem(profile.name, profile)
            if select_name and profile.name == select_name:
                selected_idx = i
        self.combo_user_profile.setCurrentIndex(selected_idx)
        self._loading_profile = False
        if selected_idx > 0:
            self.profile_selection_changed(selected_idx)

    def current_profile(self):
        return self.combo_user_profile.currentData()

    def active_participant_name(self):
        profile = self.current_profile()
        return profile.name.strip() if profile else ""

    def profile_selection_changed(self, index):
        if self._loading_profile:
            return
        profile = self.combo_user_profile.itemData(index)
        if profile is None:
            return
        self.set_starting_burr_mm(profile.starting_burr_mm)
        self.print_info(f"Loaded profile '{profile.name}' (burr {profile.starting_burr_mm} mm)")

    def pressed_new_profile(self):
        name, ok = QInputDialog.getText(self, "New User Profile", "Participant / user name:")
        if not ok or not name.strip():
            return
        try:
            profile = self.profile_store.create(name.strip(), self.get_starting_burr_mm())
        except Exception as exc:
            QMessageBox.warning(self, "Profile Error", str(exc))
            return
        self.reload_profiles_combo(select_name=profile.name)
        self.print_info(f"Created profile '{profile.name}'")

    def pressed_save_profile(self):
        profile = self.current_profile()
        if profile is None:
            self.print_info("ERROR! Select or create a user profile first")
            return
        profile.starting_burr_mm = self.get_starting_burr_mm()
        self.profile_store.save(profile)
        self.print_info(
            f"Saved profile '{profile.name}' (burr {profile.starting_burr_mm} mm)"
        )

    # ---- Burr ----
    def set_starting_burr_mm(self, mm):
        if mm not in VALID_BURR_SIZES:
            mm = DEFAULT_BURR_MM
        for btn in (self.radio_burr_1, self.radio_burr_2, self.radio_burr_4, self.radio_burr_6):
            if btn.burr_mm == mm:
                btn.setChecked(True)
                return

    def get_starting_burr_mm(self):
        for btn in (self.radio_burr_1, self.radio_burr_2, self.radio_burr_4, self.radio_burr_6):
            if btn.isChecked():
                return btn.burr_mm
        return DEFAULT_BURR_MM

    # ---- Config dialogs ----
    def connect_gui_param_to_dialog(self, name, param_id, dialog_type: DialogType):
        exec_file_text_edit = self.findChild(QtWidgets.QTextEdit, "textEdit_" + name)
        exec_file_btn = self.findChild(QtWidgets.QPushButton, "pushButton_" + name)
        if dialog_type == DialogType.FILE:
            exec_file_btn.clicked.connect(lambda: self.file_dialog(exec_file_text_edit, param_id))
        elif dialog_type == DialogType.FOLDER:
            exec_file_btn.clicked.connect(lambda: self.folder_dialog(exec_file_text_edit, param_id))

        exec_file_text_edit.setText(self.gui_configuration.params[param_id].get_as_str())
        self.gui_param_dialogs[param_id] = exec_file_text_edit
        return exec_file_text_edit

    def file_dialog(self, text_edit, param_id):
        file, check = QFileDialog.getOpenFileName(
            None,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;Python Files (*.py);;Text Files (*.txt)",
        )
        if check:
            self.gui_configuration.params[param_id].set(file)
            text_edit.setText(file)

    def folder_dialog(self, text_edit, param_id):
        folder = QFileDialog.getExistingDirectory(None, "QFileDialog.getExistingDirectory()", "")
        self.gui_configuration.params[param_id].set(folder)
        text_edit.setText(folder)

    def save_configuration(self):
        for param_id, text_edit in self.gui_param_dialogs.items():
            value = text_edit.toPlainText().strip()
            self.gui_configuration.params[param_id].set(value)
        self.gui_configuration.save()
        self.print_info("Configuration saved")

    def reload_configuration(self):
        print("Reloading Configuration")
        self.gui_configuration.reload()
        for k, v in self.gui_param_dialogs.items():
            v.setText(self.gui_configuration.params[k].get_as_str())

    # ---- Simulation launch ----
    def pressed_start_simulation(self):
        if self._ambf_process.state() != QProcess.Running:
            if not self.active_volume_adf:
                self.print_info("ERROR! Select an anatomy before starting")
                return
            self._current_args = []
            # Base: drills(0) + gaze_marker(7). World main_camera is the on-screen view.
            # OpenXR eyepiece (WiVRn) = ADF 15. Legacy OpenVR/SteamVR = ADF 14.
            use_openxr = bool(
                self.check_openxr_eyepiece is not None
                and self.check_openxr_eyepiece.isChecked()
            )
            use_openvr = bool(
                not use_openxr
                and self.check_openvr_hmd is not None
                and self.check_openvr_hmd.isChecked()
            )
            launch_file_adf_indices = "0,7"
            if use_openxr:
                launch_file_adf_indices = launch_file_adf_indices + ",15"
            elif use_openvr:
                launch_file_adf_indices = launch_file_adf_indices + ",14"
            use_hmd = use_openxr or use_openvr
            use_wivrn = use_openxr
            if not use_hmd and self.check_stream_depth.isChecked():
                launch_file_adf_indices = launch_file_adf_indices + ",4"
            if not use_hmd and self.check_stream_stereo.isChecked():
                launch_file_adf_indices = launch_file_adf_indices + ",5"
            args = [
                "--launch_file",
                str(self.gui_configuration.launch_file.get()),
                "-l",
                launch_file_adf_indices,
                "-a",
                self.active_volume_adf,
                "--nt",
                "2",
                "--fp",
                str(self.gui_configuration.footpedal_device.get()),
                "--sb",
                str(self.get_starting_burr_mm()),
                "--anatomy_adf",
                self.active_volume_adf,
                "--session_user",
                self.active_participant_name() or "(none)",
                "--session_anatomy",
                self.active_volume_name or "",
                "--startup_view",
                self.selected_startup_view_id(),
            ]
            use_rtx = bool(self.check_rtx_view is not None and self.check_rtx_view.isChecked())
            if use_rtx:
                args.extend(["--rtx_view", "1"])
            self._current_args = args
            spacenav = getattr(self.gui_configuration, "spacenav_plugin", "") or ""
            if spacenav and pathlib.Path(spacenav).expanduser().exists():
                self.add_plugin(str(pathlib.Path(spacenav).expanduser().resolve()))
            if self.check_video_record.isChecked():
                # In-plugin VideoRecordingController handles recording; optional
                # external AMBF video plugin can still be attached if configured.
                ext_vid = getattr(self.gui_configuration, "external_video_recording_plugin", "") or ""
                if ext_vid and pathlib.Path(ext_vid).expanduser().exists():
                    self.add_plugin(str(pathlib.Path(ext_vid).expanduser().resolve()))
            if self.check_pupil_capture.isChecked() and self._pupil_process.state() != QProcess.Running:
                self.start_pupil_capture()
            print("--fp", str(self.gui_configuration.footpedal_device.get()))
            print(f"Starting simulation with arguments: {self._current_args}")
            # Ensure Geomagic/OpenHaptics env is present even if the GUI was
            # started without scripts/setup_haptic_env.sh.
            env = QProcessEnvironment.systemEnvironment()
            if not env.value("GTDD_HOME"):
                env.insert("GTDD_HOME", "/usr/share/3DSystems")
            if not env.value("OH_SDK_BASE"):
                env.insert("OH_SDK_BASE", "/opt/OpenHaptics/Developer/3.4-0")
            ld = env.value("LD_LIBRARY_PATH") or ""
            for p in ("/usr/local/lib", "/usr/lib"):
                if p not in ld.split(":"):
                    ld = f"{p}:{ld}" if ld else p
            # Rebuilt libhdPhantom (opens Phantom2) must win over /usr/local/lib.
            home_lib = str(pathlib.Path.home() / ".local" / "lib")
            if home_lib not in ld.split(":"):
                ld = f"{home_lib}:{ld}" if ld else home_lib
            env.insert("LD_LIBRARY_PATH", ld)
            # AMBF + VR on NVIDIA. Avoid PRIME offload when the desktop is
            # already driven by NVIDIA (causes see-through/frozen GL windows).
            if use_hmd:
                env.insert("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
                env.insert(
                    "VK_DRIVER_FILES",
                    "/usr/share/vulkan/icd.d/nvidia_icd.json",
                )
                env.remove("__NV_PRIME_RENDER_OFFLOAD")
                env.remove("__VK_LAYER_NV_optimus")
            if use_wivrn:
                gpu_problem = wivrn_nvidia_extension_problem()
                if gpu_problem:
                    self.print_info(gpu_problem)
                    return
                # Prefer active runtime written by WiVRn dashboard when connected.
                runtime = pathlib.Path.home() / ".config/openxr/1/active_runtime.json"
                if runtime.is_file():
                    env.insert("XR_RUNTIME_JSON", str(runtime.resolve()))
                else:
                    self.print_info(
                        "ERROR! No ~/.config/openxr/1/active_runtime.json — "
                        "start WiVRn and connect the Quest first"
                    )
                    return
                ipc = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "wivrn" / "comp_ipc"
                if not ipc.exists():
                    self.print_info(
                        f"WARNING! WiVRn IPC missing ({ipc}) — is the WiVRn server running?"
                    )
            # ADF paths in launch.yaml are relative to the repo root.
            repo_root = pathlib.Path(self.gui_configuration.launch_file.get()).expanduser().resolve().parent
            self._ambf_process.setWorkingDirectory(str(repo_root))
            self._ambf_process.setProcessEnvironment(env)
            if use_openxr:
                mode = "OpenXR eyepiece (WiVRn)"
            elif use_openvr:
                mode = "OpenVR legacy"
            else:
                mode = "desktop"
            ambf_bin = str(self.gui_configuration.ambf_executable.get())
            self.print_info(f"Starting AMBF [{mode}] cwd={repo_root} -l {launch_file_adf_indices}")
            if use_wivrn:
                self.print_info(
                    f"XR_RUNTIME_JSON={env.value('XR_RUNTIME_JSON') or '(unset)'} — "
                    "Quest WiVRn must show Connection ready before HMD can stream"
                )
            self._ambf_process.start(ambf_bin, self._current_args)
            if use_rtx:
                self._start_rtx_view(repo_root)
            if not self._ambf_process.waitForStarted(5000):
                self.print_info(f"ERROR! Failed to start AMBF: {ambf_bin}")
                if self._rtx_process.state() != QProcess.NotRunning:
                    self._rtx_process.terminate()
                self.button_start_simulation.setText("Start Simulation")
                self._set_sim_running_style(False)
                return
            if use_rtx:
                self._start_fluid(repo_root)
            self.button_start_simulation.setText("Close Simulation")
            self._set_sim_running_style(True)
        else:
            if self._recording_study:
                self.print_info("Must stop logging study data first!")
            else:
                self._ambf_process.close()

    def _start_rtx_view(self, repo_root):
        venv_python = pathlib.Path(repo_root) / "middle_ear_newton" / ".venv" / "bin" / "python"
        script = pathlib.Path(repo_root) / "scripts" / "rtx_view" / "fivrs_rtx_view.py"
        if not venv_python.is_file() or not script.is_file():
            self.print_info(
                "ERROR! RTX bone view needs middle_ear_newton/.venv and scripts/rtx_view/fivrs_rtx_view.py"
            )
            return
        if self._rtx_process.state() != QProcess.NotRunning:
            self._rtx_process.terminate()
            self._rtx_process.waitForFinished(2000)
        command = (
            "set +u && "
            "source /opt/ros/jazzy/setup.bash && "
            "source \"$HOME/ros_ambf_ws/install/setup.bash\" && "
            "export DISPLAY=\"${DISPLAY:-:1}\" && "
            "export __GLX_VENDOR_LIBRARY_NAME=\"${__GLX_VENDOR_LIBRARY_NAME:-nvidia}\" && "
            "export VK_DRIVER_FILES=\"${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}\" && "
            f"exec \"{venv_python}\" \"{script}\" --anatomy-adf \"{self.active_volume_adf}\""
        )
        self._rtx_process.setWorkingDirectory(str(repo_root))
        self._rtx_process.setProcessChannelMode(QProcess.MergedChannels)
        self._rtx_process.start("bash", ["-lc", command])
        if not self._rtx_process.waitForStarted(5000):
            self.print_info(f"ERROR! RTX bone view did not start: {self._rtx_process.errorString()}")
            return
        self.print_info("Starting RTX bone view")

    def _start_fluid(self, repo_root):
        """Maelstrom jet for the RTX window. AMBF and the view are already up."""
        venv_python = pathlib.Path(repo_root) / "middle_ear_newton" / ".venv" / "bin" / "python"
        script = pathlib.Path(repo_root) / "scripts" / "fluid" / "live_sph.py"
        if not venv_python.is_file() or not script.is_file():
            self.print_info("ERROR! Fluid jet needs middle_ear_newton/.venv and scripts/fluid/live_sph.py")
            return
        if self._fluid_process.state() != QProcess.NotRunning:
            self._stop_fluid()
        adf = self.active_volume_adf
        command = (
            "set +u && "
            f"exec \"{venv_python}\" \"{script}\" --no-ambf --no-rtx --measure-s 0 "
            f"--anatomy-adf \"{adf}\""
        )
        self._fluid_process.setWorkingDirectory(str(repo_root))
        self._fluid_process.setProcessChannelMode(QProcess.MergedChannels)
        self._fluid_process.start("bash", ["-lc", command])
        if not self._fluid_process.waitForStarted(5000):
            self.print_info(f"ERROR! Fluid jet did not start: {self._fluid_process.errorString()}")
            return
        self.print_info("Starting irrigation jet. It runs while the drill is on.")

    def _stop_fluid(self):
        if self._fluid_process.state() == QProcess.NotRunning:
            return
        self._fluid_process.terminate()
        if not self._fluid_process.waitForFinished(3000):
            self._fluid_process.kill()
            self._fluid_process.waitForFinished(1000)

    def add_plugin(self, plugin_path):
        if "--plugins" in self._current_args:
            plugin_index = self._current_args.index("--plugins")
            current_plugins = self._current_args[plugin_index + 1]
            if plugin_path not in current_plugins:
                self._current_args[plugin_index + 1] += f",{plugin_path}"
        else:
            self._current_args.extend(["--plugins", plugin_path])

    def _refresh_widget_style(self, widget):
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def _set_sim_running_style(self, running):
        self.button_start_simulation.setProperty("simRunning", running)
        self._refresh_widget_style(self.button_start_simulation)

    def _set_logging_active_style(self, active):
        self.check_log_study_data.setProperty("loggingActive", active)
        self._refresh_widget_style(self.check_log_study_data)

    def _rtx_output(self):
        msg = bytes(self._rtx_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not msg:
            return
        self.print_debug(msg if msg.endswith("\n") else msg + "\n")
        for line in msg.splitlines():
            if any(token in line for token in ("Traceback", "Error", "ERROR", "Exception", "did not publish")):
                self.print_info(line)

    def _rtx_finished(self, code, status):
        if code not in (0, None) and self._ambf_process.state() == QProcess.Running:
            self.print_info(f"RTX bone view exited ({code}). See the Debug pane.")

    def _fluid_output(self):
        msg = bytes(self._fluid_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not msg:
            return
        self.print_debug(msg if msg.endswith("\n") else msg + "\n")
        for line in msg.splitlines():
            if any(token in line for token in ("Traceback", "Error", "ERROR", "Exception", "exited", "DFSPH is live")):
                self.print_info(line)

    def _fluid_finished(self, code, status):
        if code not in (0, None) and self._ambf_process.state() == QProcess.Running:
            self.print_info(f"Irrigation jet exited ({code}). See the Debug pane.")

    def _simulation_closed(self, exit_code=0, exit_status=None):
        if self._recording_study:
            self._stop_study_logging(save_metadata=True)
            self._updating_log_checkbox = True
            self.check_log_study_data.setChecked(False)
            self._updating_log_checkbox = False
        if exit_code not in (0, None) and exit_status != QProcess.CrashExit:
            self.print_info(f"AMBF exited with code {exit_code}")
        if exit_status == QProcess.CrashExit:
            self.print_info(
                "AMBF crashed — check Debug pane for OpenXR/WiVRn errors "
                "(session READY / xrCreateSession / BadWindow)"
            )
        self.button_start_simulation.setText("Start Simulation")
        self._set_sim_running_style(False)
        if self._rtx_process.state() != QProcess.NotRunning:
            self._rtx_process.terminate()
        self._stop_fluid()

    def _ambf_process_error(self, error):
        self.print_info(f"ERROR! AMBF process: {self._ambf_process.errorString()} ({error})")

    def pupil_capture_toggled(self, checked):
        if checked and self._pupil_process.state() != QProcess.Running:
            self.start_pupil_capture()

    def start_pupil_capture(self):
        try:
            print("Launching ", self.gui_configuration.pupil_executable.get())
            self._pupil_process.start(str(self.gui_configuration.pupil_executable.get()))
            self.print_info("Started Pupil Capture")
        except Exception as e:
            self.print_info("ERROR! Cant launch Pupil Capture")
            print(e)

    def update_preview(self, vinfo):
        if vinfo is None:
            self.label_preview_icon.setText("No preview")
            self.label_preview_icon.setPixmap(QPixmap())
            self.label_preview_description.setText("")
            self.label_preview_laterality.setText("Laterality: —")
            self.refresh_canonical_view_combo(None)
            return
        pixmap = QPixmap(str(vinfo.icon_path))
        if not pixmap.isNull():
            self.label_preview_icon.setPixmap(
                pixmap.scaled(
                    self.label_preview_icon.width(),
                    self.label_preview_icon.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            self.label_preview_icon.setText("")
        else:
            self.label_preview_icon.setPixmap(QPixmap())
            self.label_preview_icon.setText("(no icon)")
        self.label_preview_description.setText(vinfo.description or "")
        lat = read_adf_laterality(vinfo.adf_path)
        if lat:
            self.label_preview_laterality.setText(f"Laterality: {lat.upper()}")
        else:
            self.label_preview_laterality.setText("Laterality: unknown")
        self.refresh_canonical_view_combo(lat)

    def refresh_canonical_view_combo(self, laterality):
        combo = self.combo_canonical_view
        if combo is None:
            return
        prev = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Default (laterality)", "default")
        combo.addItem("Home", "home")
        lat = (laterality or "").strip().lower()
        for view in self._canonical_views:
            vlat = view.get("laterality") or ""
            if lat in ("left", "right") and vlat in ("left", "right") and vlat != lat:
                continue
            suffix = "" if view.get("valid") else " [empty]"
            combo.addItem(view["label"] + suffix, view["id"])
        idx = combo.findData(prev) if prev else -1
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def selected_startup_view_id(self):
        combo = self.combo_canonical_view
        if combo is None:
            return "default"
        data = combo.currentData()
        return data if data else "default"

    def radio_button_volume_selection(self):
        button = self.sender()
        if button.isChecked():
            self.print_info("Active Volume is " + button.volume_name)
            self.active_volume_adf = button.volume_adf
            self.active_volume_name = button.volume_name
            self.active_volume_info = getattr(button, "volume_info", None)
            self.update_preview(self.active_volume_info)

    def is_ready_to_record(self):
        ready = True
        if self.active_participant_name() == "":
            self.print_info("ERROR! Please select or create a user profile. Ignoring log request")
            ready = False
        if self._ambf_process.state() != QProcess.Running:
            self.print_info("ERROR! Please run simulation before logging. Ignoring log request!")
            ready = False

        if ready and self._pupil_process.state() != QProcess.Running:
            self.print_info("WARNING! Pupil capture not initialized!")
            dlg = QtWidgets.QMessageBox(self)
            dlg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            dlg.setText("Pupil capture not running. No pupil data will be recorded. OK?")
            button = dlg.exec()
            if button == QtWidgets.QMessageBox.Cancel:
                ready = False

        self.print_info("INFO! Ready to log study data? " + str(ready))
        return ready

    def get_record_options(self):
        record_options = RecordOptions()
        base_path = str(self.gui_configuration.recording_base_path.get())
        participant_name = "/" + self.active_participant_name()
        date_time = "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        record_options.path = base_path + participant_name + date_time
        record_options.simulator_data = True
        record_options.pupil_data = self._pupil_process.state() == QProcess.Running
        return record_options

    def save_metadata(self):
        notes, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Input Dialog", "Enter notes:")
        metadata = {
            "participant_name": self.active_participant_name(),
            "volume_adf": self.active_volume_adf,
            "volume_name": self.active_volume_name,
            "starting_burr_mm": self.get_starting_burr_mm(),
            "notes": notes,
        }
        metadata_path = self.record_options.path + "/metadata.json"
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile, indent=4)

    def _start_study_logging(self):
        self.record_options = self.get_record_options()
        self.study_manager.start_recording(self.record_options)
        self._recording_study = True
        self._set_logging_active_style(True)
        self.print_info("Study data logging started")

    def _stop_study_logging(self, save_metadata=True):
        if not self._recording_study:
            return
        self.study_manager.stop_recording(self.record_options)
        self._recording_study = False
        self._set_logging_active_style(False)
        if save_metadata:
            self.save_metadata()
        self.print_info("Study data logging stopped")

    def log_study_data_toggled(self, checked):
        if self._updating_log_checkbox:
            return
        if checked:
            if not self.is_ready_to_record():
                self._updating_log_checkbox = True
                self.check_log_study_data.setChecked(False)
                self._updating_log_checkbox = False
                return
            self._start_study_logging()
        else:
            if self._recording_study:
                self._stop_study_logging(save_metadata=True)

    def closeEvent(self, event):
        self.print_info("Terminate Called")
        if self._recording_study:
            self._stop_study_logging(save_metadata=True)
        self._stop_fluid()
        self._ambf_process.close()
        self._pupil_process.close()
        self.study_manager.close()
        print("GOOD BYE")

    def get_time_as_str(self):
        return "[" + datetime.datetime.now().strftime("%H:%M:%S") + "] - "

    def print_info(self, msg):
        self.textEdit_info.insertPlainText(self.get_time_as_str() + msg + "\n")

    def print_debug(self, msg):
        self.textEdit_debug.insertPlainText(self.get_time_as_str() + msg)

    def handle_stderr(self):
        msg = bytes(self._ambf_process.readAllStandardError()).decode("utf-8")
        self.print_debug(msg)

    def handle_stdout(self):
        msg = bytes(self._ambf_process.readAllStandardOutput()).decode("utf-8")
        self.print_debug(msg)


app = QtWidgets.QApplication(sys.argv)
app.setApplicationName("FIVRS")
app.setDesktopFileName("fivrs")
app.setStyleSheet(MIDNIGHT_OCEAN_STYLE)
_app_font = QFont()
_app_font.setPointSize(14)
app.setFont(_app_font)
window = Ui()
app.exec_()
