import yaml
import pathlib


class VolumeInfo:
    def __init__(self, name, adf_path, icon_path, description=""):
        self.name = name
        self.adf_path = adf_path
        self.icon_path = icon_path
        self.description = description or ""

    def print_info(self):
        print('ADF Path', self.adf_path)
        print('Icon Path', self.icon_path)
        print('Description', self.description)


class GUIParam:
    def __init__(self, id, val):
        self._id = id
        self._param = val

    def get_id(self):
        return self._id

    def get(self):
        return self._param

    def get_as_str(self):
        return str(self._param)

    def set(self, val):
        self._param = val


def _resolve_config_value(raw, *, as_path=True):
    """Resolve repo-relative paths; keep bare command names (PATH lookups) as-is."""
    text = str(raw).strip()
    if not text:
        return text
    expanded = pathlib.Path(text).expanduser()
    # Device nodes and existing files/dirs: resolve.
    if text.startswith("/dev/") or expanded.exists():
        return expanded.resolve() if as_path else str(expanded)
    # Absolute or explicit relative path components → resolve against cwd.
    if text.startswith("/") or text.startswith(".") or "/" in text or "\\" in text:
        return expanded.resolve()
    # Bare command name (e.g. ambf_simulator, pupil_capture)
    return text


# Advanced-tab simulation options. The defaults match the full lab setup.
DLSS_LEVELS = ("quality", "balanced", "performance", "ultra_performance")
SAMPLE_CHOICES = (1, 2, 4)
# 0 = the tuner's irrigation panel value (tuner_irrigation.json max_bounces).
BOUNCE_CHOICES = (0, 2, 4, 6, 8)
# Render presets. Suction is left out: it is about the second Touch, not the look.
PRESETS = {
    "fastest": {"dlss": "ultra_performance", "samples": 1, "bounces": 2, "irrigation": False, "bone_dust": False},
    "balanced": {"dlss": "performance", "samples": 1, "bounces": 0, "irrigation": True, "bone_dust": False},
    "beautiful": {"dlss": "quality", "samples": 2, "bounces": 8, "irrigation": True, "bone_dust": True},
}
SIM_OPTION_DEFAULTS = {"preset": "balanced", **PRESETS["balanced"], "suction": True}


def matching_preset(options):
    """The preset these options equal, else "custom"."""
    for name, values in PRESETS.items():
        if all(options.get(key) == value for key, value in values.items()):
            return name
    return "custom"


def _sim_options(raw):
    """Read the sim_options block; unknown or missing values fall back to the defaults."""
    raw = raw if isinstance(raw, dict) else {}
    options = dict(SIM_OPTION_DEFAULTS)
    if raw.get("dlss") in DLSS_LEVELS:
        options["dlss"] = raw["dlss"]
    if raw.get("samples") in SAMPLE_CHOICES:
        options["samples"] = raw["samples"]
    if raw.get("bounces") in BOUNCE_CHOICES:
        options["bounces"] = raw["bounces"]
    for key in ("irrigation", "bone_dust", "suction"):
        if isinstance(raw.get(key), bool):
            options[key] = raw[key]
    options["preset"] = matching_preset(options)
    return options


class GUIConfiguration:
    def __init__(self, setup_filename):
        self.setup_file_name = setup_filename
        self.yaml_file_path = pathlib.Path(self.setup_file_name)
        self.yaml_file = open(str(self.yaml_file_path), 'r')

        self.param_keys = ["ambf_executable",
                           "pupil_executable",
                           "recording_script_executable",
                           "recording_base_path",
                           "launch_file",
                           "footpedal_device"]
        # Optional absolute paths; empty string means skip.
        self.optional_path_keys = ["spacenav_plugin", "external_video_recording_plugin"]
        self.params = {}

        self.yaml_data = yaml.safe_load(self.yaml_file)

        for l in self.param_keys:
            self.params[l] = GUIParam(l, _resolve_config_value(self.yaml_data[l]))

        self.ambf_executable = self.params["ambf_executable"]
        self.pupil_executable = self.params["pupil_executable"]
        self.recording_script = self.params["recording_script_executable"]
        self.recording_base_path = self.params['recording_base_path']
        self.launch_file = self.params["launch_file"]
        self.footpedal_device = self.params["footpedal_device"]
        self.spacenav_plugin = str(self.yaml_data.get("spacenav_plugin", "") or "")
        self.external_video_recording_plugin = str(
            self.yaml_data.get("external_video_recording_plugin", "") or ""
        )

        self.sim_options = _sim_options(self.yaml_data.get("sim_options"))

        volumes_data = self.yaml_data['volumes']
        self.volumes_info = []
        for v in volumes_data:
            adf_path = pathlib.Path(self.yaml_data[v]['adf_path']).expanduser().resolve()
            icon_path = pathlib.Path(self.yaml_data[v]['icon_path']).expanduser().resolve()
            name = self.yaml_data[v]['name']
            description = self.yaml_data[v].get('description', '')
            new_volume_info = VolumeInfo(name, adf_path, icon_path, description)
            self.volumes_info.append(new_volume_info)

    def print_volumes_info(self):
        for v in self.volumes_info:
            print('-------')
            v.print_info()

    def print(self):
        print('Printing GUI Params:')
        for k,v in self.params.items():
            print('\t', k, v.get_as_str())

    def save(self):
        for k, v in self.params.items():
            self.yaml_data[k] = v.get_as_str()
            print(k, v.get_as_str())
        self.yaml_data["sim_options"] = dict(self.sim_options)
        self.yaml_file = open(str(self.yaml_file_path), 'w')
        yaml.dump(self.yaml_data, self.yaml_file)
        self.yaml_file.close()

    def save_value(self, key, value):
        """Write one top-level key to the file, leaving everything else in it as is
        (unlike save(), which also writes unsaved parameter and render changes).
        Only that key's line changes (or one line is appended), so the rest of the file
        keeps its order and formatting."""
        self.yaml_data[key] = value
        line = yaml.safe_dump({key: value}, default_flow_style=False)
        lines = self.yaml_file_path.read_text().splitlines(keepends=True)
        for index, text in enumerate(lines):
            if text.startswith(f"{key}:"):
                lines[index] = line
                break
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(line)
        self.yaml_file_path.write_text("".join(lines))

    def reload(self):
        self.yaml_file = open(str(self.yaml_file_path), 'r')
        self.yaml_data = yaml.safe_load(self.yaml_file)
        for l in self.param_keys:
            self.params[l] = GUIParam(l, _resolve_config_value(self.yaml_data[l]))

        self.ambf_executable = self.params["ambf_executable"]
        self.pupil_executable = self.params["pupil_executable"]
        self.recording_script = self.params["recording_script_executable"]
        self.recording_base_path = self.params['recording_base_path']
        self.launch_file = self.params["launch_file"]
        self.footpedal_device = self.params["footpedal_device"]
        self.spacenav_plugin = str(self.yaml_data.get("spacenav_plugin", "") or "")
        self.external_video_recording_plugin = str(
            self.yaml_data.get("external_video_recording_plugin", "") or ""
        )
        self.sim_options = _sim_options(self.yaml_data.get("sim_options"))
        self.yaml_file.close()


def main():
    gs = GUIConfiguration('gui_setup.yaml')
    gs.print_volumes_info()


if __name__ == "__main__":
    main()
