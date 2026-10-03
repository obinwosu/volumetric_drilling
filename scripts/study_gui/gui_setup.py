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
        self.yaml_file = open(str(self.yaml_file_path), 'w')
        yaml.dump(self.yaml_data, self.yaml_file)
        self.yaml_file.close()

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
        self.yaml_file.close()


def main():
    gs = GUIConfiguration('gui_setup.yaml')
    gs.print_volumes_info()


if __name__ == "__main__":
    main()
