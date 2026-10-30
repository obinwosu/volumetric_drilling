"""Named microscope views shared with the simulator pause menu.

A view is the main camera relative to its home pose. Home is the ADF
camera: position (1.5, 0, 0), looking down -X, up +Z. The file stores
that offset as t and a quaternion qx qy qz qw. AMBF reads the same file.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

VIEWS_PATH = pathlib.Path(__file__).resolve().parents[2] / "ADF" / "world" / "canonical_views.yaml"
HOME_POSITION = np.array([1.5, 0.0, 0.0], dtype=np.float64)


def _quat_matrix(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_quat(rotation):
    """qx, qy, qz, qw. Matches _quat_matrix, with a non-negative w."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    if quat[3] < 0.0:
        quat = -quat
    return quat


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_list(value: str) -> list[float]:
    text = value.strip()
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    if not text.strip():
        return []
    return [float(part) for part in text.split(",")]


def load_views(path: pathlib.Path = VIEWS_PATH) -> list[dict]:
    views = []
    current = None
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return views
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line == "views:":
            continue
        if line.startswith("- "):
            if current and current.get("id"):
                views.append(current)
            current = {
                "id": "",
                "label": "",
                "laterality": "",
                "valid": False,
                "t": [0.0, 0.0, 0.0],
                "q": [0.0, 0.0, 0.0, 1.0],
            }
            line = line[2:].strip()
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "id":
            current["id"] = _unquote(value)
        elif key == "label":
            current["label"] = _unquote(value)
        elif key == "laterality":
            current["laterality"] = _unquote(value).lower()
        elif key == "valid":
            current["valid"] = value.lower() == "true"
        elif key == "t":
            numbers = _parse_list(value)
            if len(numbers) >= 3:
                current["t"] = numbers[:3]
        elif key == "q":
            numbers = _parse_list(value)
            if len(numbers) >= 4:
                current["q"] = numbers[:4]
    if current and current.get("id"):
        views.append(current)
    return views


def _yaml_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fmt(value: float) -> str:
    return f"{float(value):.8g}"


def save_views(views: list[dict], path: pathlib.Path = VIEWS_PATH) -> None:
    lines = [
        "# T_world_camera = T_world_home * T_home_view",
        "# Volume is never moved. Quaternion is qx qy qz qw.",
        "# Add new labeled slots here (valid: false) and overwrite them from the pause menu.",
        "views:",
    ]
    for view in views:
        label = view.get("label") or view["id"]
        laterality = view.get("laterality") or ""
        translation = view["t"]
        quat = view["q"]
        lines.append(f"  - id: {view['id']}")
        lines.append(f"    label: {_yaml_quote(label)}")
        lines.append(f"    laterality: {laterality}")
        lines.append(f"    valid: {'true' if view.get('valid') else 'false'}")
        lines.append(
            "    t: [" + ", ".join(_fmt(channel) for channel in translation) + "]"
        )
        lines.append("    q: [" + ", ".join(_fmt(channel) for channel in quat) + "]")
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def pose_from_view(view: dict):
    """World position and the scope basis. Local +X looks along -forward."""
    rotation = _quat_matrix(*view["q"])
    forward = -rotation[:, 0]
    up = rotation[:, 2]
    right = np.cross(forward, up)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-12)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    position = HOME_POSITION + np.asarray(view["t"], dtype=np.float64)
    return position, right, up, forward


def view_offset(position, right, up, forward):
    """t and q for the current microscope pose."""
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-12)
    right = np.asarray(right, dtype=np.float64)
    right = right - forward * float(np.dot(right, forward))
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    rotation = np.column_stack((-forward, right, up))
    translation = np.asarray(position, dtype=np.float64) - HOME_POSITION
    return translation, _matrix_quat(rotation)


def unique_id(label: str, taken) -> str:
    """A file id for a new view. Existing ids stay put when a view is renamed."""
    slug = []
    previous = False
    for character in label.strip().lower():
        if character.isalnum():
            slug.append(character)
            previous = False
        elif not previous:
            slug.append("_")
            previous = True
    base = "".join(slug).strip("_") or "view"
    used = set(taken)
    candidate = base
    number = 2
    while candidate in used:
        candidate = f"{base}_{number}"
        number += 1
    return candidate


def home_pose():
    return (
        HOME_POSITION.copy(),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 0.0]),
    )
