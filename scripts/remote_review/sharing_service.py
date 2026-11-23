"""Runs the review site as a systemd user service, so it does not depend on the launcher.

Start sharing enables the service (it returns at each login); Stop sharing disables it.
"""
from __future__ import annotations

import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
UNIT = "fivrs-remote-review.service"
PYTHON = "/usr/bin/python3"


def unit_path() -> pathlib.Path:
    config = pathlib.Path(os.environ.get("XDG_CONFIG_HOME") or pathlib.Path.home() / ".config")
    return config / "systemd" / "user" / UNIT


def _quoted(value) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{text}"'


def unit_text(base, port: int) -> str:
    return (
        "[Unit]\n"
        "Description=FIVRS remote case review\n\n"
        "[Service]\n"
        f"ExecStart={PYTHON} {_quoted(HERE / 'review_server.py')} --base {_quoted(base)} --port {int(port)}\n"
        "Restart=on-failure\n"
        "RestartSec=3\n"
        "NoNewPrivileges=true\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=20)


def _checked(*args) -> None:
    result = run(*args)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "systemctl failed").strip())


def start(base, port: int) -> None:
    path = unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(unit_text(base, port))
    partial.replace(path)
    _checked("daemon-reload")
    _checked("enable", UNIT)
    _checked("restart", UNIT)


def stop() -> None:
    if unit_path().exists():
        _checked("disable", "--now", UNIT)
