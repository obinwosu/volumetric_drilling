"""Pause card view list and the command the simulator reads."""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pause_menu


def test_right_case_hides_left_views_and_keeps_home():
    views = [
        {"id": "left", "label": "Left", "laterality": "left", "valid": True},
        {"id": "right", "label": "Right", "laterality": "right", "valid": True},
        {"id": "note", "label": "Note", "laterality": "", "valid": True},
    ]
    shown = pause_menu.visible_views(views, "right")
    assert [view["id"] for view in shown] == ["right", "note", "home"]
    assert shown[-1]["home"] is True


def test_command_is_one_line_the_simulator_can_parse():
    assert pause_menu.command_line(3, "pause") == "3 pause\n"
    assert pause_menu.command_line(4, "apply", "right_eac") == "4 apply right_eac\n"


if __name__ == "__main__":
    test_right_case_hides_left_views_and_keeps_home()
    test_command_is_one_line_the_simulator_can_parse()
    print("pause menu tests passed")
