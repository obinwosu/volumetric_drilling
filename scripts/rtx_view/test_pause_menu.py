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


def test_burr_choice_opens_the_menu_then_changes_the_burr():
    assert pause_menu.burr_commands("size", 1) == ["open\n", "burr_size 0\n"]
    assert pause_menu.burr_commands("size", 6) == ["open\n", "burr_size 3\n"]
    assert pause_menu.burr_commands("style", "diamond") == ["open\n", "burr_type diamond\n"]


def test_burr_commands_go_one_at_a_time_and_the_card_follows_the_simulator():
    import tempfile
    import types

    with tempfile.TemporaryDirectory() as tmp:
        pause_menu.RUNTIME = pathlib.Path(tmp)
        pause_menu.BURR_COMMAND_PATH = pathlib.Path(tmp) / "proctor_menu.txt"
        viewer = types.SimpleNamespace()
        pause_menu._choose_burr(viewer, "size", 4)
        assert pause_menu.BURR_COMMAND_PATH.read_text() == "open\n"
        pause_menu.pump_burr(viewer)  # not read yet: the next command waits
        assert pause_menu.BURR_COMMAND_PATH.read_text() == "open\n"
        pause_menu.BURR_COMMAND_PATH.unlink()
        pause_menu.pump_burr(viewer)
        assert pause_menu.BURR_COMMAND_PATH.read_text() == "burr_size 2\n"
        pause_menu.sync(viewer, {"size_mm": 4, "style": "diamond"})
        pause_menu.sync(viewer, {"size_mm": 2, "style": "cutting"})
        assert viewer._pause_session["size_mm"] == 2 and viewer._pause_session["style"] == "cutting"


if __name__ == "__main__":
    test_right_case_hides_left_views_and_keeps_home()
    test_command_is_one_line_the_simulator_can_parse()
    test_burr_choice_opens_the_menu_then_changes_the_burr()
    test_burr_commands_go_one_at_a_time_and_the_card_follows_the_simulator()
    print("pause menu tests passed")
