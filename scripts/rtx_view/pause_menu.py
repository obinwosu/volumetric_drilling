"""Desktop pause card for the RTX window.

Ctrl+P opens it. The simulator still owns the pause: this window asks it
to freeze, to recall a named view, and to store the current microscope
pose. The card is drawn with ImGui on the desktop window only.
"""

from __future__ import annotations

import pathlib
import time

import camera_views

RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
COMMAND_PATH = RUNTIME / "pause_menu.txt"


def visible_views(views, laterality: str) -> list[dict]:
    """Same side filter as the simulator pause menu, plus Home."""
    side = (laterality or "").strip().lower()
    shown = []
    for view in views:
        view_side = (view.get("laterality") or "").strip().lower()
        if side in ("left", "right") and view_side in ("left", "right") and view_side != side:
            continue
        shown.append(view)
    shown.append({"id": "home", "label": "Home", "laterality": "", "valid": True, "home": True})
    return shown


def command_line(seq: int, verb: str, view_id: str = "") -> str:
    if view_id:
        return f"{int(seq)} {verb} {view_id}\n"
    return f"{int(seq)} {verb}\n"


def is_open(viewer) -> bool:
    return bool(getattr(viewer, "_pause_open", False))


def _send(viewer, verb: str, view_id: str = "") -> None:
    seq = int(getattr(viewer, "_pause_seq", 0)) + 1
    viewer._pause_seq = seq
    RUNTIME.mkdir(parents=True, exist_ok=True)
    temporary = COMMAND_PATH.with_suffix(".txt.tmp")
    temporary.write_text(command_line(seq, verb, view_id))
    temporary.replace(COMMAND_PATH)


def toggle(viewer) -> None:
    opening = not is_open(viewer)
    viewer._pause_open = opening
    viewer._pause_expect = opening
    viewer._pause_expect_at = time.perf_counter()
    viewer._pause_status = "Paused" if opening else ""
    _send(viewer, "pause" if opening else "resume")


def sync(viewer, state: dict | None) -> None:
    """Follow a pause that started in the simulator, once our last request lands."""
    state = state or {}
    session = getattr(viewer, "_pause_session", None)
    if session is None:
        session = {}
        viewer._pause_session = session
    for key in ("user", "anatomy", "laterality", "drill"):
        value = state.get(key)
        if isinstance(value, str):
            session[key] = value
    if not session.get("drill"):
        size = state.get("size_mm")
        style = state.get("style")
        if size:
            session["drill"] = f"{int(size)} mm {style or ''}".strip()
    reported = state.get("paused")
    if not isinstance(reported, bool):
        return
    expect = getattr(viewer, "_pause_expect", None)
    if expect is not None:
        if reported == expect or time.perf_counter() - float(getattr(viewer, "_pause_expect_at", 0.0)) > 2.0:
            viewer._pause_expect = None
        else:
            return
    if reported != is_open(viewer):
        viewer._pause_open = reported
        if reported:
            viewer._pause_status = "Paused"


def _quiet_stock_panels() -> None:
    from newton._src.viewer.viewer_gui import ViewerGui

    ViewerGui._render_left_panel = lambda self: None
    ViewerGui._render_stats_overlay = lambda self: None


def install(viewer) -> None:
    """Turn on ImGui and draw only this card. The model panel stays hidden."""
    _quiet_stock_panels()
    viewer.show_ui = True
    viewer._pause_open = False
    viewer._pause_session = {}
    viewer._pause_status = ""
    viewer._pause_selected = "home"
    viewer.register_ui_callback(lambda imgui: _draw(imgui, viewer), position="free")


def _reload_views(viewer) -> list[dict]:
    path = camera_views.VIEWS_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if mtime != getattr(viewer, "_pause_views_mtime", None) or not hasattr(viewer, "_pause_views"):
        viewer._pause_views = camera_views.load_views(path)
        viewer._pause_views_mtime = mtime
    return viewer._pause_views


def _choose(viewer, views) -> dict | None:
    selected = getattr(viewer, "_pause_selected", "")
    for view in views:
        if view["id"] == selected:
            return view
    return views[0] if views else None


def _go(viewer, chosen) -> None:
    if chosen is None:
        return
    if not chosen.get("valid"):
        label = chosen.get("label") or chosen["id"]
        viewer._pause_status = f"{label} is empty. Frame the camera, then save it."
        return
    _send(viewer, "apply", chosen["id"])
    viewer._pause_status = f"Showing {chosen.get('label') or chosen['id']}"


def _save(viewer, chosen) -> None:
    if chosen is None or chosen.get("home"):
        viewer._pause_status = "Home stays the ADF camera."
        return
    _send(viewer, "save", chosen["id"])
    viewer._pause_views_mtime = None
    viewer._pause_status = f"Saving {chosen.get('label') or chosen['id']}"


def _draw(imgui, viewer) -> None:
    if not is_open(viewer):
        return
    io = imgui.get_io()
    width, height = 460.0, 560.0
    display = io.display_size
    imgui.set_next_window_pos(
        imgui.ImVec2((display[0] - width) * 0.5, (display[1] - height) * 0.5),
        imgui.Cond_.always,
    )
    imgui.set_next_window_size(imgui.ImVec2(width, height), imgui.Cond_.always)
    imgui.push_style_var(imgui.StyleVar_.window_rounding, 12.0)
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(22, 20))
    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(14, 8))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(14, 12))
    flags = imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse
    imgui.begin("Paused", None, flags)

    session = getattr(viewer, "_pause_session", {}) or {}
    rows = (
        ("User", session.get("user") or ""),
        ("Anatomy", session.get("anatomy") or ""),
        ("Side", session.get("laterality") or ""),
        ("Drill", session.get("drill") or ""),
    )
    wrote_session = False
    for label, value in rows:
        if not value:
            continue
        imgui.text(label)
        imgui.same_line(110)
        imgui.text(value)
        wrote_session = True
    if wrote_session:
        imgui.separator()

    imgui.text("Views")
    views = visible_views(_reload_views(viewer), session.get("laterality") or "")
    chosen = _choose(viewer, views)
    if chosen is not None:
        viewer._pause_selected = chosen["id"]
    imgui.begin_child("pause_views", imgui.ImVec2(0, 250))
    for view in views:
        side = view.get("laterality") or ""
        mark = "" if view.get("home") else ("saved" if view.get("valid") else "empty")
        pieces = [view.get("label") or view["id"]]
        if side:
            pieces.append(side)
        if mark:
            pieces.append(mark)
        label = "    ".join(pieces) + f"##pause_{view['id']}"
        clicked, _ = imgui.selectable(label, view["id"] == viewer._pause_selected, 0, imgui.ImVec2(0, 32))
        if clicked:
            viewer._pause_selected = view["id"]
            chosen = view
    imgui.end_child()
    imgui.separator()

    if imgui.button("Go to view", imgui.ImVec2(120, 0)):
        _go(viewer, chosen)
    imgui.same_line()
    if imgui.button("Save this view", imgui.ImVec2(136, 0)):
        _save(viewer, chosen)
    imgui.same_line()
    if imgui.button("Resume", imgui.ImVec2(100, 0)):
        toggle(viewer)
    status = getattr(viewer, "_pause_status", "") or ""
    if status:
        imgui.text(status)
    imgui.text_disabled("Ctrl+P also resumes.")
    imgui.end()
    imgui.pop_style_var(4)
