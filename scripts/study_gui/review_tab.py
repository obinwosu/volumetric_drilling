"""Launcher tab "Review": the session review app, inside the launcher.

The review code (and VTK, h5py) loads the first time the tab is opened, so a launcher that never
reviews anything starts and runs as before. Playback pauses when the tab is left, so a review
left open costs the simulator nothing.
"""
from __future__ import annotations

import pathlib
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

REVIEW_DIR = pathlib.Path(__file__).resolve().parent.parent / "session_review"


class ReviewPanel(QtWidgets.QWidget):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.window_widget = None
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.note = QtWidgets.QLabel("Loading session review...")
        self.note.setAlignment(Qt.AlignCenter)
        self.note.setWordWrap(True)
        self._layout.addWidget(self.note)

    def base(self) -> str:
        return str(self.ui.gui_configuration.recording_base_path.get())

    def _ensure_built(self) -> bool:
        if self.window_widget is not None:
            return True
        try:
            if str(REVIEW_DIR) not in sys.path:
                sys.path.insert(0, str(REVIEW_DIR))
            import review_app
            self.window_widget = review_app.ReviewWindow(self.base(), embedded=True)
        except Exception as exc:  # a missing review dependency must never take the launcher down
            self.note.setText(f"Session review could not start:\n{exc}")
            return False
        self.note.hide()
        self._layout.addWidget(self.window_widget)
        return True

    def showEvent(self, event):
        super().showEvent(event)
        if self._ensure_built():
            self.window_widget.base = pathlib.Path(self.base())
            self.window_widget.refresh_sessions()
            self.window_widget.setFocus()

    def hideEvent(self, event):
        super().hideEvent(event)
        if self.window_widget is not None:
            self.window_widget.pause()

    def open_session(self, folder=None):
        """Switch to this tab and, if given, open that session."""
        tabs = self.ui.findChild(QtWidgets.QTabWidget, "tabWidget")
        if tabs is not None:
            tabs.setCurrentWidget(self)
        if folder and self._ensure_built():
            self.window_widget.load(str(folder))

    def shutdown(self):
        if self.window_widget is not None:
            self.window_widget.shutdown()
