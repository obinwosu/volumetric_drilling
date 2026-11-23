"""Review tab box "Remote review": post a session as a coded case, invite a reviewer, read feedback."""
from __future__ import annotations

import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

REMOTE_DIR = pathlib.Path(__file__).resolve().parent.parent / "remote_review"
sys.path.insert(0, str(REMOTE_DIR))
import remote_store as store  # noqa: E402
import sharing_service  # noqa: E402

PORT = 8766
LOCAL_URL = f"http://127.0.0.1:{PORT}"


def clock(seconds):
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def clean_address(text: str) -> str:
    """The tunnel's https address, or "" when blank. Anything else is refused."""
    address = text.strip().rstrip("/")
    if address and (not address.startswith("https://") or any(c.isspace() for c in address)):
        raise ValueError("The tunnel address must start with https://")
    return address


class SharePanel(QtWidgets.QGroupBox):
    def __init__(self, window):
        super().__init__("Remote review")
        self.window = window
        self.folder = None
        self.code = None
        self.process = None
        self._rows_seen = None
        layout = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel("Pick a session.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.button_post = QtWidgets.QPushButton("Post for review")
        self.button_post.clicked.connect(self.post)
        self.button_server = QtWidgets.QPushButton("Start sharing")
        self.button_server.clicked.connect(self.toggle_server)
        row.addWidget(self.button_post)
        row.addWidget(self.button_server)
        layout.addLayout(row)
        self.address = QtWidgets.QLineEdit()
        self.address.setPlaceholderText("Tunnel address (https://...)")
        self.address.editingFinished.connect(self._save_address)
        layout.addWidget(self.address)
        row = QtWidgets.QHBoxLayout()
        self.reviewer = QtWidgets.QLineEdit()
        self.reviewer.setPlaceholderText("Reviewer, e.g. Attending")
        self.reviewer.setMaxLength(store.MAX_NAME)
        self.button_invite = QtWidgets.QPushButton("Invite")
        self.button_invite.clicked.connect(self.invite)
        self.button_revoke = QtWidgets.QPushButton("Revoke all")
        self.button_revoke.clicked.connect(self.revoke_all)
        row.addWidget(self.reviewer, 1)
        row.addWidget(self.button_invite)
        row.addWidget(self.button_revoke)
        layout.addLayout(row)
        self.feedback = QtWidgets.QListWidget()
        self.feedback.setWordWrap(True)
        self.feedback.setMaximumHeight(150)
        self.feedback.itemClicked.connect(self._jump)
        layout.addWidget(self.feedback)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self.refresh)
        self.refresh()

    @property
    def root(self):
        return store.root_of(self.window.base)

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()
        self.refresh()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()

    def set_session(self, folder):
        self.folder = pathlib.Path(folder).resolve()
        self._rows_seen = None
        self.refresh()

    # Server ------------------------------------------------------------------------------------
    def _server_pid(self):
        """The running review server's pid, checked against the process's command line."""
        try:
            pid = int((self.root / "private" / "server.pid").read_text())
            command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
        except (OSError, ValueError):
            return None
        return pid if b"review_server.py" in command else None

    @staticmethod
    def _port_busy():
        with socket.socket() as probe:
            probe.settimeout(0.2)
            return probe.connect_ex(("127.0.0.1", PORT)) == 0

    def toggle_server(self):
        pid = self._server_pid()
        try:
            if pid:
                sharing_service.stop()
                if self._server_pid():  # one started before sharing became a service
                    os.kill(pid, signal.SIGTERM)
                for _ in range(40):
                    if not self._server_pid():
                        break
                    time.sleep(0.05)
            elif self._port_busy():
                QtWidgets.QMessageBox.warning(self, "Remote review", f"Port {PORT} is in use by another program.")
            else:
                sharing_service.start(self.window.base, PORT)
                for _ in range(60):
                    if self._port_busy():
                        break
                    time.sleep(0.05)
                else:
                    QtWidgets.QMessageBox.warning(self, "Remote review", "The sharing service did not start.")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            QtWidgets.QMessageBox.warning(self, "Remote review", f"Could not change sharing:\n{exc}")
        self.refresh()

    # Posting -----------------------------------------------------------------------------------
    def post(self):
        if self.folder is None or self.process is not None:
            return
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._post_output)
        self.process.finished.connect(self._post_finished)
        self._post_log = ""
        self.status.setText("Preparing the case (about a minute)...")
        self.button_post.setEnabled(False)
        self.process.start("nice", ["-n", "10", sys.executable, str(REMOTE_DIR / "package_case.py"), str(self.folder),
                                    "--base", str(self.window.base)])

    def _post_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self._post_log = (self._post_log + text)[-1500:]
        lines = [line for line in text.splitlines() if line.startswith("video ")]
        if lines:
            self.status.setText(f"Preparing the case: {lines[-1]}")

    def _post_finished(self, code, _status):
        process, self.process = self.process, None
        process.deleteLater()
        self.button_post.setEnabled(True)
        if code != 0:
            last = [line for line in self._post_log.splitlines() if line.strip()]
            QtWidgets.QMessageBox.warning(self, "Remote review", "Could not post this session:\n" + (last[-1] if last else ""))
        self.refresh()

    # Invites -----------------------------------------------------------------------------------
    def _save_address(self):
        try:
            address = clean_address(self.address.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Remote review", str(exc))
            return
        path = self.root / "private" / "public_address.txt"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(address)
        self.refresh()

    def make_invite(self, label):
        address = clean_address(self.address.text()) or LOCAL_URL
        return f"{address}/i/{store.create_invite(self.root, self.code, label)}"

    def invite(self):
        label = self.reviewer.text().strip()
        if not self.code or not label:
            QtWidgets.QMessageBox.information(self, "Remote review", "Post the case and name the reviewer first.")
            return
        link = self.make_invite(label)
        QtWidgets.QApplication.clipboard().setText(link)
        note = "" if self.address.text().strip() else "\n\nNo tunnel address is set, so this link works on this computer only."
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Invite link")
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.setText(f"Copied. It is shown only once and lasts 14 days.\n\n{link}{note}")
        box.exec_()
        self.reviewer.clear()
        self.refresh()

    def revoke_all(self):
        live = [i for i in store.list_invites(self.root, self.code) if not i["revoked"]] if self.code else []
        if not live:
            return
        answer = QtWidgets.QMessageBox.question(self, "Remote review", f"Stop {len(live)} invite link(s) for {self.code}?")
        if answer == QtWidgets.QMessageBox.Yes:
            for invite in live:
                store.revoke_invite(self.root, invite["key"])
            self.refresh()

    # Feedback ----------------------------------------------------------------------------------
    def refresh(self):
        pid = self._server_pid() if self.window.base else None
        self.button_server.setText("Stop sharing" if pid else "Start sharing")
        if self.address.text() == "" and not self.address.hasFocus():
            try:
                self.address.setText((self.root / "private" / "public_address.txt").read_text().strip())
            except OSError:
                pass
        self.code = store.code_for_source(self.root, self.folder) if self.folder else None
        self.button_post.setEnabled(self.folder is not None and self.process is None)
        self.button_post.setText("Post again" if self.code else "Post for review")
        self.button_invite.setEnabled(bool(self.code))
        self.button_revoke.setEnabled(bool(self.code))
        if self.process is not None:
            return
        if self.folder is None:
            self.status.setText("Pick a session.")
        elif not self.code:
            self.status.setText("Not posted.")
        else:
            live = sum(not i["revoked"] and i["expires"] > time.time() for i in store.list_invites(self.root, self.code))
            state = "sharing" if pid else "not sharing"
            self.status.setText(f"{self.code}, {state}, {live} live invite(s).")
        rows = store.read_feedback(self.root, self.code) if self.code else []
        if rows != self._rows_seen:
            self._rows_seen = rows
            self._fill(rows)

    def _fill(self, rows):
        self.feedback.clear()
        comments = sorted((r for r in rows if r["kind"] == "comment"), key=lambda r: r["t"])
        for row in comments:
            item = QtWidgets.QListWidgetItem(f"{clock(row['t'])}  {row['author']}: {row['text']}")
            item.setData(Qt.UserRole, float(row["t"]))
            self.feedback.addItem(item)
        for author in dict.fromkeys(r["author"] for r in rows if r["kind"] == "finished"):
            item = QtWidgets.QListWidgetItem(f"{author} finished reviewing")
            item.setFlags(Qt.ItemIsEnabled)
            self.feedback.addItem(item)

    def _jump(self, item):
        t = item.data(Qt.UserRole)
        if t is not None and self.window.session is not None:
            self.window.pause()
            self.window.seek(float(t))
