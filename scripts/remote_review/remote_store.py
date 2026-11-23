"""Files behind remote review: coded cases, invites, comments. No web or GUI code here.

    <recording base>/remote_review/
        private/cases.json     code -> source session (never served)
        private/invites.json   sha256 of each invite token, its case, label, expiry (no tokens)
        cases/<code>/          what reviewers may load: case.json, labels, times, path, video.mp4
        feedback/<code>.jsonl  comments and "opened" / "finished" events
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import secrets
import threading
import time

ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # nothing that reads like another character
MAX_COMMENT = 2000
MAX_NAME = 40
_lock = threading.RLock()


def root_of(base) -> pathlib.Path:
    root = pathlib.Path(base) / "remote_review"
    for folder in (root, root / "private", root / "cases", root / "feedback"):
        folder.mkdir(parents=True, exist_ok=True)
        os.chmod(folder, 0o700)
    return root


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write(path, value) -> None:
    partial = path.with_suffix(".part")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=1)
    partial.replace(path)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_code(root) -> str:
    with _lock:
        cases = _read(root / "private" / "cases.json", {})
        while True:
            code = "CASE-" + "".join(secrets.choice(ALPHABET) for _ in range(4))
            if code not in cases:
                return code


def code_for_source(root, source) -> str | None:
    source = str(pathlib.Path(source).resolve())
    for code, entry in _read(root / "private" / "cases.json", {}).items():
        if entry["source"] == source:
            return code
    return None


def register_case(root, code, source) -> None:
    with _lock:
        path = root / "private" / "cases.json"
        cases = _read(path, {})
        cases[code] = {"source": str(pathlib.Path(source).resolve()), "posted": time.time()}
        _write(path, cases)


def list_cases(root) -> dict:
    return _read(root / "private" / "cases.json", {})


def create_invite(root, code, label, days=14.0) -> str:
    """A new invite link token. It is shown once: only its hash is stored."""
    label = " ".join(str(label).split())[:MAX_NAME]
    if not label:
        raise ValueError("An invite needs a reviewer label.")
    if code not in list_cases(root):
        raise ValueError(f"Unknown case {code}")
    token = secrets.token_urlsafe(32)
    with _lock:
        path = root / "private" / "invites.json"
        invites = _read(path, {})
        invites[hash_token(token)] = {"case": code, "label": label, "created": time.time(),
                                      "expires": time.time() + float(days) * 86400.0, "revoked": False}
        _write(path, invites)
    return token


def find_invite(root, token) -> dict | None:
    """The live invite for this token, or None if unknown, expired or revoked."""
    if not token:
        return None
    key = hash_token(token)
    entry = _read(root / "private" / "invites.json", {}).get(key)
    if not entry or entry["revoked"] or time.time() > entry["expires"]:
        return None
    return dict(entry, key=key)


def find_invite_by_key(root, key) -> dict | None:
    entry = _read(root / "private" / "invites.json", {}).get(key)
    if not entry or entry["revoked"] or time.time() > entry["expires"]:
        return None
    return dict(entry, key=key)


def list_invites(root, code=None) -> list[dict]:
    invites = _read(root / "private" / "invites.json", {})
    return [dict(v, key=k) for k, v in invites.items() if code is None or v["case"] == code]


def revoke_invite(root, key) -> None:
    with _lock:
        path = root / "private" / "invites.json"
        invites = _read(path, {})
        if key in invites:
            invites[key]["revoked"] = True
            _write(path, invites)


def _append(root, code, entry) -> dict:
    with _lock:
        path = root / "feedback" / f"{code}.jsonl"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
    return entry


def read_feedback(root, code) -> list[dict]:
    path = root / "feedback" / f"{code}.jsonl"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a line cut short by a crash must not hide the others
    return rows


def add_comment(root, code, author, t, text, duration) -> dict:
    text = str(text).strip()
    if not text or len(text) > MAX_COMMENT:
        raise ValueError("A comment must be 1 to 2000 characters.")
    t = float(t)
    if not 0.0 <= t <= float(duration) + 1.0:
        raise ValueError("The comment's time is outside the session.")
    return _append(root, code, {"kind": "comment", "id": secrets.token_hex(6), "author": author,
                                "t": round(t, 2), "text": text, "at": time.time()})


def add_event(root, code, author, kind) -> dict:
    if kind not in ("opened", "finished"):
        raise ValueError("Unknown event")
    return _append(root, code, {"kind": kind, "author": author, "at": time.time()})
