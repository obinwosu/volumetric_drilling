#!/usr/bin/env python3
"""Invite-only review site for coded cases. It listens on this computer only; a tunnel reaches it.

    python3 scripts/remote_review/review_server.py --base <recording base> [--port 8766]

An invite link (/i/<token>) sets a cookie and shows that one case. Without a live invite every
address answers 403. Tokens are stored only as hashes.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remote_store as store  # noqa: E402

WEB = HERE / "web"
COOKIE = "fivrs_review"
WEB_FILES = {"/": "index.html", "/app.js": "app.js", "/app.css": "app.css"}
CASE_FILES = {"labels.bin.gz", "times.bin.gz", "labels_half.bin.gz", "times_half.bin.gz", "path.bin.gz",
              "curves.bin.gz", "video.mp4"}
MAX_BODY = 8192
COMMENTS_PER_MINUTE = 30
NOTIFY_GAP_S = 60.0
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self'; "
       "img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def notify(title, body="") -> None:
    try:
        subprocess.Popen(["notify-send", "-a", "FIVRS Review", title, body],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


class Handler(BaseHTTPRequestHandler):
    server_version = "review"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # request paths can hold invite tokens
        pass

    @property
    def root(self):
        return self.server.root

    def _common(self, kind, length, status=200, cache="no-store", extra=()):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Robots-Tag", "noindex, nofollow")
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()

    def _bytes(self, body, kind="application/json", status=200, extra=()):
        self._common(kind, len(body), status, extra=extra)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, value, status=200):
        self._bytes(json.dumps(value).encode(), status=status)

    def _denied(self):
        page = (WEB / "denied.html").read_bytes()
        self._bytes(page, "text/html; charset=utf-8", 403)

    def _invite(self):
        try:
            jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        except cookies.CookieError:
            return None
        return store.find_invite(self.root, jar[COOKIE].value) if COOKIE in jar else None

    def _accept(self, token):
        invite = store.find_invite(self.root, token)
        if invite is None:
            return self._denied()
        age = max(int(invite["expires"] - time.time()), 1)
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto", "").lower() == "https" else ""
        cookie = f"{COOKIE}={token}; Path=/; Max-Age={age}; HttpOnly; SameSite=Lax{secure}"
        self._common("text/plain", 0, 303, extra=(("Location", "/"), ("Set-Cookie", cookie)))

    def _file(self, path, kind, cache="no-store", encoding=None):
        size = path.stat().st_size
        start, stop, status, extra = 0, size - 1, 200, [("Accept-Ranges", "bytes")]
        wanted = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "")) if encoding is None else None
        if wanted and (wanted[1] or wanted[2]):
            if wanted[1]:
                start = int(wanted[1])
                stop = min(int(wanted[2]), size - 1) if wanted[2] else size - 1
            else:
                start, stop = max(size - int(wanted[2]), 0), size - 1
            if start > stop or start >= size:
                return self._bytes(b"", "text/plain", 416, extra=(("Content-Range", f"bytes */{size}"),))
            status = 206
            extra.append(("Content-Range", f"bytes {start}-{stop}/{size}"))
        if encoding:
            extra.append(("Content-Encoding", encoding))
        self._common(kind, stop - start + 1, status, cache, extra)
        if self.command == "HEAD":
            return None
        with path.open("rb") as handle:
            handle.seek(start)
            left = stop - start + 1
            while left:
                chunk = handle.read(min(left, 262144))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)
        return None

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _get(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/i/"):
            return self._accept(path[3:])
        invite = self._invite()
        if invite is None:
            return self._denied()
        code = invite["case"]
        folder = self.root / "cases" / code
        if path in WEB_FILES:
            name = WEB_FILES[path]
            kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
            return self._file(WEB / name, f"{kind}; charset=utf-8")
        if path == "/api/case":
            case = json.loads((folder / "case.json").read_text())
            case["reviewer"] = invite["label"]
            return self._json(case)
        if path == "/api/comments":
            rows = [{k: row[k] for k in ("id", "author", "t", "text", "at")}
                    for row in store.read_feedback(self.root, code) if row["kind"] == "comment"]
            return self._json(rows)
        if path.startswith("/case/") and path[6:] in CASE_FILES:
            name = path[6:]
            if name == "video.mp4":
                return self._file(folder / name, "video/mp4", "private, max-age=3600")
            return self._file(folder / name, "application/octet-stream", "private, max-age=3600", "gzip")
        return self._bytes(b"{}", status=404)

    def do_POST(self):
        try:
            self._post()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _post(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.close_connection = True
            return self._json({"error": "Too long."}, 413)
        body = self.rfile.read(length)
        invite = self._invite()
        if invite is None:
            return self._denied()
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._json({"error": "JSON only."}, 415)
        try:
            value = json.loads(body or b"{}")
        except ValueError:
            return self._json({"error": "Bad request."}, 400)
        code, author = invite["case"], invite["label"]
        if self.path == "/api/comments":
            if not self.server.allow_comment(invite["key"]):
                return self._json({"error": "Too many comments. Wait a minute."}, 429)
            duration = json.loads((self.root / "cases" / code / "case.json").read_text())["duration"]
            try:
                row = store.add_comment(self.root, code, author, value.get("t", 0), value.get("text", ""), duration)
            except (TypeError, ValueError, AttributeError) as exc:
                return self._json({"error": str(exc)}, 400)
            self.server.notify_once(("comment", invite["key"]), f"{author} commented on {code}")
            return self._json({k: row[k] for k in ("id", "author", "t", "text", "at")})
        if self.path in ("/api/opened", "/api/finished"):
            kind = self.path.rsplit("/", 1)[1]
            store.add_event(self.root, code, author, kind)
            if kind == "finished":
                notify(f"{author} finished reviewing {code}", "Open the Review tab to read the feedback.")
            else:
                self.server.notify_once(("opened", invite["key"]), f"{author} opened {code}", 1800.0)
            return self._json({"ok": True})
        return self._json({"error": "Not found."}, 404)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, root):
        super().__init__(address, Handler)
        self.root = root
        self._lock = threading.Lock()
        self._comment_times = {}
        self._notified = {}

    def allow_comment(self, key) -> bool:
        with self._lock:
            now = time.time()
            recent = [t for t in self._comment_times.get(key, []) if now - t < 60.0]
            if len(recent) >= COMMENTS_PER_MINUTE:
                return False
            self._comment_times[key] = recent + [now]
            return True

    def notify_once(self, key, title, gap=NOTIFY_GAP_S) -> None:
        with self._lock:
            now = time.time()
            if now - self._notified.get(key, 0.0) < gap:
                return
            self._notified[key] = now
        notify(title)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    root = store.root_of(args.base)
    server = ReviewServer(("127.0.0.1", args.port), root)  # never a public address: the tunnel connects here
    (root / "private" / "server.pid").write_text(str(os.getpid()))
    print(f"@listening http://127.0.0.1:{args.port}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # a stopped service still clears its pid file
    try:
        server.serve_forever()
    finally:
        (root / "private" / "server.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
