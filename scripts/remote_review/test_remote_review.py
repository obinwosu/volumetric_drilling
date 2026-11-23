"""Remote review: nothing identifying is packaged, and nothing is served without a live invite.

    python3 scripts/remote_review/test_remote_review.py
"""
import gzip
import http.client
import json
import os
import pathlib
import sys
import tempfile
import threading
import time

os.environ["FIVRS_REVIEW_NO_3D"] = "1"
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "session_review"))
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import make_demo_session as demo  # noqa: E402
import package_case as pc  # noqa: E402
import remote_store as store  # noqa: E402
import review_server as rs  # noqa: E402
import session_data as data  # noqa: E402

rs.notify = lambda *_: None
SPECIMEN = yaml.safe_load((data.REPO / "scripts/expert_drilling/specimens.yaml").read_text())["specimens"][0]


def request(port, method, path, cookie=None, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    sent = dict(headers or {})
    if cookie:
        sent["Cookie"] = cookie
    if body is not None:
        body = json.dumps(body)
        sent.setdefault("Content-Type", "application/json")
    connection.request(method, path, body, sent)
    reply = connection.getresponse()
    content = reply.read()
    connection.close()
    return reply, content


def test_downsample_keeps_thin_structures_and_drilling_times():
    ids = np.zeros((8, 8, 8), np.uint8)
    times = np.zeros((8, 8, 8), np.uint16)
    ids[:] = 1
    times[:] = pc.NEVER
    times[:4] = 100  # half the bone is drilled at time 100
    ids[6, 6, 6] = 2  # a one-voxel nerve
    times[6, 6, 6] = 500
    table = [{"id": 1, "voxels": 511 + 40000}, {"id": 2, "voxels": 1}]
    small_ids, small_times = pc.downsample(ids, times, table)
    assert small_ids.shape == (4, 4, 4) and small_ids[3, 3, 3] == 2 and small_times[3, 3, 3] == 500
    assert small_ids[0, 0, 0] == 1 and small_times[0, 0, 0] == 100 and small_times[2, 0, 0] == pc.NEVER


def test_case_and_access():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        folder = base / "reviewable_sessions" / "Private_Person" / "secret-anatomy" / "session_001"
        demo.build(folder, SPECIMEN)
        meta = json.loads((folder / "session.json").read_text())
        meta["participant"] = {"first_name": "Private", "last_name": "Person"}
        meta["specimen"]["name"] = "secret-anatomy"
        (folder / "session.json").write_text(json.dumps(meta))
        try:
            pc.package(folder, base, video=False)
            raise AssertionError("A session with a volume reset must be refused")
        except ValueError as error:
            assert "reset" in str(error)
        with data.h5py.File(folder / "ground_truth.h5", "r+") as h5:
            del h5["volume/epoch_1"]
        code = pc.package(folder, base, video=False)
        root = store.root_of(base)
        case_dir = root / "cases" / code
        (case_dir / "video.mp4").write_bytes(bytes(range(256)) * 8)
        text = (case_dir / "case.json").read_text().lower()
        for secret in ("private", "person", "secret-anatomy", str(tmp).lower(), "t_sim", "started"):
            assert secret not in text, secret
        case = json.loads(text)
        labels = np.frombuffer(gzip.decompress((case_dir / "labels.bin.gz").read_bytes()), np.uint8)
        assert labels.size == int(np.prod(case["shape"])) and labels.max() > 0

        token = store.create_invite(root, code, "Attending")
        other = store.new_code(root)
        store.register_case(root, other, base / "elsewhere")
        (root / "cases" / other).mkdir()
        (root / "cases" / other / "case.json").write_text('{"code": "other", "duration": 1}')
        assert token not in (root / "private" / "invites.json").read_text()

        server = rs.ReviewServer(("127.0.0.1", 0), root)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for path in ("/", "/app.js", "/api/case", "/api/comments", "/case/labels.bin.gz", "/case/video.mp4", "/i/wrong"):
                assert request(port, "GET", path)[0].status == 403, path
            assert request(port, "POST", "/api/comments", body={"t": 1, "text": "x"})[0].status == 403
            reply, _ = request(port, "GET", f"/i/{token}", headers={"X-Forwarded-Proto": "https"})
            assert reply.status == 303 and reply.getheader("Location") == "/"
            set_cookie = reply.getheader("Set-Cookie")
            assert "HttpOnly" in set_cookie and "SameSite=Lax" in set_cookie and "Secure" in set_cookie
            cookie = set_cookie.split(";")[0]
            reply, content = request(port, "GET", "/", cookie)
            assert reply.status == 200 and b"Case Review" in content and "script-src 'self'" in reply.getheader("Content-Security-Policy")
            reply, content = request(port, "GET", "/api/case", cookie)
            assert json.loads(content)["code"] == code and json.loads(content)["reviewer"] == "Attending"
            for path in ("/case/case.json", "/case/../../private/invites.json", "/case/%2e%2e/x", "/review_server.py"):
                assert request(port, "GET", path, cookie)[0].status == 404, path
            reply, content = request(port, "GET", "/case/video.mp4", cookie, headers={"Range": "bytes=10-19"})
            assert reply.status == 206 and content == bytes(range(10, 20))
            assert reply.getheader("Content-Range") == "bytes 10-19/2048"

            assert request(port, "POST", "/api/comments", cookie, {"t": -5, "text": "x"})[0].status == 400
            assert request(port, "POST", "/api/comments", cookie, {"t": 1, "text": ""})[0].status == 400
            assert request(port, "POST", "/api/comments", cookie, {"t": 1, "text": "y" * 9000})[0].status == 413
            reply, _ = request(port, "POST", "/api/comments", cookie, {"t": 1}, {"Content-Type": "text/plain"})
            assert reply.status == 415
            script = "<script>alert(1)</script>"
            assert request(port, "POST", "/api/comments", cookie, {"t": 2.5, "text": script})[0].status == 200
            rows = json.loads(request(port, "GET", "/api/comments", cookie)[1])
            assert len(rows) == 1 and rows[0]["text"] == script and rows[0]["author"] == "Attending"
            assert request(port, "POST", "/api/finished", cookie, {})[0].status == 200
            assert [r["kind"] for r in store.read_feedback(root, code)] == ["comment", "finished"]
            assert store.read_feedback(root, other) == []

            store.revoke_invite(root, store.hash_token(token))
            assert request(port, "GET", "/api/case", cookie)[0].status == 403
            expired = store.create_invite(root, code, "Late", days=-1)
            assert request(port, "GET", f"/i/{expired}")[0].status == 403
        finally:
            server.shutdown()
            server.server_close()


def test_review_tab_box_posts_invites_and_reads_feedback():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets

    import review_app
    import share_panel

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        folder = base / "reviewable_sessions" / "A_B" / "C" / "session_001"
        demo.build(folder, SPECIMEN)
        window = review_app.ReviewWindow(tmp)
        window.show()
        window.load(folder)
        panel = window.share
        assert panel.status.text() == "Not posted." and not panel.button_invite.isEnabled()
        root = store.root_of(base)
        code = store.new_code(root)
        store.register_case(root, code, folder)
        panel.refresh()
        assert panel.code == code and panel.button_invite.isEnabled()
        panel.address.setText("http://not-private.example")
        try:
            panel.make_invite("Attending")
            raise AssertionError("A plain http tunnel address must be refused")
        except ValueError:
            pass
        panel.address.setText("https://tunnel.example/")
        link = panel.make_invite("Attending")
        assert link.startswith("https://tunnel.example/i/")
        assert store.find_invite(root, link.rsplit("/", 1)[1])["label"] == "Attending"
        store.add_comment(root, code, "Attending", 5.0, "Thin the tegmen here", window.session.duration)
        store.add_event(root, code, "Attending", "finished")
        panel.refresh()
        assert panel.feedback.count() == 2 and "Thin the tegmen" in panel.feedback.item(0).text()
        panel._jump(panel.feedback.item(0))
        assert abs(window.t - 5.0) < 0.1
        assert share_panel.clean_address("") == ""
        window.close()
        app.processEvents()


def test_sharing_is_a_user_service():
    import sharing_service as service

    calls = []
    real_run, real_config = service.run, os.environ.get("XDG_CONFIG_HOME")
    service.run = lambda *args: calls.append(args) or type("Done", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["XDG_CONFIG_HOME"] = tmp
            service.start("/data/my base 100%", 8766)
            text = service.unit_path().read_text()
            assert '--base "/data/my base 100%%" --port 8766' in text and "WantedBy=default.target" in text
            assert "127.0.0.1" not in text and "Restart=on-failure" in text
            assert calls == [("daemon-reload",), ("enable", service.UNIT), ("restart", service.UNIT)]
            calls.clear()
            service.stop()
            assert calls == [("disable", "--now", service.UNIT)]
    finally:
        service.run = real_run
        if real_config is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = real_config


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
