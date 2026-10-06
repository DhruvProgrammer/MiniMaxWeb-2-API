"""Agent H - the --login flow, for real: headless Chromium against a local fake MiniMax Agent page."""
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from common import m, make_jwt

try:
    import playwright.sync_api  # noqa: F401
    HAVE_PW = True
except ImportError:
    HAVE_PW = False


def fake_site(js_after_load):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            body = f"<html><body>fake agent<script>{js_after_load}</script></body></html>".encode()
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@unittest.skipUnless(HAVE_PW, "playwright not installed")
class Login(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def run_login(self, js, **kw):
        srv, url = fake_site(js)
        try:
            return m.login_interactive("global", url, self.d / "tokens.json", headless=True,
                                       profile_dir=self.d / "profile", **kw)
        finally:
            srv.shutdown()

    def test_user_logs_in_later_token_and_realuserid_captured(self):
        jwt = make_jwt(user_id="450777")
        js = ("setTimeout(function(){localStorage.setItem('_token'," + json.dumps(jwt) + ");"
              "localStorage.setItem('user_detail_agent', JSON.stringify({data:{userInfo:{realUserID:'450999888'}}}));},1200);")
        tok = self.run_login(js, timeout=30)
        self.assertEqual(tok, "450999888+" + jwt)
        saved = m.read_tokens_file(self.d / "tokens.json")
        self.assertEqual(saved, [tok])
        self.assertEqual(oct((self.d / "tokens.json").stat().st_mode & 0o777), "0o600")
        self.assertEqual(oct((self.d / "profile").stat().st_mode & 0o777), "0o700")
        info = m.TokenInfo(saved[0])
        self.assertEqual(info.user_id, "450999888")

    def test_bare_token_accepted_after_grace(self):
        jwt = make_jwt(user_id="12345")
        tok = self.run_login("localStorage.setItem('_token'," + json.dumps(jwt) + ");", timeout=30, bare_token_grace=1.0)
        self.assertEqual(tok, jwt)

    def test_garbage_token_is_ignored_then_times_out(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_login("localStorage.setItem('_token','not-a-jwt');", timeout=3, bare_token_grace=0.5)
        self.assertIn("timed out", str(cm.exception))
        self.assertFalse((self.d / "tokens.json").exists())

    def test_never_logged_in_times_out_cleanly(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_login("/* nothing */", timeout=2)
        self.assertIn("timed out", str(cm.exception))

    def test_unreachable_site_is_friendly(self):
        with self.assertRaises(SystemExit) as cm:
            m.login_interactive("global", "http://127.0.0.1:9", self.d / "t.json", headless=True,
                                profile_dir=self.d / "p2", timeout=3)
        msg = str(cm.exception)
        self.assertIn("could not drive the browser", msg)
        self.assertLessEqual(len(msg.splitlines()), 4)

    def test_headed_without_display_is_friendly_not_a_traceback(self):
        import os
        if os.environ.get("DISPLAY"):
            self.skipTest("display available")
        with self.assertRaises(SystemExit) as cm:
            m.login_interactive("global", "http://127.0.0.1:9", self.d / "t.json", headless=False,
                                profile_dir=self.d / "p3", timeout=3)
        self.assertIn("print-snippet", str(cm.exception))
        self.assertLessEqual(len(str(cm.exception).splitlines()), 4)

    def test_login_then_server_uses_token_end_to_end(self):
        """login saves token -> running server hot-loads it -> request succeeds against the mock upstream."""
        from common import Stack
        s = Stack(tokens=[])
        try:
            self.assertEqual(s.chat_json("x")[0], 503)
            jwt = s.jwts[0]
            js = ("localStorage.setItem('_token'," + json.dumps(jwt) + ");"
                  "localStorage.setItem('user_detail_agent', JSON.stringify({u:{realUserID:'450000000000'}}));")
            srv, url = fake_site(js)
            try:
                m.login_interactive("global", url, Path(s.cfg["tokens_file"]), headless=True,
                                    profile_dir=self.d / "p4", timeout=30)
            finally:
                srv.shutdown()
            st, d = s.chat_json("after login")
            self.assertEqual(st, 200, d)
            self.assertEqual(d["choices"][0]["message"]["content"], "ECHO: after login")
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
