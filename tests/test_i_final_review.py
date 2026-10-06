"""Agent I - final adversarial review: things found by reading the code with hostile eyes."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from common import Stack, m, make_jwt

SCRIPT = str(Path(__file__).resolve().parent.parent / "minimax_web2api.py")


def cli(args, env=None):
    e = {**os.environ, "HTTPS_PROXY": "", "https_proxy": "", **(env or {})}
    return subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True, timeout=20, env=e)


class FinalReview(unittest.TestCase):
    def test_nonstream_client_disconnect_stops_polling(self):
        s = Stack(settle_sec=3.0, request_timeout_sec=30)
        try:
            sock = socket.create_connection(("127.0.0.1", s.port), timeout=5)
            body = json.dumps({"messages": [{"role": "user", "content": "[[pause]] hold on"}]}).encode()
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                         b"Content-Length: %d\r\n\r\n" % len(body) + body)
            time.sleep(0.4)       # request is in flight, nothing written back yet
            sock.close()
            time.sleep(1.0)
            n1 = s.mock.detail_calls
            time.sleep(1.0)
            n2 = s.mock.detail_calls
            self.assertLessEqual(n2 - n1, 2, f"kept polling MiniMax for a client that left ({n1}->{n2})")
            # and the slot is free again
            self.assertEqual(s.chat_json("next")[0], 200)
        finally:
            s.close()

    def test_nonstream_normal_request_not_cancelled_by_alive_check(self):
        s = Stack()
        try:
            for _ in range(5):
                st, d = s.chat_json("still here")
                self.assertEqual(st, 200)
        finally:
            s.close()

    def test_pipelined_next_request_is_not_mistaken_for_disconnect(self):
        s = Stack()
        try:
            body = json.dumps({"messages": [{"role": "user", "content": "p1"}]}).encode()
            req = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                   b"Content-Length: %d\r\n\r\n" % len(body)) + body
            sock = socket.create_connection(("127.0.0.1", s.port), timeout=10)
            sock.sendall(req + b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            time.sleep(2.5)
            data = sock.recv(65536)
            sock.close()
            self.assertIn(b"ECHO: p1", data)
        finally:
            s.close()

    def test_non_ascii_auth_header_does_not_crash(self):
        s = Stack(api_keys=["sk-ok"])
        try:
            sock = socket.create_connection(("127.0.0.1", s.port), timeout=3)
            sock.sendall(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer \xff\xfe\xc3\r\n\r\n")
            self.assertIn(b" 401 ", sock.recv(4096).split(b"\r\n")[0])
            sock.close()
            self.assertEqual(s.req("GET", "/v1/models", headers={"Authorization": "Bearer sk-ok"})[0], 200)
        finally:
            s.close()

    def test_ipv6_host_binding(self):
        try:
            probe = socket.socket(socket.AF_INET6); probe.bind(("::1", 0)); probe.close()
        except OSError:
            self.skipTest("no IPv6 loopback")
        cfg = {**m.DEFAULTS, "host": "::1", "port": 0, "tokens": [], "tokens_file": None}
        srv = m.create_server(m.validate_config(cfg))
        try:
            self.assertEqual(srv.address_family, socket.AF_INET6)
        finally:
            srv.server_close()

    def test_port_in_use_is_friendly(self):
        blocker = socket.socket(); blocker.bind(("127.0.0.1", 0)); blocker.listen(1)
        port = blocker.getsockname()[1]
        try:
            r = cli(["--port", str(port), "--tokens-file", os.path.join(tempfile.mkdtemp(), "t.json")])
            self.assertNotEqual(r.returncode, 0)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("port", (r.stderr + r.stdout).lower())
        finally:
            blocker.close()

    def test_check_without_tokens(self):
        home = tempfile.mkdtemp()
        r = cli(["--tokens-file", os.path.join(home, "none.json")], {"MINIMAX_W2A_HOME": home}) if False else \
            cli(["--check", "--tokens-file", os.path.join(home, "none.json")], {"MINIMAX_W2A_HOME": home})
        self.assertEqual(r.returncode, 1)
        self.assertIn("no tokens", r.stdout)
        self.assertNotIn("Traceback", r.stderr)

    def test_startup_banner_warns_when_exposed_without_keys(self):
        home = tempfile.mkdtemp()
        p = subprocess.Popen([sys.executable, SCRIPT, "--host", "0.0.0.0", "--port", "0",
                              "--tokens-file", os.path.join(home, "t.json")],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, "HTTPS_PROXY": "", "MINIMAX_W2A_HOME": home})
        try:
            time.sleep(1.5)
        finally:
            p.terminate()
            out = p.communicate(timeout=5)[0]
        self.assertIn("WITHOUT api_keys", out)
        self.assertIn("no tokens yet", out)

    def test_no_secret_in_exception_paths(self):
        # every ProxyError/UpstreamError message must be free of token material
        j = make_jwt(sig="ULTRASECRET")
        t = m.TokenInfo("1+" + j)
        for e in (m.UpstreamError("auth", "x"), m.NoTokenError("y")):
            self.assertNotIn("ULTRASECRET", str(e))
        self.assertNotIn("ULTRASECRET", repr(t))
        self.assertNotIn("ULTRASECRET", str(vars(t).get("name")))


if __name__ == "__main__":
    unittest.main()
