"""Agent G - real CLI subprocess flow + official OpenAI SDK compatibility."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from common import m, make_jwt
from mock_upstream import Mock

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = str(ROOT / "minimax_web2api.py")


def run(args, env=None, timeout=20):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True, timeout=timeout, env=e)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class CLI(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.tf = os.path.join(self.home, "tokens.json")
        self.env = {"MINIMAX_W2A_HOME": self.home, "HTTPS_PROXY": "", "https_proxy": ""}

    def test_snippet(self):
        r = run(["--print-snippet"], self.env)
        self.assertEqual(r.returncode, 0)
        self.assertIn("localStorage.getItem('_token')", r.stdout)
        self.assertIn("--add-token", r.stdout)

    def test_add_token_valid_invalid_and_perms(self):
        j = make_jwt()
        r = run(["--tokens-file", self.tf, "--add-token", "123456+" + j], self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn(j, r.stdout + r.stderr)             # never echoes the secret
        self.assertEqual(oct(os.stat(self.tf).st_mode & 0o777), "0o600")
        r = run(["--tokens-file", self.tf, "--add-token", "garbage"], self.env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("invalid token", r.stderr)
        r = run(["--tokens-file", self.tf, "--add-token", "1+" + make_jwt(exp_delta=-9, sig="x")], self.env)
        self.assertEqual(r.returncode, 2)

    def test_bad_config_exit_code(self):
        cfgp = os.path.join(self.home, "c.json")
        Path(cfgp).write_text('{"port": 99999}')
        r = run(["--config", cfgp], self.env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("config error", r.stderr)
        r = run(["--config", os.path.join(self.home, "nope.json")], self.env)
        self.assertEqual(r.returncode, 2)

    def test_login_without_playwright_is_friendly(self):
        r = run(["--tokens-file", self.tf, "--login"], {**self.env, "PYTHONPATH": ""})
        if r.returncode == 0:
            self.skipTest("playwright present")
        self.assertIn("print-snippet", r.stderr + r.stdout)
        self.assertNotIn("Traceback", r.stderr)

    def test_check_and_serve_flow(self):
        jwt = make_jwt(user_id="450999")
        mock = Mock(valid_jwts=[jwt])
        base = mock.start()
        port = free_port()
        proc = None
        try:
            common = ["--tokens-file", self.tf, "--base-url", base]
            self.assertEqual(run(common + ["--add-token", "450999+" + jwt], self.env).returncode, 0)
            r = run(common + ["--check"], self.env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("OK", r.stdout)
            self.assertNotIn(jwt, r.stdout + r.stderr)
            # dead token -> --check fails
            mock.valid.clear()
            r = run(common + ["--check"], self.env)
            self.assertEqual(r.returncode, 1)
            self.assertIn("FAIL", r.stdout)
            mock.valid.add(jwt)

            # serve for real, then hit it with the official OpenAI SDK
            proc = subprocess.Popen([sys.executable, SCRIPT] + common + ["--port", str(port)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    env={**os.environ, **self.env})
            for _ in range(50):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                self.fail("server did not start")
            try:
                from openai import OpenAI
            except ImportError:
                self.skipTest("openai SDK not installed")
            client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="anything", max_retries=0, timeout=30)
            self.assertEqual(client.models.list().data[0].id, "minimax-agent")
            r = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "你好 SDK"}])
            self.assertEqual(r.choices[0].message.content, "ECHO: 你好 SDK")
            self.assertEqual(r.choices[0].finish_reason, "stop")
            self.assertGreater(r.usage.total_tokens, 0)
            parts, fin = [], None
            for ev in client.chat.completions.create(model="x", stream=True, messages=[
                    {"role": "system", "content": "s"}, {"role": "user", "content": "stream SDK"}]):
                if ev.choices:
                    parts.append(ev.choices[0].delta.content or "")
                    fin = ev.choices[0].finish_reason or fin
            self.assertTrue("".join(parts).startswith("ECHO: The following is a conversation transcript"))
            self.assertIn("stream SDK", "".join(parts))
            self.assertEqual(fin, "stop")
            self.assertGreater(len(parts), 2)
            # SDK error mapping: empty messages -> BadRequestError
            import openai
            with self.assertRaises(openai.BadRequestError):
                client.chat.completions.create(model="x", messages=[])
            # token expires upstream -> SDK sees a clean error, not a hang
            mock.valid.clear()
            with self.assertRaises(openai.APIStatusError) as cm:
                client.chat.completions.create(model="x", messages=[{"role": "user", "content": "hi"}])
            self.assertEqual(cm.exception.status_code, 502)
            self.assertIn("--login", str(cm.exception))
        finally:
            if proc:
                proc.terminate()
                try:
                    out = proc.communicate(timeout=5)[0]
                except Exception:
                    proc.kill(); out = ""
                self.assertNotIn(jwt, out, "server log leaked the token")
                self.assertNotIn("Traceback", out)
            mock.stop()


if __name__ == "__main__":
    unittest.main()
