"""Agent E - adversarial / security: try to break it."""
import json
import logging
import socket
import time
import unittest

from common import Stack, m


def raw_exchange(port, data: bytes, read_timeout=3.0, half_close=False):
    s = socket.create_connection(("127.0.0.1", port), timeout=read_timeout)
    s.sendall(data)
    if half_close:
        s.shutdown(socket.SHUT_WR)
    out = b""
    t0 = time.time()
    try:
        while time.time() - t0 < read_timeout:
            chunk = s.recv(65536)
            if not chunk:
                break
            out += chunk
    except socket.timeout:
        pass
    s.close()
    return out


def status_lines(blob: bytes):
    # count anywhere in the stream: a smuggled 2nd response is glued right after the 1st body (no CRLF before it)
    import re
    return re.findall(rb"HTTP/1\.[01] \d{3} [A-Za-z ]*", blob)


class Auth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Stack(api_keys=["sk-right", "sk-other"])

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def status(self, headers, path="/v1/models", method="GET"):
        return self.s.req(method, path, headers=headers)[0]

    def test_matrix(self):
        self.assertEqual(self.status({}), 401)
        self.assertEqual(self.status({"Authorization": "Bearer wrong"}), 401)
        self.assertEqual(self.status({"Authorization": "Bearer sk-righ"}), 401)       # prefix
        self.assertEqual(self.status({"Authorization": "Bearer sk-rightX"}), 401)     # superstring
        self.assertEqual(self.status({"Authorization": "Bearer "}), 401)
        self.assertEqual(self.status({"Authorization": "Bearer"}), 401)
        self.assertEqual(self.status({"Authorization": "sk-right"}), 401)             # missing scheme
        self.assertEqual(self.status({"x-api-key": ""}), 401)
        self.assertEqual(self.status({"Authorization": "Bearer sk-right"}), 200)
        self.assertEqual(self.status({"Authorization": "bearer sk-other"}), 200)
        self.assertEqual(self.status({"x-api-key": "sk-right"}), 200)

    def test_chat_requires_key_before_parsing_body(self):
        st, _, p = self.s.req("POST", "/v1/chat/completions", raw=b"{not json")
        self.assertEqual(st, 401, "unauthenticated callers must not learn anything from body parsing")

    def test_health_is_public_but_minimal(self):
        st, _, p = self.s.req("GET", "/health")
        self.assertEqual(st, 200)
        self.assertEqual(set(json.loads(p)), {"status", "service", "version", "tokens"})

    def test_401_does_not_echo_key(self):
        st, _, p = self.s.req("GET", "/v1/models", headers={"Authorization": "Bearer sk-SECRETGUESS"})
        self.assertNotIn(b"SECRETGUESS", p)

    def test_no_request_smuggling_via_unread_body(self):
        inner = (b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-right\r\n\r\n")
        req = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
               b"Content-Length: %d\r\n\r\n" % len(inner)) + inner
        out = raw_exchange(self.s.port, req)
        self.assertEqual(len(status_lines(out)), 1, f"body was parsed as a second request: {out[:400]!r}")
        self.assertIn(b"401", status_lines(out)[0])

    def test_smuggling_after_other_early_errors(self):
        inner = b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
        for head in (b"POST /nope HTTP/1.1", b"PUT /v1/chat/completions HTTP/1.1", b"POST /v1/models HTTP/1.1"):
            req = head + b"\r\nHost: localhost\r\nAuthorization: Bearer sk-right\r\nContent-Type: application/json\r\n" \
                  b"Content-Length: %d\r\n\r\n" % len(inner) + inner
            out = raw_exchange(self.s.port, req)
            self.assertEqual(len(status_lines(out)), 1, f"{head!r}: {out[:300]!r}")


class HostAndBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Stack(allowed_hosts=["my.lan"])

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def st(self, host):
        return self.s.req("GET", "/v1/models", headers={"Host": host})[0]

    def test_dns_rebinding_and_host_tricks(self):
        p = self.s.port
        for ok in ("localhost", f"localhost:{p}", "127.0.0.1", f"127.0.0.1:{p}", f"[::1]:{p}", "LOCALHOST", "my.lan", "my.lan:80"):
            self.assertEqual(self.st(ok), 200, ok)
        for bad in ("evil.com", f"evil.com:{p}", "localhost.evil.com", "evil.com#localhost", "localhost@evil.com",
                    "127.0.0.1.evil.com", "0.0.0.0", "192.168.1.5", "", "my.lan.evil.com"):
            self.assertEqual(self.st(bad), 403, repr(bad))

    def test_missing_host_header(self):
        out = raw_exchange(self.s.port, b"GET /v1/models HTTP/1.0\r\n\r\n")
        self.assertIn(b" 403 ", out.split(b"\r\n")[0])

    def test_content_type_csrf_guard(self):
        body = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
        for ct in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data", "", "application/jsonp"):
            h = {"Content-Type": ct} if ct else {}
            c = __import__("http.client").client.HTTPConnection("127.0.0.1", self.s.port)
            c.request("POST", "/v1/chat/completions", body=body, headers=h)
            r = c.getresponse(); r.read(); c.close()
            self.assertEqual(r.status, 415, repr(ct))
        for ct in ("application/json", "application/json; charset=utf-8", "APPLICATION/JSON"):
            st, _, _ = self.s.req("POST", "/v1/chat/completions", raw=body, headers={"Content-Type": ct})
            self.assertEqual(st, 200, ct)

    def test_no_cors_by_default(self):
        st, h, _ = self.s.req("GET", "/v1/models", headers={"Origin": "https://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", h)
        st, h, _ = self.s.req("OPTIONS", "/v1/chat/completions", headers={"Origin": "https://evil.example",
                                                                         "Access-Control-Request-Method": "POST"})
        self.assertNotIn("Access-Control-Allow-Origin", h)

    def test_cors_allowlist(self):
        s = Stack(cors_origins=["http://localhost:3000"])
        try:
            st, h, _ = s.req("OPTIONS", "/v1/chat/completions", headers={"Origin": "http://localhost:3000"})
            self.assertEqual(h.get("Access-Control-Allow-Origin"), "http://localhost:3000")
            st, h, _ = s.req("GET", "/v1/models", headers={"Origin": "http://localhost:3000.evil.com"})
            self.assertNotIn("Access-Control-Allow-Origin", h)
        finally:
            s.close()

    def test_hosts_not_checked_when_api_keys_set(self):
        s = Stack(api_keys=["k"])
        try:
            st, _, _ = s.req("GET", "/v1/models", headers={"Host": "anything.example", "Authorization": "Bearer k"})
            self.assertEqual(st, 200)
        finally:
            s.close()


class MalformedInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Stack(max_body_bytes=2048)

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def post(self, raw, headers=None):
        return self.s.req("POST", "/v1/chat/completions", raw=raw, headers=headers)

    def test_bodies(self):
        cases = {
            b"": 400, b"null": 400, b"[]": 400, b'"str"': 400, b"123": 400, b"{": 400, b"\xff\xfe\x00": 400,
            b"\xef\xbb\xbf{}": 400, b'{"messages": []}': 400, b'{"messages": "x"}': 400,
            b'{"messages":[{"role":"user","content":"x"}],"stream":"yes"}': 400,
            b'{"messages":[{"role":"user","content":"x"}],"stream":1}': 400,
        }
        for raw, want in cases.items():
            st, _, p = self.post(raw)
            self.assertEqual(st, want, raw[:40])
            json.loads(p)  # error body is always valid JSON

    def test_deeply_nested_json_is_400_not_crash(self):
        s = Stack()
        try:
            for raw in (b"[" * 100000, b'{"a":' * 50000 + b"1" + b"}" * 50000):
                st, _, p = s.req("POST", "/v1/chat/completions", raw=raw)
                self.assertEqual(st, 400)
            self.assertEqual(s.req("GET", "/health")[0], 200)
        finally:
            s.close()

    def test_odd_but_valid_optional_fields(self):
        base = {"messages": [{"role": "user", "content": "hi"}]}
        for extra in ({"model": 5}, {"model": None}, {"model": "x" * 300}, {"stream_options": "bad"},
                      {"stream_options": None}, {"stream": None}, {"tools": "nope"}, {"temperature": "hot"}):
            st, _, p = self.post(json.dumps({**base, **extra}).encode())
            self.assertEqual(st, 200, f"{extra!r}: {p[:200]!r}")

    def test_body_too_large(self):
        st, _, p = self.post(b'{"messages":[{"role":"user","content":"' + b"a" * 5000 + b'"}]}')
        self.assertEqual(st, 413)

    def test_huge_content_length_not_read(self):
        out = raw_exchange(self.s.port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                           b"Content-Type: application/json\r\nContent-Length: 999999999999\r\n\r\n")
        self.assertIn(b" 413 ", out.split(b"\r\n")[0])

    def test_bad_content_length_values(self):
        for cl in (b"-5", b"1e3", b"0x10", b"abc", b"1 2", b"+5", b"\xd9\xa5", b"\xb2", b"\xb9", b"9" * 40):  # last: arabic-indic digit
            out = raw_exchange(self.s.port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                               b"Content-Type: application/json\r\nContent-Length: " + cl + b"\r\n\r\n{}", read_timeout=1.5)
            line = out.split(b"\r\n")[0]
            self.assertTrue(b" 400 " in line or b" 411 " in line or b" 413 " in line, f"{cl!r}: {line!r}")
            self.assertNotIn(b" 500 ", line)

    def test_chunked_rejected(self):
        out = raw_exchange(self.s.port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                           b"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n")
        self.assertIn(b" 411 ", out.split(b"\r\n")[0])
        self.assertEqual(len(status_lines(out)), 1)

    def test_cl_and_te_together_no_desync(self):
        out = raw_exchange(self.s.port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                           b"Content-Type: application/json\r\nContent-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\n"
                           b"0\r\n\r\nGET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        self.assertEqual(len(status_lines(out)), 1, out[:300])

    def test_truncated_body(self):
        out = raw_exchange(self.s.port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                           b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{\"mes", half_close=True)
        self.assertIn(b" 400 ", out.split(b"\r\n")[0])

    def test_methods_and_paths(self):
        for method, path, want in (("PUT", "/v1/chat/completions", 405), ("DELETE", "/v1/chat/completions", 405),
                                   ("GET", "/v1/chat/completions", 405), ("POST", "/v1/models", 404),
                                   ("GET", "/nope", 404), ("GET", "//v1/models", 200), ("GET", "/v1/models/../x", 200),
                                   ("GET", "/v1/chat", 404), ("GET", "/v1/models?x=1", 200),
                                   ("POST", "/v1/chat/completions/", 400)):
            st, _, p = self.s.req(method, path, raw=b"{}" if method in ("POST", "PUT", "DELETE") else None)
            self.assertEqual(st, want, f"{method} {path}")
            json.loads(p)

    def test_slow_body_times_out_and_server_survives(self):
        old = m.Handler.timeout
        m.Handler.timeout = 1
        try:
            s = socket.create_connection(("127.0.0.1", self.s.port), timeout=5)
            s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                      b"Content-Length: 50\r\n\r\n{\"a\"")
            t0 = time.time()
            data = s.recv(4096)
            self.assertLess(time.time() - t0, 4)
            self.assertTrue(data == b"" or b" 408 " in data.split(b"\r\n")[0], data[:100])
            s.close()
        finally:
            m.Handler.timeout = old
        self.assertEqual(self.s.req("GET", "/health")[0], 200)


class LogHygiene(unittest.TestCase):
    def test_no_secrets_or_control_chars_in_logs(self):
        records = []

        class H(logging.Handler):
            def emit(self, rec):
                records.append(rec.getMessage())

        h = H()
        lg = logging.getLogger("minimax-web2api")
        old = lg.level
        lg.setLevel(logging.DEBUG)
        lg.addHandler(h)
        s = Stack(log_requests=True, api_keys=["sk-topsecret"])
        try:
            hd = {"Authorization": "Bearer sk-topsecret"}
            s.chat_json("hello", headers=hd)
            s.chat_json("[[authbiz]]", headers=hd)
            s.chat_json("[[send500]]", headers=hd)
            s.req("GET", "/v1/models", headers={"Authorization": "Bearer sk-wrongguess"})
            raw_exchange(s.port, b"GET /\x1b[2J\x1b[31mINJECT HTTP/1.1\r\nHost: localhost\r\n\r\n", read_timeout=1)
            time.sleep(0.2)
        finally:
            s.close()
            lg.removeHandler(h)
            lg.setLevel(old)
        blob = "\n".join(records)
        self.assertTrue(records)
        for j in s.jwts:
            self.assertNotIn(j, blob)
            self.assertNotIn(j.split(".")[2], blob)
            self.assertNotIn(j.split(".")[1], blob)
        self.assertNotIn("sk-topsecret", blob)
        self.assertNotIn("sk-wrongguess", blob)
        self.assertNotIn("\x1b", blob, "control characters must be neutralised in logs")
        self.assertNotIn("\r", blob)


class Defaults(unittest.TestCase):
    def test_safe_defaults(self):
        self.assertEqual(m.DEFAULTS["host"], "127.0.0.1")
        self.assertEqual(m.DEFAULTS["cors_origins"], [])
        self.assertLessEqual(m.DEFAULTS["max_concurrent"], 8)


if __name__ == "__main__":
    unittest.main()
