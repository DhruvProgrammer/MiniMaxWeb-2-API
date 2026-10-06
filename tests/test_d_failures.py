"""Agent D - failure analysis: every upstream failure mode must yield a clean, correct, bounded response."""
import http.client
import json
import socket
import time
import unittest

from common import Stack, m


class UpstreamFailures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Stack()

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def err(self, marker, status, code=None, **kw):
        st, d = self.s.chat_json(marker, **kw)
        self.assertEqual(st, status, d)
        self.assertIn("error", d)
        for k in ("message", "type", "code", "param"):
            self.assertIn(k, d["error"])
        if code:
            self.assertEqual(d["error"]["code"], code)
        return d["error"]

    def test_send_500(self):
        self.err("[[send500]]", 502, "upstream_error")

    def test_send_429(self):
        self.err("[[send429]]", 429, "upstream_rate_limited")

    def test_business_error(self):
        e = self.err("[[sendbiz]]", 502, "upstream_error")
        self.assertIn("boom", e["message"])

    def test_auth_business_error_has_login_hint(self):
        e = self.err("[[authbiz]]", 502, "upstream_auth_failed")
        self.assertIn("--login", e["message"])

    def test_non_json(self):
        self.err("[[badjson]]", 502, "upstream_protocol_error")

    def test_missing_chat_id(self):
        self.err("[[nochatid]]", 502, "upstream_protocol_error")

    def test_upstream_hang_is_timeout_and_not_duplicated(self):
        before = len(self.s.mock.chats)
        self.err("[[hang]]", 504, "upstream_timeout")
        time.sleep(3.2)  # let the mock's sleeping handler finish
        self.assertLessEqual(len(self.s.mock.chats) - before, 1, "a read timeout must never be retried (duplicate agent run)")

    def test_detail_always_500(self):
        self.err("[[detail500]]", 502, "upstream_error")

    def test_detail_flaky_is_tolerated(self):
        st, d = self.s.chat_json("[[detailflaky]] x")
        self.assertEqual(st, 200, d)
        self.assertEqual(d["choices"][0]["message"]["content"], "ECHO: [[detailflaky]] x")

    def test_auth_expires_mid_chat(self):
        self.err("[[detailauth]]", 502, "upstream_auth_failed")

    def test_never_any_content(self):
        t0 = time.time()
        self.err("[[empty]]", 504, "upstream_timeout")
        self.assertLess(time.time() - t0, 6)

    def test_error_bodies_never_contain_token(self):
        for mk in ("[[send500]]", "[[authbiz]]", "[[badjson]]", "[[empty]]"):
            st, h, p = self.s.chat(mk)
            for j in self.s.jwts:
                self.assertNotIn(j.encode(), p)
                self.assertNotIn(j.split(".")[2].encode(), p)

    def test_stream_preheader_errors_are_real_http_errors(self):
        st, h, p = self.s.chat("[[send500]]", stream=True)
        self.assertEqual(st, 502)
        self.assertTrue(h["Content-Type"].startswith("application/json"))

    def test_stream_midway_error_is_inband(self):
        st, h, p = self.s.chat("[[detailauth]]", stream=True)
        self.assertEqual(st, 200)
        ev = self.s.parse_sse(p)
        self.assertEqual(ev[-1], "[DONE]")
        errs = [e for e in ev if e != "[DONE]" and "error" in e]
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0]["error"]["code"], "upstream_auth_failed")

    def test_stream_no_content_is_inband_timeout(self):
        st, h, p = self.s.chat("[[empty]]", stream=True)
        ev = self.s.parse_sse(p)
        errs = [e for e in ev if e != "[DONE]" and "error" in e]
        self.assertEqual(errs[0]["error"]["code"], "upstream_timeout")
        self.assertEqual(ev[-1], "[DONE]")

    def test_token_cooldown_then_still_usable_when_only_token(self):
        # authbiz marks the only token bad; the next healthy request must still go through
        self.err("[[authbiz]]", 502)
        st, d = self.s.chat_json("recovered")
        self.assertEqual(st, 200, d)


class StreamingEdge(unittest.TestCase):
    def test_rewrite_live_vs_buffered(self):
        for mode in ("live", "buffered"):
            s = Stack(stream_mode=mode)
            try:
                st, d = s.chat_json("[[rewrite]]")
                self.assertEqual(d["choices"][0]["message"]["content"], "FINAL REWRITTEN ANSWER")
                st, ev, txt = s.stream_text("[[rewrite]]")
                self.assertEqual(st, 200)
                if mode == "buffered":
                    self.assertEqual(txt, "FINAL REWRITTEN ANSWER", "buffered mode must deliver exactly the final text")
                else:
                    self.assertTrue(txt.endswith("FINAL REWRITTEN ANSWER") or txt.startswith("draft"),
                                    "live mode: documented best-effort")
            finally:
                s.close()

    def test_buffered_normal_and_long(self):
        s = Stack(stream_mode="buffered")
        try:
            st, ev, txt = s.stream_text("hello buffered")
            self.assertEqual(txt, "ECHO: hello buffered")
            st, ev, txt = s.stream_text("[[long]]")
            self.assertEqual(len(txt), 200_000)
            self.assertEqual(ev[-1], "[DONE]")
        finally:
            s.close()

    def test_pause_longer_than_settle_truncates_documented_limit(self):
        s = Stack(settle_sec=0.4)
        try:
            st, d = s.chat_json("[[pause]] some answer text here")
            full = "ECHO: [[pause]] some answer text here"
            got = d["choices"][0]["message"]["content"]
            self.assertTrue(full.startswith(got) and len(got) < len(full), "expected truncation with small settle_sec")
        finally:
            s.close()

    def test_pause_shorter_than_settle_is_complete(self):
        s = Stack(settle_sec=2.0, request_timeout_sec=10)
        try:
            st, d = s.chat_json("[[pause]] some answer text here")
            self.assertEqual(d["choices"][0]["message"]["content"], "ECHO: [[pause]] some answer text here")
        finally:
            s.close()

    def test_request_timeout_returns_partial_with_length(self):
        s = Stack(settle_sec=3.0, request_timeout_sec=1.0)
        try:
            st, d = s.chat_json("[[pause]] some answer text here")
            self.assertEqual(st, 200)
            self.assertEqual(d["choices"][0]["finish_reason"], "length")
            self.assertTrue(d["choices"][0]["message"]["content"])
        finally:
            s.close()

    def test_client_disconnect_stops_polling(self):
        s = Stack(settle_sec=3.0, request_timeout_sec=30)
        try:
            sock = socket.create_connection(("127.0.0.1", s.port), timeout=5)
            body = json.dumps({"messages": [{"role": "user", "content": "[[pause]] hold on"}], "stream": True}).encode()
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                         b"Content-Length: %d\r\n\r\n" % len(body) + body)
            sock.recv(4096)
            sock.close()
            time.sleep(1.0)
            n1 = s.mock.detail_calls
            time.sleep(1.0)
            n2 = s.mock.detail_calls
            self.assertLessEqual(n2 - n1, 2, f"polling continued after disconnect ({n1}->{n2})")
        finally:
            s.close()

    def test_semaphore_released_after_disconnect(self):
        s = Stack(max_concurrent=1, settle_sec=3.0, request_timeout_sec=30, queue_wait_sec=0.1)
        try:
            sock = socket.create_connection(("127.0.0.1", s.port), timeout=5)
            body = json.dumps({"messages": [{"role": "user", "content": "[[pause]] a"}], "stream": True}).encode()
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                         b"Content-Length: %d\r\n\r\n" % len(body) + body)
            sock.recv(4096)
            sock.close()
            time.sleep(1.2)
            st, d = s.chat_json("after")
            self.assertEqual(st, 200, d)
        finally:
            s.close()


class Capacity(unittest.TestCase):
    def test_saturation_gives_429(self):
        s = Stack(max_concurrent=1, queue_wait_sec=0.05, settle_sec=0.6)
        try:
            import threading
            res = []
            t = threading.Thread(target=lambda: res.append(s.chat_json("first")[0]))
            t.start()
            time.sleep(0.2)
            st, d = s.chat_json("second")
            self.assertEqual(st, 429)
            self.assertEqual(d["error"]["code"], "too_many_requests")
            t.join()
            self.assertEqual(res, [200])
        finally:
            s.close()

    def test_upstream_down(self):
        s = Stack()
        try:
            s.mock.stop()
            t0 = time.time()
            st, d = s.chat_json("anyone?")
            self.assertEqual(st, 502)
            self.assertEqual(d["error"]["code"], "upstream_unreachable")
            self.assertLess(time.time() - t0, 6)
        finally:
            s.srv.shutdown(); s.srv.server_close()

    def test_no_tokens_and_only_expired_tokens(self):
        from mock_upstream import make_jwt
        s = Stack(tokens=[f"1+{make_jwt(exp_delta=-50, sig='dead')}"])
        try:
            st, d = s.chat_json("hi")
            self.assertEqual(st, 503)
            self.assertEqual(d["error"]["code"], "no_token")
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
