"""Agent C - end-to-end behaviour through the real HTTP server against the strict mock."""
import json
import threading
import time
import unittest

from common import Stack, m, make_jwt


class Normal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Stack()

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_health_and_models(self):
        st, _, p = self.s.req("GET", "/health")
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(p)["tokens"], 1)
        st, _, p = self.s.req("GET", "/v1/models")
        d = json.loads(p)
        self.assertEqual(d["object"], "list")
        self.assertEqual(d["data"][0]["id"], "minimax-agent")
        st, _, p = self.s.req("GET", "/v1/models/anything")
        self.assertEqual(json.loads(p)["id"], "anything")

    def test_non_stream_shape(self):
        st, d = self.s.chat_json("hello there")
        self.assertEqual(st, 200)
        self.assertEqual(d["object"], "chat.completion")
        self.assertTrue(d["id"].startswith("chatcmpl-"))
        self.assertEqual(d["model"], "x")
        ch = d["choices"][0]
        self.assertEqual(ch["message"], {"role": "assistant", "content": "ECHO: hello there"})
        self.assertEqual(ch["finish_reason"], "stop")
        u = d["usage"]
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])

    def test_stream_matches_non_stream(self):
        st, ev, txt = self.s.stream_text("stream me please")
        self.assertEqual(st, 200)
        self.assertEqual(txt, "ECHO: stream me please")
        self.assertEqual(ev[-1], "[DONE]")
        self.assertEqual(ev[0]["choices"][0]["delta"].get("role"), "assistant")
        fin = [e for e in ev if e != "[DONE]" and e["choices"] and e["choices"][0]["finish_reason"]]
        self.assertEqual(len(fin), 1)
        self.assertEqual(fin[0]["choices"][0]["finish_reason"], "stop")
        ids = {e["id"] for e in ev if e != "[DONE]"}
        self.assertEqual(len(ids), 1)
        self.assertGreater(len([e for e in ev if e != "[DONE]" and e["choices"][0]["delta"].get("content")]), 1,
                           "should arrive incrementally, not in one chunk")

    def test_stream_headers(self):
        st, h, _ = self.s.chat("hdr", stream=True)
        self.assertTrue(h["Content-Type"].startswith("text/event-stream"))
        self.assertEqual(h.get("Cache-Control"), "no-cache")

    def test_include_usage(self):
        st, ev, _ = self.s.stream_text("usage", extra={"stream_options": {"include_usage": True}})
        u = [e for e in ev if e != "[DONE]" and e.get("usage")]
        self.assertEqual(len(u), 1)
        self.assertEqual(u[0]["choices"], [])

    def test_unicode_roundtrip(self):
        st, d = self.s.chat_json("[[unicode]] 你好 🌍")
        self.assertEqual(d["choices"][0]["message"]["content"], "你好, мир 🌍 café — \"quotes\" \\backslash\nnewline\ttab ✓")
        st, ev, txt = self.s.stream_text("[[unicode]]")
        self.assertEqual(txt, "你好, мир 🌍 café — \"quotes\" \\backslash\nnewline\ttab ✓")

    def test_prompt_with_unicode_is_signed_and_delivered_intact(self):
        text = "日本語 🌍 \"q\" \\ \n 𝒳"
        st, d = self.s.chat_json(text)
        self.assertEqual(st, 200)  # would be 502 (401 from mock) if signature mismatched
        self.assertEqual(d["choices"][0]["message"]["content"], "ECHO: " + text)
        self.assertEqual(self.s.mock.sign_failures, 0)

    def test_multi_turn_reaches_upstream_as_transcript(self):
        body = {"messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q1"},
                             {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}]}
        st, _, p = self.s.req("POST", "/v1/chat/completions", body)
        self.assertEqual(st, 200)
        sent = [b for (u, b, _) in self.s.mock.requests if u.endswith("send_msg")][-1]["text"]
        self.assertIn("system: SYS", sent)
        self.assertIn("user: Q2", sent)
        self.assertEqual(sent.strip().splitlines()[-1], "assistant:")

    def test_send_payload_shape(self):
        self.s.chat_json("shape")
        b = [b for (u, b, _) in self.s.mock.requests if u.endswith("send_msg")][-1]
        self.assertEqual((b["msg_type"], b["chat_type"], b["attachments"], b["sub_agent_ids"]), (1, 1, [], []))

    def test_model_name_echo_and_default(self):
        st, _, p = self.s.req("POST", "/v1/chat/completions",
                              {"model": "gpt-4o", "messages": [{"role": "user", "content": "m"}]})
        self.assertEqual(json.loads(p)["model"], "gpt-4o")
        st, _, p = self.s.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "m"}]})
        self.assertEqual(json.loads(p)["model"], "minimax-agent")

    def test_ignored_params_accepted(self):
        st, d = self.s.chat_json("p", extra={"temperature": 0.2, "max_tokens": 5, "top_p": 1, "n": 1, "user": "u",
                                             "tools": [{"type": "function", "function": {"name": "f"}}]})
        self.assertEqual(st, 200)

    def test_message_pick_last_vs_first(self):
        st, d = self.s.chat_json("[[multi]] hey")
        self.assertEqual(d["choices"][0]["message"]["content"], "ECHO: [[multi]] hey")

    def test_gzip_upstream(self):
        st, d = self.s.chat_json("[[gzip]] zip")
        self.assertEqual(st, 200)

    def test_lone_surrogate_in_upstream_answer_does_not_500(self):
        st, d = self.s.chat_json("[[surrogate]]")
        self.assertEqual(st, 200, d)
        self.assertTrue(d["choices"][0]["message"]["content"].startswith("ok \U0001F600 and broken"))
        st, ev, txt = self.s.stream_text("[[surrogate]]")
        self.assertEqual(st, 200)
        self.assertEqual(ev[-1], "[DONE]")
        self.assertTrue(txt.startswith("ok \U0001F600 and broken"))

    def test_lone_surrogate_in_model_name_does_not_500(self):
        raw = b'{"model":"\\ud800x","messages":[{"role":"user","content":"hi"}]}'
        st, _, p = self.s.req("POST", "/v1/chat/completions", raw=raw)
        self.assertEqual(st, 200, p)
        raw = b'{"model":"\\ud800x","stream":true,"messages":[{"role":"user","content":"hi"}]}'
        st, _, p = self.s.req("POST", "/v1/chat/completions", raw=raw)
        self.assertEqual(st, 200)

    def test_long_answer(self):
        st, d = self.s.chat_json("[[long]]")
        self.assertEqual(len(d["choices"][0]["message"]["content"]), 200_000)
        st, ev, txt = self.s.stream_text("[[long]]")
        self.assertEqual(len(txt), 200_000)


class Behaviours(unittest.TestCase):
    def test_retry_on_503_then_success(self):
        s = Stack()
        try:
            st, d = s.chat_json("[[send503once]] go")
            self.assertEqual(st, 200)
            # exactly one failed attempt + one successful retry (never a duplicate chat)
            self.assertEqual(s.mock.counters["503once:[[send503once]] go"], 2)
            self.assertEqual(len(s.mock.chats), 1)
        finally:
            s.close()

    def test_token_failover_first_token_rejected(self):
        s = Stack(n_tokens=2)
        try:
            s.mock.valid.discard(s.jwts[0])           # first token is dead upstream
            for _ in range(4):
                st, d = s.chat_json("hello")
                self.assertEqual(st, 200, d)
            self.assertEqual({j for (_, _, j) in s.mock.requests}, {s.jwts[1]})
        finally:
            s.close()

    def test_hot_reload_token_while_running(self):
        s = Stack(tokens=[])
        try:
            st, d = s.chat_json("no token yet")
            self.assertEqual(st, 503)
            self.assertEqual(d["error"]["code"], "no_token")
            m.write_tokens_file(__import__("pathlib").Path(s.cfg["tokens_file"]), [f"450000000000+{s.jwts[0]}"])
            st, d = s.chat_json("now with token")
            self.assertEqual(st, 200, d)
        finally:
            s.close()

    def test_concurrency_limit_respected_upstream(self):
        s = Stack(max_concurrent=3, queue_wait_sec=10)
        try:
            out = []
            ts = [threading.Thread(target=lambda i=i: out.append(s.chat_json(f"c{i}")[0])) for i in range(9)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            self.assertEqual(out.count(200), 9)
            self.assertLessEqual(s.mock.max_active, 3)
        finally:
            s.close()

    def test_check_flow_user_info(self):
        s = Stack()
        try:
            tok = s.app.pool.acquire()
            r = s.app.upstream.call(m.USERINFO_URI, {}, tok, method="GET")
            self.assertIn("data", r)
        finally:
            s.close()

    def test_connection_keepalive_multiple_requests(self):
        import http.client
        s = Stack()
        try:
            c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=10)
            for _ in range(3):
                c.request("GET", "/v1/models")
                r = c.getresponse()
                r.read()
                self.assertEqual(r.status, 200)
            c.close()
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
