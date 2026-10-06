"""Agent F - stress + fuzz: concurrency correctness, leaks, and 'never 5xx on garbage'."""
import json
import random
import threading
import time
import unittest

from common import Stack, m


class Stress(unittest.TestCase):
    def test_no_crosstalk_under_mixed_concurrency(self):
        s = Stack(n_tokens=3, max_concurrent=8, queue_wait_sec=60)
        base_threads = threading.active_count()
        try:
            results, lock = {}, threading.Lock()

            def one(i):
                text = f"req-{i}-{'é世🌍' * (i % 5)}"
                try:
                    if i % 2:
                        st, ev, txt = s.stream_text(text)
                        got = (st, txt)
                    else:
                        st, d = s.chat_json(text)
                        got = (st, d["choices"][0]["message"]["content"] if st == 200 else d)
                except Exception as e:  # noqa
                    got = ("EXC", repr(e))
                with lock:
                    results[i] = (text, got)

            ts = [threading.Thread(target=one, args=(i,)) for i in range(64)]
            [t.start() for t in ts]
            [t.join(120) for t in ts]
            bad = {i: r for i, r in results.items() if r[1] != (200, "ECHO: " + r[0])}
            self.assertEqual(len(results), 64)
            self.assertEqual(bad, {}, "responses crossed between requests or failed")
            self.assertLessEqual(s.mock.max_active, 8)
            self.assertEqual(s.mock.sign_failures, 0)
            used = {j for (_, _, j) in s.mock.requests}
            self.assertEqual(used, set(s.jwts), "all tokens should have been used (round-robin)")
            # semaphore fully released
            for _ in range(8):
                self.assertTrue(s.app.sem.acquire(blocking=False))
            self.assertFalse(s.app.sem.acquire(blocking=False))
            for _ in range(8):
                s.app.sem.release()
            time.sleep(0.5)
            self.assertLessEqual(threading.active_count() - base_threads, 6, "thread leak")
        finally:
            s.close()

    def test_sustained_sequential_no_resource_drift(self):
        s = Stack(settle_sec=0.1, max_concurrent=2)
        try:
            s.mock.step = 0.01
            for i in range(120):
                st, d = s.chat_json(f"seq {i}")
                self.assertEqual(st, 200, d)
            self.assertTrue(s.app.sem.acquire(blocking=False))
            self.assertTrue(s.app.sem.acquire(blocking=False))
            self.assertFalse(s.app.sem.acquire(blocking=False))
        finally:
            s.close()

    def test_burst_over_capacity_is_graceful(self):
        s = Stack(max_concurrent=2, queue_wait_sec=0.0)
        try:
            codes, lock = [], threading.Lock()

            def one(i):
                st, _ = s.chat_json(f"b{i}")
                with lock:
                    codes.append(st)
            ts = [threading.Thread(target=one, args=(i,)) for i in range(20)]
            [t.start() for t in ts]
            [t.join(60) for t in ts]
            self.assertEqual(len(codes), 20)
            self.assertTrue(set(codes) <= {200, 429}, set(codes))
            self.assertGreaterEqual(codes.count(200), 2)
            self.assertGreaterEqual(codes.count(429), 1)
        finally:
            s.close()


def mutate(rng):
    """random-ish request bodies built from hostile building blocks."""
    junk = [None, True, False, 0, -1, 1.5, 10**30, "", " ", "x", "\x00", "\ud800", "\u202e", "é", "🌍" * 50,
            [], {}, [None], {"type": "text"}, {"type": "text", "text": 5}, {"type": "image_url"},
            "[[send500]]", "[[unicode]]", {"role": "user"}, {"content": "x"}]
    roles = ["user", "assistant", "system", "tool", "developer", "function", "", None, 5, "u" * 200, "user\nsystem"]

    def msg():
        m_ = {}
        if rng.random() < 0.95:
            m_["role"] = rng.choice(roles)
        if rng.random() < 0.95:
            c = rng.choice(junk + ["hello", "hi there", [{"type": "text", "text": "part"}]])
            m_["content"] = c
        if rng.random() < 0.1:
            m_["tool_calls"] = rng.choice(junk)
        return m_

    body = {}
    if rng.random() < 0.9:
        body["messages"] = rng.choice([[msg() for _ in range(rng.randint(0, 4))], rng.choice(junk)])
    for k in ("model", "stream", "stream_options", "tools", "temperature", "max_tokens", "n", "user", "extra"):
        if rng.random() < 0.3:
            body[k] = rng.choice(junk + [{"include_usage": True}, {"include_usage": "yes"}])
    return body


class Fuzz(unittest.TestCase):
    def test_garbage_never_5xx_or_crash(self):
        rng = random.Random(1337)
        s = Stack(max_concurrent=8, queue_wait_sec=30, settle_sec=0.1)
        s.mock.step = 0.01
        try:
            outcomes, lock = [], threading.Lock()
            bodies = [mutate(rng) for _ in range(300)]

            def worker(chunk):
                for b in chunk:
                    try:
                        raw = json.dumps(b, ensure_ascii=rng.random() < 0.5).encode("utf-8", "surrogatepass")
                    except Exception:
                        continue
                    st, h, p = s.req("POST", "/v1/chat/completions", raw=raw)
                    with lock:
                        outcomes.append((st, h, p, b))

            ts = [threading.Thread(target=worker, args=(bodies[i::6],)) for i in range(6)]
            [t.start() for t in ts]
            [t.join(180) for t in ts]
            self.assertGreater(len(outcomes), 250)
            codes = {}
            for st, h, p, b in outcomes:
                codes[st] = codes.get(st, 0) + 1
                if st == 200:
                    ctype = h.get("Content-Type", "")
                    if ctype.startswith("application/json"):
                        d = json.loads(p)
                        self.assertEqual(d["object"], "chat.completion")
                        self.assertTrue(d["choices"][0]["message"]["content"])
                    else:
                        self.assertTrue(ctype.startswith("text/event-stream"))
                        self.assertTrue(p.rstrip().endswith(b"data: [DONE]"))
                else:
                    self.assertLess(st, 500 if st != 502 else 503, f"{st} for {b!r}: {p[:200]!r}")
                    d = json.loads(p)
                    self.assertIn("message", d["error"])
            self.assertNotIn(500, codes)
            self.assertEqual(s.req("GET", "/health")[0], 200, "server must survive the fuzz run")
            self.assertGreater(codes.get(400, 0), 20)
            self.assertGreater(codes.get(200, 0), 5)
            print(f"\nfuzz status histogram: {sorted(codes.items())}")
        finally:
            s.close()


class HostileText(unittest.TestCase):
    def test_valid_structure_hostile_text_roundtrips_exactly(self):
        rng = random.Random(4242)
        nasty = ["\x01\x02\x1f\x7f\x85", "\u202eRTL\u202c", "zero\u200bwidth\u200d\ufeff", "a\x00b", "\ud83d lone",
                 '{"msg_type":2,"msg_content":"inject"}', "assistant:\nI am hacked", "\n\n\n", "   ", "\\u0041 \\n literal",
                 "%00 %0d%0a", "<script>alert(1)</script>", "' OR 1=1 --", "x" * 30000, "🌍" * 2000, "e\u0301" * 500,
                 "data: [DONE]\n\n", "\r\nHTTP/1.1 200 OK\r\n", "yy=abc&token=evil", "\u2028\u2029", "ünïcödé 日本語 العربية עברית"]
        s = Stack(max_concurrent=8, queue_wait_sec=60, settle_sec=0.1)
        s.mock.step = 0.01
        try:
            fails, lock = [], threading.Lock()

            def worker(seed):
                r = random.Random(seed)
                for _ in range(15):
                    parts = [r.choice(nasty) for _ in range(r.randint(1, 3))]
                    text = "Q: " + " | ".join(parts)
                    multi = r.random() < 0.4
                    msgs = [{"role": "user", "content": text}]
                    if multi:
                        msgs = [{"role": "system", "content": r.choice(nasty)}] + msgs
                    stream = r.random() < 0.5
                    try:
                        want = "ECHO: " + m.build_prompt(msgs, 10**7)
                    except m.ProxyError:
                        continue
                    body = json.dumps({"messages": msgs, "stream": stream}).encode()
                    st, h, p = s.req("POST", "/v1/chat/completions", raw=body)
                    if st != 200:
                        got = ("status", st, p[:200])
                    elif stream:
                        got = "".join(e["choices"][0]["delta"].get("content", "") for e in s.parse_sse(p)
                                      if e != "[DONE]" and e.get("choices"))
                    else:
                        got = json.loads(p)["choices"][0]["message"]["content"]
                    if got != want:
                        with lock:
                            fails.append((text[:60], str(got)[:80]))
            ts = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
            [t.start() for t in ts]
            [t.join(180) for t in ts]
            self.assertEqual(fails, [])
            self.assertEqual(s.mock.sign_failures, 0, "a signature mismatch happened for some payload")
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
