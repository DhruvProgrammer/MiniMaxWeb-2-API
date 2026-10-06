import http.client
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimax_web2api as m  # noqa: E402
from mock_upstream import Mock, make_jwt  # noqa: E402

FAST = {
    "port": 0, "poll_interval_sec": 0.03, "settle_sec": 0.4, "first_content_timeout_sec": 2.0,
    "request_timeout_sec": 5.0, "upstream_timeout_sec": 1.5, "retry_attempts": 2, "retry_delay_sec": 0.02,
    "poll_error_limit": 3, "queue_wait_sec": 0.2, "keepalive_sec": 0.1, "log_requests": False,
    "token_cooldown_sec": 60,
}


class Stack:
    """mock upstream + proxy, torn down cleanly."""

    def __init__(self, n_tokens=1, **cfg_over):
        self.tmp = tempfile.mkdtemp()
        self.jwts = [make_jwt(user_id=str(450000000000 + i), device_id=str(111 + i), sig=f"s{i}") for i in range(n_tokens)]
        self.mock = Mock(valid_jwts=self.jwts)
        self.upstream_url = self.mock.start()
        cfg = dict(m.DEFAULTS)
        cfg.update(FAST)
        cfg.update({"base_url": self.upstream_url, "tokens_file": os.path.join(self.tmp, "tokens.json"),
                    "tokens": [f"{450000000000 + i}+{j}" for i, j in enumerate(self.jwts)]})
        cfg.update(cfg_over)
        self.cfg = m.validate_config(cfg)
        self.app = m.App(self.cfg)
        self.srv = m.Server(("127.0.0.1", 0), self.app)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.mock.stop()

    def req(self, method, path, body=None, headers=None, raw=None, timeout=15):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        payload = r.read()
        c.close()
        return r.status, dict(r.getheaders()), payload

    def chat(self, content, stream=False, extra=None, headers=None):
        body = {"model": "x", "messages": [{"role": "user", "content": content}], "stream": stream}
        body.update(extra or {})
        return self.req("POST", "/v1/chat/completions", body, headers)

    def chat_json(self, content, **kw):
        s, h, p = self.chat(content, **kw)
        return s, json.loads(p)

    @staticmethod
    def parse_sse(payload: bytes):
        events, text = [], payload.decode("utf-8")
        for block in text.split("\n\n"):
            block = block.strip()
            if not block or block.startswith(":"):
                continue
            assert block.startswith("data: "), block
            d = block[6:]
            events.append(d if d == "[DONE]" else json.loads(d))
        return events

    def stream_text(self, content, **kw):
        s, h, p = self.chat(content, stream=True, **kw)
        ev = self.parse_sse(p)
        txt = "".join(e["choices"][0]["delta"].get("content", "") for e in ev
                      if e != "[DONE]" and e.get("choices"))
        return s, ev, txt
