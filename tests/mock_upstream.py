"""Strict mock of the MiniMax Agent web API.

Deliberately re-implements signature verification on its own (not importing
anything from the proxy) so a signing bug in the proxy cannot hide.
Scenario markers in the prompt text drive failure modes, which keeps them
safe to use under concurrency.
"""
import base64
import gzip
import hashlib
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REQUIRED_QUERY = ["device_platform", "biz_id", "app_id", "version_code", "uuid", "device_id", "os_name",
                  "browser_name", "device_memory", "cpu_core_num", "browser_language", "browser_platform",
                  "user_id", "screen_width", "screen_height", "unix", "lang", "token"]


def make_jwt(user_id="450123456789", device_id="987654321", exp_delta=3600, sig="sig", extra=None):
    def b64(o):
        return base64.urlsafe_b64encode(json.dumps(o).encode()).decode().rstrip("=")
    payload = {"user": {"id": user_id, "deviceID": device_id}}
    if exp_delta is not None:
        payload["exp"] = int(time.time()) + exp_delta
    payload.update(extra or {})
    return f"{b64({'alg': 'HS256', 'typ': 'JWT'})}.{b64(payload)}.{sig}"


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class Mock:
    def __init__(self, valid_jwts=(), step=0.05, chunks=4):
        self.valid = set(valid_jwts)
        self.step, self.chunks = step, chunks
        self.lock = threading.Lock()
        self.chats = {}
        self.next_id = 1000
        self.requests = []        # (uri, body_dict, jwt)
        self.counters = {}
        self.active = 0
        self.max_active = 0
        self.detail_calls = 0
        self.sign_failures = 0
        self.httpd = None

    # ---- lifecycle
    def start(self):
        mock = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                mock.handle(self, "GET")

            def do_POST(self):
                mock.handle(self, "POST")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ---- helpers
    def _reply(self, h, status, obj=None, raw=None, gz=False, ctype="application/json"):
        if raw is not None:
            body = raw
        else:
            try:
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            except UnicodeEncodeError:  # lone surrogates -> emit as \\udXXX escapes, like a real server might
                body = json.dumps(obj, ensure_ascii=True).encode("utf-8")
        if gz:
            body = gzip.compress(body)
        h.send_response(status)
        h.send_header("Content-Type", ctype)
        if gz:
            h.send_header("Content-Encoding", "gzip")
        h.send_header("Content-Length", str(len(body)))
        try:
            h.end_headers()
            h.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # proxy gave up (timeout test); not interesting

    def _count(self, key):
        with self.lock:
            self.counters[key] = self.counters.get(key, 0) + 1
            return self.counters[key]

    def verify(self, h, body_str):
        """Return None if the request is correctly signed, else an error string."""
        raw_path = h.path
        parts = urllib.parse.urlsplit(raw_path)
        q = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
        for k in REQUIRED_QUERY:
            if k not in q:
                return f"missing query param {k}"
        jwt = h.headers.get("token")
        if not jwt or q["token"][0] != jwt:
            return "token header/query mismatch"
        if q["uuid"][0] != q["user_id"][0]:
            return "uuid != user_id"
        unix = q["unix"][0]
        ts = h.headers.get("x-timestamp", "")
        if not (unix.isdigit() and ts.isdigit()):
            return "bad unix/x-timestamp"
        if abs(int(ts) - time.time()) > 120 or abs(int(unix) / 1000 - time.time()) > 120:
            return "clock skew"
        if h.headers.get("x-signature") != md5(ts + jwt + body_str):
            return "bad x-signature"
        expect_yy = md5(urllib.parse.quote(raw_path, safe="-_.!~*'()") + "_" + body_str + md5(unix) + "ooui")
        if h.headers.get("yy") != expect_yy:
            return "bad yy"
        for hdr in ("Origin", "Referer", "User-Agent"):
            if not h.headers.get(hdr):
                return f"missing header {hdr}"
        return None

    # ---- main handler
    def handle(self, h, method):
        n = int(h.headers.get("Content-Length") or 0)
        body_str = h.rfile.read(n).decode("utf-8") if n else ""
        if method == "GET":
            body_str = "{}"  # web app signs "{}" for body-less requests
        err = self.verify(h, body_str)
        if err:
            with self.lock:
                self.sign_failures += 1
            return self._reply(h, 401, {"error": err})
        jwt = h.headers.get("token")
        if jwt not in self.valid:
            return self._reply(h, 401, {"error": "unknown token"})
        path = urllib.parse.urlsplit(h.path).path
        try:
            body = json.loads(body_str) if body_str else {}
        except ValueError:
            return self._reply(h, 400, {"error": "bad json"})
        with self.lock:
            self.requests.append((path, body, jwt))
        if path == "/v1/api/user/info":
            return self._reply(h, 200, {"statusInfo": {"code": 0, "message": "ok"}, "data": {"userInfo": {"id": 1}}})
        if path == "/matrix/api/v1/chat/send_msg":
            return self.send_msg(h, body, jwt)
        if path == "/matrix/api/v1/chat/get_chat_detail":
            return self.detail(h, body)
        return self._reply(h, 404, {"error": "no such path"})

    def send_msg(self, h, body, jwt):
        for k in ("msg_type", "text", "chat_type", "attachments", "selected_mcp_tools", "backend_config"):
            if k not in body:
                return self._reply(h, 400, {"base_resp": {"status_code": 2013, "status_msg": f"missing {k}"}})
        text = body["text"]
        if "[[send500]]" in text:
            return self._reply(h, 500, {"error": "boom"})
        if "[[send503once]]" in text and self._count("503once:" + text) == 1:
            return self._reply(h, 503, {"error": "busy"})
        if "[[send429]]" in text:
            return self._reply(h, 429, {"error": "slow down"})
        if "[[sendbiz]]" in text:
            return self._reply(h, 200, {"base_resp": {"status_code": 1234, "status_msg": "boom"}})
        if "[[authbiz]]" in text:
            return self._reply(h, 200, {"base_resp": {"status_code": 1004, "status_msg": "token expired, please login"}})
        if "[[badjson]]" in text:
            return self._reply(h, 200, raw=b"<html>not json</html>", ctype="text/html")
        if "[[nochatid]]" in text:
            return self._reply(h, 200, {"base_resp": {"status_code": 0, "status_msg": "success"}})
        if "[[hang]]" in text:
            time.sleep(3)
        with self.lock:
            self.next_id += 1
            cid = self.next_id
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.chats[cid] = {"text": text, "t0": time.monotonic(), "polls": 0, "counted": False}
        self._reply(h, 200, {"chat_id": cid, "msg_id": cid * 10, "agent_pod_ip": "10.0.0.1",
                             "base_resp": {"status_code": 0, "status_msg": "success"}},
                    gz="[[gzip]]" in text)

    def _full_reply(self, text):
        if "[[unicode]]" in text:
            return "你好, мир 🌍 café — \"quotes\" \\backslash\nnewline\ttab ✓"
        if "[[surrogate]]" in text:
            return "ok \U0001F600 and broken \ud83d end"
        if "[[long]]" in text:
            return ("0123456789abcdef" * 12500)  # 200k chars
        if "[[rewrite]]" in text:
            return "FINAL REWRITTEN ANSWER"
        return "ECHO: " + text

    def detail(self, h, body):
        with self.lock:
            self.detail_calls += 1
        cid = body.get("chat_id")
        chat = self.chats.get(cid)
        if chat is None:
            return self._reply(h, 200, {"base_resp": {"status_code": 2001, "status_msg": "chat not found"}})
        text = chat["text"]
        chat["polls"] += 1
        if "[[detail500]]" in text:
            return self._reply(h, 500, {"error": "boom"})
        if "[[detailflaky]]" in text and chat["polls"] % 2 == 0:
            return self._reply(h, 503, {"error": "flaky"})
        if "[[detailauth]]" in text and chat["polls"] >= 2:
            return self._reply(h, 401, {"error": "expired mid-chat"})
        elapsed = time.monotonic() - chat["t0"]
        total = self.step * self.chunks
        full = self._full_reply(text)
        if "[[empty]]" in text:
            content = ""
        elif "[[rewrite]]" in text:
            content = "draft answer that will be replaced" if elapsed < total / 2 else full
        elif "[[pause]]" in text:
            # reveals half, stalls 1.2s, then the rest
            if elapsed < total / 2:
                content = full[: int(len(full) * elapsed / total)]
            elif elapsed < total / 2 + 1.2:
                content = full[: len(full) // 2]
            else:
                content = full
        else:
            frac = min(1.0, elapsed / total)
            content = full[: max(1, int(len(full) * frac))] if elapsed >= self.step else ""
        if elapsed >= total + 0.2 and not chat["counted"] and "[[pause]]" not in text:
            chat["counted"] = True
            with self.lock:
                self.active -= 1
        msgs = [{"msg_type": 1, "msg_content": text, "msg_id": 1}]
        if "[[multi]]" in text:
            msgs.append({"msg_type": 2, "msg_content": "first progress note", "msg_id": 2})
            msgs.append({"msg_type": 2, "msg_content": content, "msg_id": 3})
        elif content:
            msgs.append({"msg_type": 2, "msg_content": content, "msg_id": 2})
        self._reply(h, 200, {"messages": msgs, "base_resp": {"status_code": 0, "status_msg": "success"}})
