#!/usr/bin/env python3
"""minimax-web2api: turn your logged-in MiniMax Agent web session into an
OpenAI-compatible API (/v1/chat/completions, /v1/models).

Single file, Python 3.8+, standard library only. (Playwright is optional and
only used by `--login`.)

Flow:   login once  ->  token saved  ->  `python minimax_web2api.py`  ->  API

This talks to the same private endpoints the MiniMax Agent web app uses, as
reverse-engineered by the community (see README). Personal use only; it can
break whenever MiniMax changes their web app. For anything serious use the
official API: https://platform.minimax.io
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import hmac
import json
import logging
import os
import random
import re
import secrets
import select
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

__version__ = "1.0.0"
log = logging.getLogger("minimax-web2api")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# NOTE: only agent.minimaxi.com ("cn") is confirmed by the community projects.
# "global" assumes the same API paths on agent.minimax.io (unverified).
REGIONS = {
    "global": "https://agent.minimax.io",
    "cn": "https://agent.minimaxi.com",
}
APP_DIR = Path(os.environ.get("MINIMAX_W2A_HOME", str(Path.home() / ".minimax-web2api")))
SEND_URI = "/matrix/api/v1/chat/send_msg"
DETAIL_URI = "/matrix/api/v1/chat/get_chat_detail"
USERINFO_URI = "/v1/api/user/info"
AI_MSG_TYPE = 2  # msg_type of assistant messages in get_chat_detail

DEFAULTS: Dict[str, Any] = {
    "host": "127.0.0.1",          # loopback by default: this proxy holds YOUR account
    "port": 8082,
    "region": "global",
    "base_url": None,             # overrides region
    "tokens_file": str(APP_DIR / "tokens.json"),
    "tokens": [],                 # inline tokens (prefer tokens_file)
    "api_keys": [],               # empty = no auth (loopback only!)
    "allowed_hosts": [],          # extra Host header values accepted when api_keys is empty
    "cors_origins": [],           # e.g. ["http://localhost:3000"]; empty = no CORS
    "proxy": None,                # http://127.0.0.1:7890
    "model_name": "minimax-agent",
    "poll_interval_sec": 1.0,
    "settle_sec": 4.0,            # no content change for this long => answer finished
    "first_content_timeout_sec": 120.0,
    "request_timeout_sec": 300.0,
    "upstream_timeout_sec": 20.0,
    "retry_attempts": 2,
    "retry_delay_sec": 1.0,
    "poll_error_limit": 5,
    "message_pick": "last",       # which AI message of the chat to expose: last|first
    "stream_mode": "live",        # live: forward text as it grows | buffered: send only the settled final text
    "max_concurrent": 4,
    "queue_wait_sec": 5.0,
    "max_body_bytes": 4 * 1024 * 1024,
    "max_prompt_chars": 200_000,
    "keepalive_sec": 10.0,
    "token_cooldown_sec": 300.0,
    "log_requests": True,
}

FIXED_QUERY = [
    ("device_platform", "web"), ("biz_id", "3"), ("app_id", "3001"),
    ("version_code", "22201"), ("uuid", None), ("device_id", None),
    ("os_name", "Mac"), ("browser_name", "chrome"), ("device_memory", "8"),
    ("cpu_core_num", "11"), ("browser_language", "zh-CN"),
    ("browser_platform", "MacIntel"), ("user_id", None),
    ("screen_width", "1920"), ("screen_height", "1080"), ("unix", None),
    ("lang", "zh"), ("token", None),
]
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36")

# JS run in the logged-in MiniMax Agent tab (DevTools console or Playwright).
# Returns {token} or {error}. Kept here so --print-snippet and --login share it.
EXTRACT_JS = r"""(function () {
  function find(o, k, d) {
    if (d > 6 || o === null || typeof o !== 'object') return null;
    if (Object.prototype.hasOwnProperty.call(o, k) &&
        (typeof o[k] === 'string' || typeof o[k] === 'number')) return String(o[k]);
    var vals = Object.keys(o).map(function (x) { return o[x]; });
    for (var i = 0; i < vals.length; i++) { var r = find(vals[i], k, d + 1); if (r) return r; }
    return null;
  }
  var tok = localStorage.getItem('_token');
  if (!tok) return { error: 'no _token in localStorage - are you logged in?' };
  tok = tok.trim().replace(/^"+|"+$/g, '');
  var uid = null, keys = ['user_detail_agent', 'user_detail'];
  for (var i = 0; i < keys.length && !uid; i++) {
    try { var raw = localStorage.getItem(keys[i]); if (raw) uid = find(JSON.parse(raw), 'realUserID', 0); }
    catch (e) {}
  }
  return { token: uid ? uid + '+' + tok : tok, hasUserId: !!uid };
})()"""


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class ProxyError(Exception):
    """Error that maps to an OpenAI-style HTTP error response."""

    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 code: Optional[str] = None, param: Optional[str] = None):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type_, code, param


class UpstreamError(Exception):
    """Failure talking to MiniMax. kind: auth|rate|timeout|network|server|protocol"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class NoTokenError(Exception):
    pass


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------
def _b64url_json(part: str) -> Optional[dict]:
    try:
        raw = base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))
        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def normalize_token(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'").strip()
    if s.lower().startswith("bearer "):
        s = s[7:].strip()
    return s


class TokenInfo:
    """A parsed MiniMax session token: `realUserID+JWT` or a bare JWT."""

    def __init__(self, raw: str, name: str = ""):
        raw = normalize_token(raw)
        if not raw or any(c.isspace() for c in raw):
            raise ValueError("token is empty or contains whitespace")
        prefix, jwt = (raw.split("+", 1) if "+" in raw else ("", raw))
        parts = jwt.split(".")
        if len(parts) != 3 or not all(parts):
            raise ValueError("token must be a JWT (xxx.yyy.zzz), optionally prefixed with 'realUserID+'")
        if prefix and not (prefix.isascii() and prefix.isdigit()):
            raise ValueError("the part before '+' must be the numeric realUserID")
        payload = _b64url_json(parts[1]) or {}
        user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
        user_id = prefix or (str(user.get("id")) if user.get("id") else "")
        if not user_id:
            raise ValueError("cannot determine user id: use the 'realUserID+token' form")
        if not _SAFE_ID.match(user_id):  # these go into a signed URL query: no '&', '#', spaces, ...
            raise ValueError("user id contains unexpected characters")
        device = str(user.get("deviceID") or "")
        self.jwt = jwt
        self.user_id = user_id
        self.device_id = device if _SAFE_ID.match(device) else str(random.randint(10_000_000, 99_999_999))
        exp = payload.get("exp")
        self.exp = float(exp) if isinstance(exp, (int, float)) and not isinstance(exp, bool) else None
        self.raw = raw
        self.name = name or f"user{user_id[-4:]}"

    def expired(self, now: Optional[float] = None) -> bool:
        return self.exp is not None and self.exp < (now if now is not None else time.time())

    def masked(self) -> str:
        fp = hashlib.sha256(self.jwt.encode()).hexdigest()[:8]  # fingerprint only; no token characters
        return f"{self.name}[uid {self.user_id[:3]}…{self.user_id[-3:]}|fp {fp}]"

    def __repr__(self) -> str:  # never leak the secret via repr/logging
        return f"<TokenInfo {self.masked()}>"


def read_tokens_file(path: Path) -> List[str]:
    """Accepts JSON ({"tokens": [...]} or [...]; items str or {"token": ..}) or one token per line."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        data = json.loads(text)
        items = data.get("tokens", []) if isinstance(data, dict) else data
        out = []
        for it in items:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict) and isinstance(it.get("token"), str):
                out.append(it["token"])
        return out
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def write_tokens_file(path: Path, tokens: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"tokens": tokens}, f, indent=2)
    os.replace(str(tmp), str(path))
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass


class TokenPool:
    """Round-robin pool with cooldown for rejected tokens and hot-reload of the file."""

    def __init__(self, inline: List[str], tokens_file: Optional[str], cooldown: float = 300.0):
        self._inline_raw = list(inline)
        self._file = Path(tokens_file) if tokens_file else None
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._tokens: List[TokenInfo] = []
        self._bad_until: Dict[str, float] = {}
        self._rr = 0
        self._mtime: Optional[float] = None
        self._load(first=True)

    def _load(self, first: bool = False) -> None:
        raws = list(self._inline_raw)
        env = os.environ.get("MINIMAX_TOKEN", "")
        raws += [t for t in env.split(",") if t.strip()]
        mtime = None
        if self._file and self._file.exists():
            try:
                mtime = self._file.stat().st_mtime
                raws += read_tokens_file(self._file)
            except Exception as e:
                log.warning("cannot read tokens file %s: %s", self._file, e)
        parsed, seen = [], set()
        for i, raw in enumerate(raws):
            try:
                t = TokenInfo(raw, name=f"t{i + 1}")
            except ValueError as e:
                log.warning("ignoring token #%d: %s", i + 1, e)
                continue
            if t.jwt in seen:
                continue
            seen.add(t.jwt)
            if t.expired():
                log.warning("token %s is expired - run --login again", t.masked())
                continue
            parsed.append(t)
        with self._lock:
            self._tokens, self._mtime = parsed, mtime
        if not first:
            log.info("tokens reloaded: %d usable", len(parsed))

    def _maybe_reload(self) -> None:
        if not self._file:
            return
        try:
            m = self._file.stat().st_mtime if self._file.exists() else None
        except OSError:
            return
        if m != self._mtime:
            self._load()

    def __len__(self) -> int:
        with self._lock:
            return len(self._tokens)

    def acquire(self, exclude: Optional[set] = None) -> TokenInfo:
        self._maybe_reload()
        exclude = exclude or set()
        now = time.monotonic()
        with self._lock:
            cands = [t for t in self._tokens if t.jwt not in exclude]
            if not cands:
                raise NoTokenError("no usable MiniMax token. Run: python minimax_web2api.py --login")
            healthy = [t for t in cands if self._bad_until.get(t.jwt, 0) <= now]
            pool = healthy or sorted(cands, key=lambda t: self._bad_until.get(t.jwt, 0))[:1]
            t = pool[self._rr % len(pool)]
            self._rr += 1
            return t

    def mark_bad(self, tok: TokenInfo) -> None:
        with self._lock:
            self._bad_until[tok.jwt] = time.monotonic() + self._cooldown
        log.warning("token %s rejected by MiniMax; cooling down %.0fs", tok.masked(), self._cooldown)

    def mark_good(self, tok: TokenInfo) -> None:
        with self._lock:
            self._bad_until.pop(tok.jwt, None)


# --------------------------------------------------------------------------
# Request signing (mirrors the web app / community reverse-engineering)
# --------------------------------------------------------------------------
def md5hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def js_encode_uri_component(s: str) -> str:
    return urllib.parse.quote(s, safe="-_.!~*'()")


def compact_json(obj: Any) -> str:
    """Same bytes as JS JSON.stringify for our payloads."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def sign_request(uri: str, payload: Any, tok: TokenInfo, base_url: str,
                 now: Optional[float] = None):
    """Return (path_with_query, body_json_str, headers)."""
    ts = int(now if now is not None else time.time())
    unix = str(ts * 1000)
    values = {"uuid": tok.user_id, "device_id": tok.device_id, "user_id": tok.user_id,
              "unix": unix, "token": tok.jwt}
    query = "&".join(
        f"{k}={values[k] if v is None else v}" for k, v in FIXED_QUERY
        if not (v is None and values.get(k) is None))
    path = f"{uri}?{query}"
    data_json = compact_json(payload if payload is not None else {})
    yy = md5hex(js_encode_uri_component(path) + "_" + data_json + md5hex(unix) + "ooui")
    signature = md5hex(f"{ts}{tok.jwt}{data_json}")
    p = urllib.parse.urlsplit(base_url)
    origin = f"{p.scheme}://{p.netloc}"
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cache-Control": "no-cache",
        "Origin": origin,
        "Referer": origin + "/",
        "User-Agent": USER_AGENT,
        "token": tok.jwt,
        "x-timestamp": str(ts),
        "x-signature": signature,
        "yy": yy,
    }
    return path, data_json, headers


# --------------------------------------------------------------------------
# Upstream client
# --------------------------------------------------------------------------
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_AUTH_HINTS = ("token", "login", "unauthor", "auth", "登录", "未登录", "expired", "invalid user")


class Upstream:
    def __init__(self, base_url: str, proxy: Optional[str] = None, timeout: float = 20.0,
                 retries: int = 2, retry_delay: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout, self.retries, self.retry_delay = timeout, retries, retry_delay
        handlers: List[Any] = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        self._opener = urllib.request.build_opener(*handlers)

    def _once(self, method: str, uri: str, payload: Any, tok: TokenInfo) -> dict:
        path, data_json, headers = sign_request(uri, payload, tok, self.base_url)
        body = None
        if method == "POST":
            body = data_json.encode("utf-8", "replace")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw = resp.read(32 * 1024 * 1024)
                enc = (resp.headers.get("Content-Encoding") or "").lower()
        except urllib.error.HTTPError as e:
            try:
                e.read(4096)
            except Exception:
                pass
            if e.code in (401, 403):
                raise UpstreamError("auth", f"HTTP {e.code} from MiniMax")
            if e.code == 429:
                raise UpstreamError("rate", "HTTP 429 from MiniMax")
            if e.code >= 500:
                raise UpstreamError("server", f"HTTP {e.code} from MiniMax")
            raise UpstreamError("protocol", f"HTTP {e.code} from MiniMax")
        except (socket.timeout, TimeoutError):
            raise UpstreamError("timeout", "MiniMax request timed out")
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise UpstreamError("timeout", "MiniMax request timed out")
            raise UpstreamError("network", f"cannot reach MiniMax: {e.reason}")
        except (ConnectionError, OSError) as e:
            raise UpstreamError("network", f"connection error: {e}")
        try:
            if enc == "gzip":
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            raise UpstreamError("protocol", "MiniMax returned a non-JSON response")
        if not isinstance(data, dict):
            raise UpstreamError("protocol", "MiniMax returned an unexpected JSON shape")
        br = data.get("base_resp")
        si = data.get("statusInfo")
        code, msg = None, ""
        if isinstance(br, dict):
            code, msg = br.get("status_code"), str(br.get("status_msg") or "")
        elif isinstance(si, dict):
            code, msg = si.get("code"), str(si.get("message") or "")
        if code not in (None, 0):
            kind = "auth" if any(h in msg.lower() for h in _AUTH_HINTS) else "server"
            raise UpstreamError(kind, f"MiniMax error {code}: {msg or 'unknown'}")
        return data

    def call(self, uri: str, payload: Any, tok: TokenInfo, method: str = "POST",
             retry: bool = False) -> dict:
        """retry=True only for idempotent-safe cases (never-sent / explicit 429,5xx)."""
        attempt = 0
        while True:
            try:
                return self._once(method, uri, payload, tok)
            except UpstreamError as e:
                retryable = e.kind in ("network", "rate") or (e.kind == "server" and "HTTP 5" in str(e))
                if not retry or not retryable or attempt >= self.retries:
                    raise
                attempt += 1
                time.sleep(self.retry_delay * attempt)


# --------------------------------------------------------------------------
# OpenAI messages -> one MiniMax prompt
# --------------------------------------------------------------------------
def _clean(s: str) -> str:
    # lone surrogates (from "\ud800" JSON escapes) would crash UTF-8 encoding
    return s.encode("utf-8", "replace").decode("utf-8").replace("\x00", "")


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text") \
                    and isinstance(part.get("text"), str):
                out.append(part["text"])
            elif isinstance(part, dict):
                raise ProxyError(400, f"unsupported content part type {part.get('type')!r}: "
                                 "only text is supported (no image/file input yet)",
                                 code="unsupported_content")
            else:
                raise ProxyError(400, "invalid content part", code="invalid_content")
        return "\n".join(out)
    raise ProxyError(400, "message content must be a string or an array of parts", code="invalid_content")


_ROLE_LABEL = {"system": "system", "developer": "system", "user": "user",
               "assistant": "assistant", "tool": "tool", "function": "tool"}


def build_prompt(messages: Any, max_chars: int) -> str:
    if not isinstance(messages, list) or not messages:
        raise ProxyError(400, "'messages' must be a non-empty array", param="messages")
    items = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ProxyError(400, f"messages[{i}] must be an object", param="messages")
        role = m.get("role")
        if not isinstance(role, str) or not role:
            raise ProxyError(400, f"messages[{i}].role is required", param="messages")
        text = _clean(content_to_text(m.get("content")))
        items.append((_ROLE_LABEL.get(role, role[:32]), text))
    if not any(t.strip() for _, t in items):
        raise ProxyError(400, "all messages are empty", param="messages")
    if len(items) == 1 and items[0][0] == "user":
        prompt = items[0][1]
    else:
        lines = ["The following is a conversation transcript. Reply only with the next message "
                 "from \"assistant\". Do not repeat the transcript and do not prefix your answer "
                 "with a role label.", ""]
        for role, text in items:
            lines.append(f"{role}: {text}")
        lines.append("assistant:")
        prompt = "\n".join(lines)
    if len(prompt) > max_chars:
        raise ProxyError(400, f"prompt too long ({len(prompt)} > {max_chars} chars)",
                         code="context_length_exceeded", param="messages")
    return prompt


def est_tokens(s: str) -> int:
    return max(1, len(s) // 4) if s else 0


# --------------------------------------------------------------------------
# One chat run: send_msg, then poll get_chat_detail until the text settles
# --------------------------------------------------------------------------
class ChatRun:
    def __init__(self, up: Upstream, tok: TokenInfo, prompt: str, cfg: dict, cancel: threading.Event,
                 alive: Optional[Callable[[], bool]] = None):
        self.up, self.tok, self.prompt, self.cfg, self.cancel = up, tok, prompt, cfg, cancel
        self.alive = alive  # returns False once the HTTP client has gone away
        self.chat_id: Any = None
        self.text = ""
        self.finish_reason = "stop"

    def start(self) -> None:
        payload = {"msg_type": 1, "text": self.prompt, "chat_type": 1, "attachments": [],
                   "selected_mcp_tools": [], "backend_config": {}, "sub_agent_ids": []}
        data = self.up.call(SEND_URI, payload, self.tok, retry=True)
        self.chat_id = data.get("chat_id")
        if self.chat_id in (None, ""):
            raise UpstreamError("protocol", "send_msg returned no chat_id")

    def _pick(self, detail: dict) -> str:
        msgs = detail.get("messages")
        if not isinstance(msgs, list):
            return ""
        ai = [m for m in msgs if isinstance(m, dict) and m.get("msg_type") == AI_MSG_TYPE]
        ai = [m for m in ai if m.get("msg_content")]
        if not ai:
            return ""
        m = ai[0] if self.cfg["message_pick"] == "first" else ai[-1]
        c = m.get("msg_content")
        return _clean(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))

    def snapshots(self) -> Iterator[str]:
        """Yield the full answer-so-far after every poll; return when settled."""
        cfg = self.cfg
        t0 = last_change = time.monotonic()
        errors = 0
        while True:
            if self.cancel.wait(cfg["poll_interval_sec"]):
                return
            if self.alive is not None and not self.alive():
                log.info("client disconnected; abandoning chat %s", self.chat_id)
                self.cancel.set()
                return
            now = time.monotonic()
            try:
                detail = self.up.call(DETAIL_URI, {"chat_id": self.chat_id}, self.tok)
                errors = 0
            except UpstreamError as e:
                if e.kind == "auth":
                    raise
                errors += 1
                log.warning("poll error %d/%d: %s", errors, cfg["poll_error_limit"], e)
                if errors >= cfg["poll_error_limit"]:
                    raise
                detail = None
            if detail is not None:
                content = self._pick(detail)
                if content != self.text:
                    self.text, last_change = content, now
                yield self.text
            if self.text and now - last_change >= cfg["settle_sec"]:
                return
            if not self.text and now - t0 > cfg["first_content_timeout_sec"]:
                raise UpstreamError("timeout", "MiniMax produced no answer in time")
            if now - t0 > cfg["request_timeout_sec"]:
                if not self.text:
                    raise UpstreamError("timeout", "MiniMax request timed out")
                self.finish_reason = "length"
                log.warning("request_timeout_sec reached; returning partial answer")
                return


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------
class App:
    def __init__(self, cfg: dict, pool: Optional[TokenPool] = None, upstream: Optional[Upstream] = None):
        self.cfg = cfg
        base = (cfg.get("base_url") or REGIONS[cfg["region"]]).rstrip("/")
        self.base_url = base
        self.pool = pool or TokenPool(cfg["tokens"], cfg["tokens_file"], cfg["token_cooldown_sec"])
        self.upstream = upstream or Upstream(base, cfg["proxy"], cfg["upstream_timeout_sec"],
                                             cfg["retry_attempts"], cfg["retry_delay_sec"])
        self.sem = threading.BoundedSemaphore(cfg["max_concurrent"])
        self.api_keys = [k.encode() for k in cfg["api_keys"] if isinstance(k, str) and k]
        self.hosts_ok = {"localhost", "127.0.0.1", "::1", "[::1]"} | {h.lower() for h in cfg["allowed_hosts"]}

    def check_key(self, supplied: str) -> bool:
        s = supplied.encode()
        ok = False
        for k in self.api_keys:  # no early exit
            ok |= hmac.compare_digest(s, k)
        return ok


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"minimax-web2api/{__version__}"
    timeout = 30  # slow-client protection for reads

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # route through logging; neutralise control chars (log injection)
        if self.app.cfg["log_requests"]:
            msg = _CTRL.sub(lambda m_: "\\x%02x" % ord(m_.group()), fmt % args)
            log.info("%s - %s", self.address_string(), msg)

    # -- helpers -----------------------------------------------------------
    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if origin and origin in self.app.cfg["cors_origins"]:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "authorization, content-type, x-api-key")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _client_alive(self) -> bool:
        """False once the peer closed its end (EOF). Pipelined bytes count as alive."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return True
            return self.connection.recv(1, socket.MSG_PEEK) != b""
        except (OSError, ValueError):
            return False

    def _safe_send_error(self, e: ProxyError) -> None:
        try:
            self._send_error_obj(e)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, socket.timeout):
            self.close_connection = True  # client already gone; nothing to tell it

    def _close_if_body_unread(self) -> None:
        """If we answer without consuming the request body, those bytes would be parsed as the
        next request on a keep-alive connection (desync/smuggling). Close instead."""
        if getattr(self, "_unread_body", False):
            self.send_header("Connection", "close")
            self.close_connection = True

    def _send_json(self, status: int, obj: Any, extra: Optional[dict] = None) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8", "replace")
        self.send_response(status)
        self._close_if_body_unread()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_obj(self, e: ProxyError) -> None:
        self._send_json(e.status, {"error": {"message": e.message, "type": e.type,
                                             "param": e.param, "code": e.code}})

    def _precheck(self) -> None:
        app = self.app
        if app.api_keys:
            auth = self.headers.get("Authorization", "")
            key = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if not key or not app.check_key(key):
                raise ProxyError(401, "invalid or missing API key", "authentication_error", "invalid_api_key")
        else:
            host = (self.headers.get("Host") or "").lower()
            hostname = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            if hostname not in app.hosts_ok:
                raise ProxyError(403, "Host not allowed while no api_keys are configured "
                                 "(add it to allowed_hosts or set api_keys)", "permission_error", "host_not_allowed")

    def _read_json(self) -> dict:
        ct = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ct != "application/json":
            raise ProxyError(415, "Content-Type must be application/json", code="unsupported_media_type")
        if self.headers.get("Transfer-Encoding"):
            raise ProxyError(411, "chunked request bodies are not supported; send Content-Length",
                             code="length_required")
        cl = self.headers.get("Content-Length")
        if cl is None:
            raise ProxyError(411, "Content-Length required", code="length_required")
        if not (cl.isascii() and cl.isdigit()) or len(cl) > 15:
            raise ProxyError(400, "invalid Content-Length")
        n = int(cl)
        if n > self.app.cfg["max_body_bytes"]:
            raise ProxyError(413, "request body too large", code="request_too_large")
        try:
            body = self.rfile.read(n)
        except (socket.timeout, TimeoutError):
            self.close_connection = True
            raise ProxyError(408, "timed out reading request body", code="request_timeout")
        self._unread_body = False
        if len(body) < n:
            self.close_connection = True
            raise ProxyError(400, "incomplete request body")
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise ProxyError(400, "request body is not valid JSON", code="invalid_json")
        if not isinstance(data, dict):
            raise ProxyError(400, "request body must be a JSON object", code="invalid_json")
        return data

    # -- routing -----------------------------------------------------------
    def _dispatch(self, method: str) -> None:
        self._sse_started = False
        cl = (self.headers.get("Content-Length") or "").strip()
        self._unread_body = ("Transfer-Encoding" in self.headers) or (cl not in ("", "0"))
        try:
            path = urllib.parse.urlsplit(self.path).path
            path = path.rstrip("/") or "/"
            if method == "OPTIONS":
                self.send_response(204)
                self._close_if_body_unread()
                self.send_header("Content-Length", "0")
                self._cors()
                self.end_headers()
                return
            if path in ("/", "/health") and method == "GET":
                return self._send_json(200, {"status": "ok", "service": "minimax-web2api",
                                             "version": __version__, "tokens": len(self.app.pool)})
            self._precheck()
            if path == "/v1/models" and method == "GET":
                m = self.app.cfg["model_name"]
                return self._send_json(200, {"object": "list", "data": [
                    {"id": m, "object": "model", "created": 0, "owned_by": "minimax"}]})
            if path.startswith("/v1/models/") and method == "GET":
                mid = urllib.parse.unquote(path[len("/v1/models/"):])
                return self._send_json(200, {"id": mid, "object": "model", "created": 0, "owned_by": "minimax"})
            if path == "/v1/chat/completions":
                if method != "POST":
                    raise ProxyError(405, "use POST", code="method_not_allowed")
                return self._chat(self._read_json())
            raise ProxyError(404, f"unknown route {method} {path}", code="not_found")
        except ProxyError as e:
            if self._sse_started:
                return
            self._safe_send_error(e)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        except Exception:
            log.exception("unhandled error")
            self.close_connection = True
            if not self._sse_started:
                try:
                    self._send_error_obj(ProxyError(500, "internal server error", "server_error"))
                except Exception:
                    pass

    do_GET = lambda self: self._dispatch("GET")        # noqa: E731
    do_POST = lambda self: self._dispatch("POST")      # noqa: E731
    do_OPTIONS = lambda self: self._dispatch("OPTIONS")  # noqa: E731
    do_PUT = do_DELETE = do_PATCH = lambda self: self._dispatch("OTHER")  # noqa: E731

    # -- chat --------------------------------------------------------------
    @staticmethod
    def _map_upstream(e: UpstreamError) -> ProxyError:
        m = {
            "auth": (502, "upstream_auth_failed",
                     "MiniMax rejected the session token. Re-login: python minimax_web2api.py --login"),
            "rate": (429, "upstream_rate_limited", "MiniMax rate-limited the request; retry later"),
            "timeout": (504, "upstream_timeout", str(e)),
            "network": (502, "upstream_unreachable", str(e)),
            "server": (502, "upstream_error", str(e)),
            "protocol": (502, "upstream_protocol_error", str(e)),
        }
        status, code, msg = m.get(e.kind, (502, "upstream_error", str(e)))
        return ProxyError(status, msg, "api_error", code)

    def _start_run(self, prompt: str, cancel: threading.Event) -> ChatRun:
        app, tried = self.app, set()
        last: Optional[UpstreamError] = None
        for _ in range(max(1, len(app.pool))):
            try:
                tok = app.pool.acquire(exclude=tried)
            except NoTokenError as e:
                if last is not None:
                    break
                raise ProxyError(503, str(e), "api_error", "no_token")
            tried.add(tok.jwt)
            run = ChatRun(app.upstream, tok, prompt, app.cfg, cancel, self._client_alive)
            try:
                run.start()
                app.pool.mark_good(tok)
                return run
            except UpstreamError as e:
                last = e
                if e.kind == "auth":
                    app.pool.mark_bad(tok)
                    continue
                raise self._map_upstream(e)
        raise self._map_upstream(last or UpstreamError("auth", "no token accepted"))

    def _chat(self, req: dict) -> None:
        app = self.app
        stream = req.get("stream", False)
        if stream is None:
            stream = False
        if not isinstance(stream, bool):
            raise ProxyError(400, "'stream' must be a boolean", param="stream")
        model = req.get("model")
        model = _clean(model) if isinstance(model, str) and model and len(model) <= 200 else app.cfg["model_name"]
        so = req.get("stream_options")
        include_usage = bool(isinstance(so, dict) and so.get("include_usage"))
        if req.get("tools"):
            log.warning("request has 'tools': tool calling is not supported and is ignored")
        prompt = build_prompt(req.get("messages"), app.cfg["max_prompt_chars"])

        if not app.sem.acquire(timeout=app.cfg["queue_wait_sec"]):
            raise ProxyError(429, "too many concurrent requests", "rate_limit_error", "too_many_requests")
        cancel = threading.Event()
        try:
            run = self._start_run(prompt, cancel)
            cid = "chatcmpl-" + secrets.token_hex(12)
            created = int(time.time())
            if stream:
                self._stream(run, cid, created, model, prompt, include_usage, cancel)
            else:
                try:
                    for _ in run.snapshots():
                        pass
                except UpstreamError as e:
                    if e.kind == "auth":
                        app.pool.mark_bad(run.tok)
                    raise self._map_upstream(e)
                if cancel.is_set():  # client left while we were waiting
                    self.close_connection = True
                    return
                if not run.text:
                    raise ProxyError(502, "MiniMax returned an empty answer", "api_error", "empty_response")
                usage = {"prompt_tokens": est_tokens(prompt), "completion_tokens": est_tokens(run.text),
                         "total_tokens": est_tokens(prompt) + est_tokens(run.text)}
                self._send_json(200, {
                    "id": cid, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": run.text},
                                 "finish_reason": run.finish_reason}],
                    "usage": usage})
        finally:
            cancel.set()
            app.sem.release()

    def _sse(self, payload: Any) -> None:
        s = payload if isinstance(payload, str) else "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        self.wfile.write(s.encode("utf-8", "replace"))
        self.wfile.flush()

    def _stream(self, run: ChatRun, cid: str, created: int, model: str, prompt: str,
                include_usage: bool, cancel: threading.Event) -> None:
        def chunk(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        self.close_connection = True
        self._sse_started = True
        emitted, last_write = "", time.monotonic()
        try:
            self._sse(chunk({"role": "assistant", "content": ""}))
            try:
                live = self.app.cfg["stream_mode"] == "live"
                for snap in run.snapshots():
                    if not live:
                        if time.monotonic() - last_write >= self.app.cfg["keepalive_sec"]:
                            self._sse(": keepalive\n\n")
                            last_write = time.monotonic()
                        continue
                    if len(snap) > len(emitted) and snap.startswith(emitted):
                        self._sse(chunk({"content": snap[len(emitted):]}))
                        emitted, last_write = snap, time.monotonic()
                    elif time.monotonic() - last_write >= self.app.cfg["keepalive_sec"]:
                        self._sse(": keepalive\n\n")
                        last_write = time.monotonic()
                final = run.text
                if len(final) > len(emitted):  # diverged mid-way; best-effort tail
                    if not final.startswith(emitted):
                        log.warning("answer text changed non-monotonically; tail is best-effort")
                    tail = final[len(emitted):]
                    for i in range(0, len(tail), 256):
                        self._sse(chunk({"content": tail[i:i + 256]}))
                    emitted = final
                if not emitted:
                    raise UpstreamError("protocol", "MiniMax returned an empty answer")
            except UpstreamError as e:
                if e.kind == "auth":
                    self.app.pool.mark_bad(run.tok)
                pe = self._map_upstream(e)
                self._sse({"error": {"message": pe.message, "type": pe.type, "code": pe.code}})
                self._sse("data: [DONE]\n\n")
                return
            self._sse(chunk({}, run.finish_reason))
            if include_usage:
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                           "choices": [], "usage": {
                               "prompt_tokens": est_tokens(prompt), "completion_tokens": est_tokens(emitted),
                               "total_tokens": est_tokens(prompt) + est_tokens(emitted)}})
            self._sse("data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, socket.timeout):
            cancel.set()  # client went away: stop polling MiniMax
            log.info("client disconnected mid-stream")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, addr, app: App):
        self.app = app
        if ":" in str(addr[0]):
            self.address_family = socket.AF_INET6
        super().__init__(addr, Handler)


# --------------------------------------------------------------------------
# Config / CLI
# --------------------------------------------------------------------------
def validate_config(cfg: dict) -> dict:
    def num(k, lo, hi=None):
        v = cfg[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < lo or (hi is not None and v > hi):
            raise ValueError(f"config '{k}' must be a number >= {lo}")
    for k, lo in (("poll_interval_sec", 0.01), ("settle_sec", 0.05), ("first_content_timeout_sec", 1),
                  ("request_timeout_sec", 1), ("upstream_timeout_sec", 1), ("retry_delay_sec", 0),
                  ("queue_wait_sec", 0), ("keepalive_sec", 0.1), ("token_cooldown_sec", 0)):
        num(k, lo)
    for k, lo in (("port", 0), ("retry_attempts", 0), ("poll_error_limit", 1), ("max_concurrent", 1),
                  ("max_body_bytes", 1024), ("max_prompt_chars", 100)):
        num(k, lo)
    if not 0 <= int(cfg["port"]) <= 65535:
        raise ValueError("config 'port' must be 0-65535")
    if cfg["region"] not in REGIONS:
        raise ValueError(f"config 'region' must be one of {sorted(REGIONS)}")
    if cfg["stream_mode"] not in ("live", "buffered"):
        raise ValueError("config 'stream_mode' must be 'live' or 'buffered'")
    if cfg["message_pick"] not in ("first", "last"):
        raise ValueError("config 'message_pick' must be 'first' or 'last'")
    for k in ("api_keys", "tokens", "allowed_hosts", "cors_origins"):
        if not isinstance(cfg[k], list) or not all(isinstance(x, str) for x in cfg[k]):
            raise ValueError(f"config '{k}' must be a list of strings")
    if cfg["base_url"] is not None:
        u = urllib.parse.urlsplit(str(cfg["base_url"]))
        if u.scheme not in ("http", "https") or not u.netloc:
            raise ValueError("config 'base_url' must be an http(s) URL")
    return cfg


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    cfg = dict(DEFAULTS)
    p = Path(path) if path else Path("config.json")
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            raise ValueError(f"{p}: invalid JSON: {e}")
        if not isinstance(data, dict):
            raise ValueError(f"{p}: must contain a JSON object")
        unknown = set(data) - set(DEFAULTS)
        if unknown:
            log.warning("unknown config keys ignored: %s", ", ".join(sorted(unknown)))
        cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
    elif path:
        raise ValueError(f"config file not found: {path}")
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    if not cfg["proxy"]:
        cfg["proxy"] = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None
    return validate_config(cfg)


def create_server(cfg: dict, app: Optional[App] = None) -> Server:
    app = app or App(cfg)
    return Server((cfg["host"], int(cfg["port"])), app)


def login_interactive(region: str, base_url: Optional[str], out: Path, timeout: float = 600,
                      headless: bool = False, bare_token_grace: float = 20.0,
                      profile_dir: Optional[Path] = None) -> str:
    """Open a browser on the MiniMax Agent site, wait for you to log in, save the token."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        raise SystemExit("--login needs Playwright:\n  pip install playwright && playwright install chromium\n"
                         "or use the no-install route: python minimax_web2api.py --print-snippet")
    url = (base_url or REGIONS[region]).rstrip("/") + "/"
    profile = Path(profile_dir) if profile_dir else APP_DIR / "browser-profile"
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)  # holds your logged-in session
    print(f"Opening {url}\nLog in with your MiniMax account in the browser window...")
    token = None
    try:
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(str(profile), headless=headless)
            try:
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto(url, timeout=60_000)
                start = time.time()
                while time.time() - start < timeout:
                    try:
                        r = page.evaluate(EXTRACT_JS)
                    except Exception:
                        r = None  # page navigating (login redirects) - try again
                    if isinstance(r, dict) and r.get("token"):
                        try:
                            TokenInfo(r["token"])
                        except ValueError:
                            r = None
                        else:
                            token = r["token"]
                            if r.get("hasUserId") or time.time() - start >= bare_token_grace:
                                break
                    time.sleep(0.5 if headless else 2)
            finally:
                ctx.close()
    except SystemExit:
        raise
    except Exception as e:
        first = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
        raise SystemExit(f"could not drive the browser: {first}\n"
                         "(if Chromium is missing: playwright install chromium; on a server without a display "
                         "use the no-install route: python minimax_web2api.py --print-snippet)")
    if not token:
        raise SystemExit("login timed out; no token found")
    add_token(out, token)
    return token


def add_token(path: Path, raw: str) -> TokenInfo:
    info = TokenInfo(raw)  # validates
    if info.expired():
        raise ValueError("that token is already expired")
    existing = read_tokens_file(path) if path.exists() else []
    kept = []
    for t in existing:
        try:
            if TokenInfo(t).jwt != info.jwt:
                kept.append(t)
        except ValueError:
            continue
    kept.append(normalize_token(raw))
    write_tokens_file(path, kept)
    return info


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MiniMax Agent web -> OpenAI-compatible API")
    ap.add_argument("--config")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--region", choices=sorted(REGIONS), help="global=agent.minimax.io, cn=agent.minimaxi.com")
    ap.add_argument("--base-url")
    ap.add_argument("--proxy")
    ap.add_argument("--tokens-file")
    ap.add_argument("--token", action="append", help="session token (repeatable); prefer --add-token")
    ap.add_argument("--api-key", action="append", help="require this API key (repeatable)")
    ap.add_argument("--login", action="store_true", help="open a browser, log in, save the token (needs Playwright)")
    ap.add_argument("--add-token", metavar="TOKEN", help="validate and save a token to the tokens file")
    ap.add_argument("--print-snippet", action="store_true", help="print the DevTools console snippet that extracts your token")
    ap.add_argument("--check", action="store_true", help="check saved tokens against MiniMax and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if a.print_snippet:
        print("1) Open the MiniMax Agent site and log in.\n2) Open DevTools (F12) -> Console, paste this, press Enter:\n")
        print(EXTRACT_JS)
        print("\n3) Save the printed token:  python minimax_web2api.py --add-token \"<token>\"")
        return 0
    try:
        cfg = load_config(a.config, {"host": a.host, "port": a.port, "region": a.region, "base_url": a.base_url,
                                     "proxy": a.proxy, "tokens_file": a.tokens_file,
                                     "tokens": a.token, "api_keys": a.api_key})
    except ValueError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    tf = Path(cfg["tokens_file"])
    if a.add_token:
        try:
            info = add_token(tf, a.add_token)
        except ValueError as e:
            print(f"invalid token: {e}", file=sys.stderr)
            return 2
        print(f"saved {info.masked()} -> {tf}")
        return 0
    if a.login:
        login_interactive(cfg["region"], cfg["base_url"], tf)
        print(f"token saved -> {tf}")
        return 0
    app = App(cfg)
    if a.check:
        n_ok = 0
        for t in [app.pool.acquire(exclude=set()) for _ in range(len(app.pool))] if len(app.pool) else []:
            try:
                app.upstream.call(USERINFO_URI, {}, t, method="GET")
                print(f"OK    {t.masked()}")
                n_ok += 1
            except UpstreamError as e:
                print(f"FAIL  {t.masked()}  ({e.kind}: {e})")
        if not len(app.pool):
            print("no tokens found; run --login or --add-token")
        return 0 if n_ok else 1
    try:
        srv = Server((cfg["host"], int(cfg["port"])), app)
    except OSError as e:
        print(f"cannot listen on {cfg['host']}:{cfg['port']}: {e.strerror or e}\n"
              "(is another instance already running? choose a different --port)", file=sys.stderr)
        return 1
    host, port = srv.server_address[:2]
    log.info("minimax-web2api %s on http://%s:%s/v1  (upstream %s, %d token(s))",
             __version__, host, port, app.base_url, len(app.pool))
    if not len(app.pool):
        log.warning("no tokens yet - run `--login` (or `--add-token`) in another terminal; they are picked up live")
    if not cfg["api_keys"] and host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("listening on %s WITHOUT api_keys: anyone who can reach this port can use your account", host)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
