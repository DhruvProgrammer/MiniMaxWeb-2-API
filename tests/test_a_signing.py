"""Agent A - signing correctness vs the original JS algorithm (Node oracle)."""
import json
import subprocess
import unittest

from common import m, make_jwt

JS = __file__.replace("test_a_signing.py", "golden_sign.js")

PAYLOADS = [
    {"msg_type": 1, "text": "hello", "chat_type": 1, "attachments": [], "selected_mcp_tools": [], "backend_config": {}, "sub_agent_ids": []},
    {"text": "你好，世界 🌍 émoji 𝒳 and CJK ＆ fullwidth"},
    {"text": 'quotes " backslash \\ slash / newline \n tab \t cr \r ctrl \x01\x1f del \x7f'},
    {"text": "line sep \u2028 para sep \u2029 nbsp \u00a0 zwj \u200d"},
    {"text": "url ?a=b&c=d%20e#frag + plus ~ ! * ' ( )"},
    {"chat_id": 337867554939466},
    {},
    {"text": "x" * 50000},
    {"nested": {"a": [1, 2.5, None, True, "z"], "b": {"c": "d"}}},
]


class Signing(unittest.TestCase):
    def test_matches_original_js(self):
        tok = m.TokenInfo("450123456789+" + make_jwt(device_id="987654321"))
        cases, ts = [], 1_760_000_000
        for i, p in enumerate(PAYLOADS):
            for uri in ("/matrix/api/v1/chat/send_msg", "/v1/api/user/info"):
                cases.append({"uri": uri, "data": p, "ts": ts + i, "user_id": tok.user_id,
                              "device_id": tok.device_id, "token": tok.jwt})
        out = json.loads(subprocess.run(["node", JS], input=json.dumps(cases), capture_output=True,
                                        text=True, check=True).stdout)
        self.assertEqual(len(out), len(cases))
        for c, o in zip(cases, out):
            path, body, headers = m.sign_request(c["uri"], c["data"], tok, "https://agent.minimax.io", now=c["ts"])
            self.assertEqual(path, o["fullUri"], c["data"])
            self.assertEqual(body, o["dataJson"])
            self.assertEqual(headers["yy"], o["yy"], f"yy mismatch for {str(c['data'])[:60]}")
            self.assertEqual(headers["x-signature"], o["signature"])
            self.assertEqual(headers["x-timestamp"], str(c["ts"]))

    def test_empty_device_id_is_omitted_like_js(self):
        tok = m.TokenInfo("450123456789+" + make_jwt(device_id=None))
        # random device id is generated; query must still contain device_id exactly once
        path, _, _ = m.sign_request("/x", {}, tok, "https://a.b", now=1)
        self.assertEqual(path.count("device_id="), 1)

    def test_query_param_order(self):
        tok = m.TokenInfo("450123456789+" + make_jwt())
        path, _, _ = m.sign_request("/x", {}, tok, "https://a.b", now=5)
        keys = [kv.split("=")[0] for kv in path.split("?")[1].split("&")]
        self.assertEqual(keys, ["device_platform", "biz_id", "app_id", "version_code", "uuid", "device_id", "os_name",
                                "browser_name", "device_memory", "cpu_core_num", "browser_language",
                                "browser_platform", "user_id", "screen_width", "screen_height", "unix", "lang", "token"])

    def test_lone_surrogates_do_not_crash(self):
        p = m.build_prompt([{"role": "user", "content": "bad \ud800 surrogate \udfff end"}], 1000)
        tok = m.TokenInfo("450123456789+" + make_jwt())
        _, body, _ = m.sign_request("/x", {"text": p}, tok, "https://a.b", now=1)
        body.encode("utf-8")  # must not raise

    def test_body_bytes_are_what_was_signed(self):
        # upstream receives encode("utf-8") of the same string used for signing
        tok = m.TokenInfo("450123456789+" + make_jwt())
        _, body, headers = m.sign_request("/x", {"text": "日本語"}, tok, "https://a.b", now=9)
        self.assertEqual(m.md5hex("9" + tok.jwt + body), headers["x-signature"])


if __name__ == "__main__":
    unittest.main()
