"""HMAC-SHA256 device authentication — identical to xiaozhi-esp32-server."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time


class AuthManager:
    """Token = base64url(HMAC-SHA256(client_id|device_id|ts)) . ts"""

    def __init__(self, secret_key: str, expire_seconds: int = 60 * 60 * 24 * 30):
        self.secret_key = secret_key
        self.expire_seconds = max(expire_seconds, 1)

    def _sign(self, content: str) -> str:
        sig = hmac.new(
            self.secret_key.encode(), content.encode(), hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(sig).decode().rstrip("=")

    def generate_token(self, client_id: str, device_id: str) -> str:
        ts = int(time.time())
        return f"{self._sign(f'{client_id}|{device_id}|{ts}')}.{ts}"

    def verify_token(self, token: str, client_id: str, device_id: str) -> bool:
        try:
            sig_part, ts_str = token.split(".")
            ts = int(ts_str)
            if int(time.time()) - ts > self.expire_seconds:
                return False
            return hmac.compare_digest(sig_part, self._sign(f"{client_id}|{device_id}|{ts}"))
        except Exception:
            return False
