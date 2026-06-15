"""Gmail message wrapper."""

import base64
from typing import Optional


class GmailMessage:
    """Wraps a raw Gmail API message dict with convenient accessors."""

    def __init__(self, data: dict):
        self._data = data

    @property
    def thread_id(self) -> Optional[str]:
        return self._data.get("threadId")

    @property
    def reply_text(self) -> Optional[str]:
        payload = self._data.get("payload", {})
        text = _extract_text_plain(payload)
        if not text:
            return None
        return _strip_quoted_reply(text)

    def get_header(self, name: str) -> Optional[str]:
        headers = self._data.get("payload", {}).get("headers", [])
        for h in headers:
            if h.get("name", "").lower() == name.lower():
                return h.get("value")
        return None


def _extract_text_plain(payload: dict) -> Optional[str]:
    """Recursively extract text/plain content from message payload."""
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})
    data = body.get("data")

    if mime_type == "text/plain" and data:
        return _decode_base64url(data)

    for part in payload.get("parts", []):
        result = _extract_text_plain(part)
        if result:
            return result

    return None


def _decode_base64url(data: str) -> str:
    """Decode base64url-encoded string."""
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="replace")


def _strip_quoted_reply(text: str) -> str:
    """Strip quoted reply text, keeping only the new content."""
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">"):
            break
        if stripped.startswith("On ") and stripped.endswith("wrote:"):
            break
        if stripped.startswith("---------- Forwarded message"):
            break
        result.append(line)

    return "\n".join(result).strip()
