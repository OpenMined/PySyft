"""File content in a model: text as it is, bytes as base64 in JSON.

A model keeps ``content`` next to a computed ``content_type``, so that a read
gives bytes back as bytes.
"""

import base64
from typing import Any, Literal

ContentType = Literal["text", "binary"]


def content_type_of(content: str | bytes | None) -> ContentType | None:
    """``"binary"`` for bytes, ``"text"`` for a str, None for no content."""
    if content is None:
        return None
    return "binary" if isinstance(content, bytes) else "text"


def content_to_json(content: str | bytes | None) -> str | None:
    """``content`` as JSON text: bytes go as base64."""
    if isinstance(content, bytes):
        return base64.b64encode(content).decode("ascii")
    return content


def decode_content(data: Any) -> Any:
    """The input of a model, with its base64 ``content`` decoded to bytes when
    its ``content_type`` is ``"binary"``.

    Raises:
        ValueError: if the content is not valid base64.
    """
    if (
        isinstance(data, dict)
        and data.get("content_type") == "binary"
        and isinstance(data.get("content"), str)
    ):
        return {**data, "content": base64.b64decode(data["content"], validate=True)}
    return data
