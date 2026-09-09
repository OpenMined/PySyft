from pydantic import BaseModel


class RawMessage(BaseModel):
    """Generic container for downloaded bytes + metadata.

    Transport returns this — no deserialization, no encryption awareness.
    """

    data: bytes
    sender_email: str | None = None
    platform_id: str | None = None
    filename: str | None = None
