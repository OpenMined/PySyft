from typing import Optional

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str
    metadata: str


class TokenPayload(BaseModel):
    sub: Optional[int] = None
