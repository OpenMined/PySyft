# future
from __future__ import annotations

# stdlib
import hashlib
from typing import Any

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from pydantic import field_validator

# relative
from ..serde.serializable import serializable
from ..types.base import SyftBaseModel

SIGNING_KEY_FOR = "Corresponding Public Key"


@serializable(canonical_name="SyftVerifyKey", version=1)
class SyftVerifyKey(SyftBaseModel):
    verify_key: VerifyKey

    def __init__(self, verify_key: str | VerifyKey):
        if isinstance(verify_key, str):
            verify_key = VerifyKey(bytes.fromhex(verify_key))
        super().__init__(verify_key=verify_key)

    def __str__(self) -> str:
        return self.verify_key.encode(encoder=HexEncoder).decode("utf-8")

    @staticmethod
    def from_string(key_str: str) -> SyftVerifyKey:
        return SyftVerifyKey(verify_key=VerifyKey(bytes.fromhex(key_str)))

    @property
    def verify(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SyftVerifyKey):
            return False
        return self.verify_key == other.verify_key

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(self.verify_key)


@serializable(canonical_name="SyftSigningKey", version=1)
class SyftSigningKey(SyftBaseModel):
    signing_key: SigningKey

    @field_validator("signing_key", mode="before")
    @classmethod
    def make_signing_key(cls, v: Any) -> Any:
        return SigningKey(bytes.fromhex(v)) if isinstance(v, str) else v

    def deterministic_hash(self) -> str:
        return hashlib.sha256(self.signing_key._seed).hexdigest()

    @property
    def verify_key(self) -> SyftVerifyKey:
        return SyftVerifyKey(verify_key=self.signing_key.verify_key)

    def __str__(self) -> str:
        return self.signing_key.encode(encoder=HexEncoder).decode("utf-8")

    @staticmethod
    def generate() -> SyftSigningKey:
        return SyftSigningKey(signing_key=SigningKey.generate())

    @staticmethod
    def from_string(key_str: str) -> SyftSigningKey:
        return SyftSigningKey(signing_key=SigningKey(bytes.fromhex(key_str)))

    def __repr__(self) -> str:
        return f"<{SIGNING_KEY_FOR}: {self.verify}>"

    def _coll_repr_(self) -> dict[str, str]:
        return {
            SIGNING_KEY_FOR: self.verify,
        }

    @property
    def verify(self) -> str:
        return str(self.verify_key)

    def __hash__(self) -> int:
        return hash(self.signing_key)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SyftSigningKey):
            return False
        return self.signing_key == other.signing_key


SyftCredentials = SyftVerifyKey | SyftSigningKey


@serializable(canonical_name="UserLoginCredentials", version=1)
class UserLoginCredentials(SyftBaseModel):
    email: str
    password: str
