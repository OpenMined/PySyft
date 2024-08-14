# future
from __future__ import annotations

# stdlib
from typing import Any

# third party
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from pydantic import field_validator

# relative
from ..serde.serializable import serializable
from ..types.base import SyftBaseModel

SIGNING_KEY_FOR = "Corresponding Public Key"


@serializable(canonical_name="SyftVerifyKey", version=1)
class SyftVerifyKey(SyftBaseModel):
    verify_key: RSAPublicKey

    @field_validator("verify_key", mode="before")
    @classmethod
    def make_verify_key(cls, value: str | RSAPublicKey | bytes) -> RSAPublicKey:
        if isinstance(value, str):
            value = value.encode("utf-8")
        if isinstance(value, bytes):
            value = serialization.load_pem_public_key(value)
        return value

    def __bytes__(self) -> bytes:
        return self.verify_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def __str__(self) -> str:
        return bytes(self).decode("utf-8")

    @staticmethod
    def from_string(key_str: str) -> SyftVerifyKey:
        return SyftVerifyKey(verify_key=serialization.load_pem_public_key(key_str.encode("utf-8")))

    def verify(self, signature: bytes, message: bytes) -> None:
        self.verify_key.verify(
            signature,
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SyftVerifyKey):
            return False
        return self.verify_key == other.verify_key

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(bytes(self))


@serializable(canonical_name="SyftSigningKey", version=1)
class SyftSigningKey(SyftBaseModel):
    signing_key: RSAPrivateKey

    @field_validator("signing_key", mode="before")
    @classmethod
    def make_signing_key(cls, value: str | RSAPrivateKey | bytes) -> RSAPrivateKey:
        if isinstance(value, str):
            value = value.encode("utf-8")
        if isinstance(value, bytes):
            value = serialization.load_pem_private_key(value, password=None)
        return value

    @property
    def verify_key(self) -> SyftVerifyKey:
        return SyftVerifyKey(verify_key=self.signing_key.public_key())

    def __bytes__(self) -> bytes:
        return self.signing_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
    

    def __str__(self) -> str:
        return bytes(self).decode("utf-8")

    @staticmethod
    def generate() -> SyftSigningKey:
        return SyftSigningKey(signing_key=
                              rsa.generate_private_key(
                            public_exponent=65537,
                            key_size=2048,
                        )
                    )
    
    def sign(self, message: bytes) -> bytes:
        return self.signing_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

    @staticmethod
    def from_string(key_str: str) -> SyftSigningKey:
        return SyftSigningKey(signing_key=serialization.load_pem_private_key(key_str.encode("utf-8"), password=None))

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
        return hash(bytes(self))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SyftSigningKey):
            return False
        return self.signing_key.private_numbers() == other.signing_key.private_numbers()


SyftCredentials = SyftVerifyKey | SyftSigningKey


@serializable(canonical_name="UserLoginCredentials", version=1)
class UserLoginCredentials(SyftBaseModel):
    email: str
    password: str
