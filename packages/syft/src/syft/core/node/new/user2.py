# stdlib
import datetime
from enum import Enum
import hashlib
import os
from typing import Optional

# third party
import pydantic
from pydantic import EmailStr
from pydantic.fields import Field
import pytz

# relative
from ..common.node_table.syft_object import SyftObject
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey


class ServiceRole(Enum):
    GUEST = 1


class User(SyftObject):
    # version
    __canonical_name__ = "User"
    __version__ = 1

    # fields
    email: EmailStr
    name: Optional[str] = None
    hashed_password: Optional[bytes] = None
    salt: Optional[bytes] = None
    signing_key: Optional[SyftSigningKey] = None
    verify_key: Optional[SyftVerifyKey] = None
    role: Optional[int] = None
    created_at: Optional[datetime.datetime] = None

    # serde / storage rules
    __attr_state__ = [
        "email",
        "name",
        "hashed_password",
        "salt",
        "signing_key",
        "verify_key",
        "role",
        "created_at",
    ]
    __attr_searchable__ = ["name", "email", "verify_key"]
    __attr_unique__ = ["email", "signing_key"]


class UserCreate(User):
    password: str
    created_at: datetime.datetime = Field(default=datetime.datetime.now(tz=pytz.UTC))
    role: Optional[int] = Field(default=ServiceRole.GUEST.value)

    @pydantic.validator("salt", pre=True, always=True)
    def make_salt(cls, v: Optional[bytes] = None) -> bytes:
        return os.urandom(32) if v is None else v

    @pydantic.validator("password", pre=True)
    def hash_password(cls, v: Optional[bytes], values) -> Optional[bytes]:
        if v:
            salt = values["salt"]
            values["hashed_password"] = hashlib.pbkdf2_hmac(
                "sha256", v.encode(), salt, 10000
            )
        return v

    @property
    def hashed_password(self):
        if self.salt and self.password:
            return hashlib.pbkdf2_hmac(
                "sha256", self.password.encode(), self.salt, 10000
            )


user = UserCreate(name="Shubham", email="email@email.com", password="Hello")
user.dict()
# """
# {'id': None,
#  'email': 'email@email.com',
#  'name': 'Shubham',
#  'hashed_password':
#  b'*\xd3\xe2\x8a\x11\x08\x0e#\xd3 \xdb\xfb=\x88\x04\xb9\xe6\x80>\x98\xab\xcaJ\xcc\xc9\xcdK\x1b`g\xc9\x8f',
#  'salt': b"\xab\xa5\xbb\rnTC\xbd\x03L?\x11 \xb0\xeb\xa8g\x9a\xa2$}\xce\xc3S\xd6e'\xf0\x0c^S2",
#  'signing_key': None,
#  'verify_key': None,
#  'role': 1,
#  'created_at': datetime.datetime(2022, 12, 22, 8, 46, 44, 233793, tzinfo=<UTC>),
#  'password': 'Hello'}
# """
