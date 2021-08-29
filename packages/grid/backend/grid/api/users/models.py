# stdlib
from typing import Optional
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from pydantic import BaseModel
from pydantic import EmailStr


class BaseUser(BaseModel):
    email: Optional[EmailStr]
    name: Optional[str]
    role: Union[Optional[int], Optional[str]]  # TODO: Should be int in SyftUser

    class Config:
        orm_mode = True


class UserCreate(BaseUser):
    email: EmailStr
    role: str
    name: str
    password: str


class UserUpdate(BaseUser):
    password: Optional[str]


class User(BaseUser):
    id: int
    role: Union[int, str]  # TODO: This should be int. Perhaps add role_name instead?


class UserPrivate(User):
    private_key: str

    def get_signing_key(self) -> SigningKey:
        return SigningKey(self.private_key.encode(), encoder=HexEncoder)


class UserSyft(User):
    hashed_password: str
    salt: str
    verify_key: str
