# stdlib
from datetime import datetime
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
    daa_pdf: Optional[bytes] = b""
    budget: Optional[float]

    class Config:
        orm_mode = True


class UserCreate(BaseUser):
    email: EmailStr
    role: str = "Data Scientist"
    name: str
    password: str
    institution: Optional[str]
    website: Optional[str]
    budget: float


class ApplicantStatus(BaseModel):
    status: str


class UserUpdate(BaseUser):
    password: Optional[str]
    budget: Optional[float]
    institution: Optional[str]
    website: Optional[str]
    new_password: Optional[str]


class UserCandidate(BaseUser):
    email: EmailStr
    status: str = "pending"
    name: str


class User(BaseUser):
    id: int
    role: Union[int, str]  # TODO: This should be int. Perhaps add role_name instead?
    budget_spent: Optional[float]
    budget: Optional[float]
    institution: Optional[str]
    website: Optional[str]
    added_by: Optional[str]
    created_at: Union[Optional[str], Optional[datetime]]


class UserPrivate(User):
    private_key: str

    def get_signing_key(self) -> SigningKey:
        return SigningKey(self.private_key.encode(), encoder=HexEncoder)


class UserSyft(User):
    hashed_password: str
    salt: str
    verify_key: str
