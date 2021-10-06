# stdlib
from datetime import datetime
from typing import List
from typing import Optional

# third party
from pydantic import BaseModel


class BaseRequest(BaseModel):
    id: Optional[str]
    date: Optional[datetime]
    object_id: Optional[str]
    size: Optional[float] = 0.0
    subjects: Optional[int] = 0
    reason: Optional[str]
    status: Optional[str]  # Literal['pending', 'accepted', 'denied']
    request_type: Optional[str]
    object_type: Optional[str]
    tags: Optional[List[str]]

    class Config:
        orm_mode = True


class UserRequest(BaseModel):
    name: str
    email: str
    role: str
    budget_spent: Optional[float] = 0.0
    allocated_budget: Optional[float] = 0.0
    company: Optional[str] = ""
    website: Optional[str] = ""


class AccessRequest(BaseModel):
    user: UserRequest
    req: BaseRequest

    class Config:
        orm_mode = True


class RequestUpdate(BaseRequest):
    status: str  # Literal['pending', 'accepted', 'denied']


class Request(AccessRequest):
    pass


class RequestInDB(BaseRequest):
    verify_key: str
