# stdlib
from datetime import datetime
from typing import List
from typing import Optional

# third party
from pydantic import BaseModel


class Review(BaseModel):
    name: Optional[str] = ""
    role: Optional[str] = ""
    comment: Optional[str] = ""
    updated_on: Optional[datetime]


class BaseRequest(BaseModel):
    id: Optional[str]
    date: Optional[datetime]
    status: Optional[str]  # Literal['pending', 'accepted', 'denied']
    reason: Optional[str]
    request_type: Optional[str]
    review: Optional[Review]


class DataAccessRequest(BaseRequest):
    object_id: Optional[str]
    size: Optional[float] = 0.0
    subjects: Optional[int] = 0
    object_type: Optional[str]
    tags: Optional[List[str]]

    class Config:
        orm_mode = True


class UserRequest(BaseModel):
    name: str
    email: str
    role: str
    budget_spent: Optional[float] = 0.0
    current_budget: Optional[float] = 0.0
    institution: Optional[str] = ""
    website: Optional[str] = ""


class AccessRequestResponse(BaseModel):
    user: UserRequest
    req: DataAccessRequest

    class Config:
        orm_mode = True


class BudgetRequest(BaseRequest):
    current_budget: Optional[float] = 0.0
    requested_budget: Optional[float] = 0.0


class BudgetRequestResponse(BaseModel):
    user: UserRequest
    req: BudgetRequest

    class Config:
        orm_mode = True


class RequestUpdate(BaseRequest):
    status: str  # Literal['pending', 'accepted', 'denied']


class Request(AccessRequestResponse):
    pass


class RequestInDB(BaseRequest):
    verify_key: str
