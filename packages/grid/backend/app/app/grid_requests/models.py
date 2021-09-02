# stdlib
from datetime import datetime
from typing import List
from typing import Optional

# third party
from pydantic import BaseModel


class BaseRequest(BaseModel):
    id: Optional[str]
    date: Optional[datetime]
    user_id: Optional[int]
    user_name: Optional[str]
    object_id: Optional[str]
    reason: Optional[str]
    status: Optional[str]  # Literal['pending', 'accepted', 'denied']
    request_type: Optional[str]
    object_type: Optional[str]
    tags: Optional[List[str]]

    class Config:
        orm_mode = True


class RequestUpdate(BaseRequest):
    status: str  # Literal['pending', 'accepted', 'denied']


class Request(BaseRequest):
    pass


class RequestInDB(BaseRequest):
    verify_key: str
