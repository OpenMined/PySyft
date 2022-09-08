# stdlib
from typing import List

# third party
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base
from ....common.uid import UID
from .syft_object import SyftObject


class Request(Base):
    """Request.

    Columns:
        id (Integer, Primary Key): Cycle ID.
        date (TIME): Start time.
        user_id (Integer, Foreign Key): ID of the User that created the request.
        user_name (String): Email of the User that created the request.
        object_id (String): Target object to change in permisions.
        reason (String): Motivation of the request.
        status (String): The status of the request, wich can be 'pending', 'accepted' or 'denied'.
        request_type (String): Wheter the type of the request is 'permissions' or 'budget'.
        verify_key (String): User Verify Key.
    """

    __tablename__ = "request"

    id = Column(String(255), primary_key=True)
    date = Column(DateTime())
    user_id = Column(Integer())
    user_name = Column(String(255))
    user_email = Column(String(255))
    user_role = Column(String(255))
    user_budget = Column(Float())
    institution = Column(String(255))
    website = Column(String(255))
    object_id = Column(String(255))
    reason = Column(String(255))
    status = Column(String(255), default="pending")
    request_type = Column(String(255))
    verify_key = Column(String(255))
    object_type = Column(String(255))
    tags = Column(JSON())
    updated_on = Column(DateTime())
    reviewer_name = Column(String(255))
    reviewer_role = Column(String(255))
    reviewer_comment = Column(String(255))
    requested_budget = Column(Float())
    current_budget = Column(Float())

    def __str__(self) -> str:
        return (
            f"< Request id : {self.id}, user: {self.user_id}, Date: {self.date}, Object: {self.object_id},"
            f" reason: {self.reason}, status: {self.status}, type: {self.object_type} >"
        )


class NoSQLRequest(SyftObject):
    # version
    __canonical_name__ = "Request"
    __version__ = 1

    # fields
    id: UID
    date: str
    user_id: int
    user_name: str
    user_email: str
    user_role: str
    user_budget: float
    institution: str
    website: str
    object_id: str
    reason: str
    status: str = "pending"
    request_type: str
    verify_key: str
    object_type: str
    tags: List[str] = []
    updated_on: str
    reviewer_name: str
    reviewer_role: str
    reviewer_comment: str
    requested_budget: float
    current_budget: float

    # serde / storage rules
    __attr_state__ = [
        "id",
        "date",
        "user_id",
        "user_name",
        "user_email" "user_role",
        "user_budget",
        "institution",
        "website",
        "object_id",
        "reason",
        "status",
        "request_type",
        "verify_key",
        "object_type",
        "tags",
        "updated_on",
        "reviewer_name",
        "reviewer_role",
        "reviewer_comment",
        "requested_budget",
        "current_budget",
    ]
    __attr_searchable__: List[str] = ["id", "user_email"]
    __attr_unique__: List[str] = []
