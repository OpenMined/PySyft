# third party
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base


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
    user_id = Column(Integer, ForeignKey("syft_user.id"))
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
