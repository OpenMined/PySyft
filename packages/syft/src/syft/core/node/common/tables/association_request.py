# stdlib
from datetime import datetime

# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class AssociationRequest(Base):
    """Association Request.

    Columns:
        id (Integer, Primary Key): Cycle ID.
        date (TIME): Start time.
        name (String): Organization / Domain name.
        address (String) : Organization / Domain Address.
        accepted (Bool) :  If request was accepted or not.
        pending (Bool) : If association request is pending.
        handshake_value (String) : Association request unique identifier
    """

    __tablename__ = "association_request"

    id = Column(Integer, primary_key=True, autoincrement=True)
    requested_date = Column(String(255))
    accepted_date = Column(String(255), default="")
    address = Column(String(255), default="")
    node = Column(String(255), default="")
    name = Column(String(255), default="")
    email = Column(String(255), default="")
    reason = Column(String(255), default="")
    status = Column(String(255), default="")

    def __str__(self):
        return f"< Association Request id : {self.id}, Name: {self.name}, Address: {self.address} , pending: {self.pending}, accepted: {self.accepted}, Date: {self.date}>"
