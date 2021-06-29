# grid relative
# syft relative
from . import Base
from sqlalchemy import Boolean, Column, Integer, String, DateTime


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
    date = Column(DateTime())
    name = Column(String(255))
    address = Column(String(255))
    sender_address = Column(String(255))
    accepted = Column(Boolean(), default=False)
    pending = Column(Boolean(), default=True)
    handshake_value = Column(String(255))

    def __str__(self):
        return f"< Association Request id : {self.id}, Name: {self.name}, Address: {self.address} , pending: {self.pending}, accepted: {self.accepted}, Date: {self.date}>"
