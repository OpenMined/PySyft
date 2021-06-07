# grid relative
from .. import BaseModel
from .. import db


class AssociationRequest(BaseModel):
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

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.DateTime())
    name = db.Column(db.String(255))
    address = db.Column(db.String(255))
    sender_address = db.Column(db.String(255))
    accepted = db.Column(db.Boolean(), default=False)
    pending = db.Column(db.Boolean(), default=True)
    handshake_value = db.Column(db.String(255))

    def __str__(self):
        return f"< Association Request id : {self.id}, Name: {self.name}, Address: {self.address} , pending: {self.pending}, accepted: {self.accepted}, Date: {self.date}>"
