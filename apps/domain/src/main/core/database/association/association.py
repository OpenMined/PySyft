# grid relative
from .. import BaseModel
from .. import db


class Association(BaseModel):
    """Association.

    Columns:
        id (Integer, Primary Key): Cycle ID.
        date (TIME): Start time.
        network (String): Network name.
        network_address (String) : Network Address.
    """

    __tablename__ = "association"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.DateTime())
    name = db.Column(db.String(255))
    address = db.Column(db.String(255))

    def __str__(self):
        return f"< Association id : {self.id}, Name: {self.name}, Address: {self.address}, Date: {self.date}>"
