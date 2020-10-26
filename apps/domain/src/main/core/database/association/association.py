from .. import BaseModel, db


class Association(BaseModel):
    """Association.

    Columns:
        id (Integer, Primary Key): Cycle ID.
        date (TIME): Start time.
        network (String): Network name.
        network_address (String) : Network Address.
    """

    __tablename__ = "association_request"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.DateTime())
    network = db.Column(db.String(255))
    network_address = db.Column(db.String(255))

    def __str__(self):
        return f"< Association id : {self.id}, Network: {self.network}, Network Address: {self.network_address}, Date: {self.date}>"
