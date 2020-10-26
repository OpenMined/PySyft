from .. import BaseModel, db


class Association(BaseModel):
    """ Association.
    
    Columns:
        id (Integer, Primary Key): Cycle ID.
        date (TIME): Start time.
        name (String): Organization/Domain name.
        address (String) : Organization/Domain Address.
    """

    __tablename__ = "association_request"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.DateTime())
    name = db.Column(db.String(255))
    address = db.Column(db.String(255))


    def __str__(self):
        return f"< Association id : {self.id}, Organization Name: {self.name}, Address: {self.address}, Date: {self.date}>"
