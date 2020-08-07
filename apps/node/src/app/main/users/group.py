from ... import BaseModel, db


class Group(BaseModel):
    __tablename__ = "group"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    name = db.Column(db.String())

    def __str__(self):
        return f"<Group id: {self.id}, name: {self.name}>"
