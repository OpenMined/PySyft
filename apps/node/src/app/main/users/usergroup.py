from ... import BaseModel, db


class UserGroup(BaseModel):
    __tablename__ = "usergroup"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    user = db.Column(db.Integer, db.ForeignKey("user.id"))
    group = db.Column(db.Integer, db.ForeignKey("group.id"))

    def __str__(self):
        return f"<UserGroup id: {self.id}, user: {self.user}, " f"group: {self.group}>"
