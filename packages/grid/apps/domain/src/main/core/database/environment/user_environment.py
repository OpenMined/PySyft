# grid relative
from .. import BaseModel
from .. import db


class UserEnvironment(BaseModel):
    __tablename__ = "userenvironment"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    user = db.Column(db.Integer, db.ForeignKey("user.id"))
    environment = db.Column(db.Integer, db.ForeignKey("environment.id"))

    def __str__(self):
        return (
            f"<UserGroup id: {self.id}, user: {self.user}, "
            f"group: {self.environment}>"
        )
