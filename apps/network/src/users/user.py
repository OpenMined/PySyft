from .. import db


class User(db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    email = db.Column(db.String())
    hashed_password = db.Column(db.String())
    salt = db.Column(db.String())
    role = db.Column(db.Integer, db.ForeignKey("role.id"))

    def __str__(self):
        return f"<User id: {self.id}, email: {self.email}, role: {self.role}>"
