from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


##TODO(amr): create PyGridApp Generic Model for all apps
class Domain(db.Model):
    __tablename__ = "domains"

    id = db.Column(db.Integer(), primary_key=True)
    # user_id = db.Column(db.Integer())  # TODO: foreign key
    provider = db.Column(db.String(64))
    region = db.Column(db.String(64))
    instance_type = db.Column(db.String(64))
    state = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.now())
    destroyed_at = db.Column(db.DateTime, default=datetime.now())

    def __str__(self):
        return f"<Domain id: {self.id}, Instance: {self.instance_type}>"

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
