from .config import db
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship

import syft as sy


class Worker(db.Model):
    __tablename__ = "workers"

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(64), index=True, unique=True)
    worker_objects = db.relationship("WorkerObject", backref="workers", lazy="dynamic")

    def __repr__(self):
        return f"<Worker {self.id}>"


class WorkerObject(db.Model):
    __tablename__ = "worker_objects"
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(64), index=True, unique=True)
    data = db.Column(db.Binary(64))
    worker_id = db.Column(db.Integer, db.ForeignKey("workers.id"))
    worker = db.relationship("Worker", load_on_pending=True)

    def __init__(self, **kwargs):
        super(WorkerObject, self).__init__(**kwargs)

    @property
    def object(self):
        return sy.serde.deserialize(self.data)

    @object.setter
    def object(self, value):
        self.data = sy.serde.serialize(value)

    def __repr__(self):
        return f"<Tensor {self.id}>"
