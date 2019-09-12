from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
import syft as sy


class Worker(db.Model):
    """ Database table that represents workers.

        Collumns:
            id (primary key) : Worker id, used to recover stored workers (UNIQUE).
            worker_objects : Tensor objects stored.
    """

    __tablename__ = "workers"

    id = db.Column(db.String(64), primary_key=True)
    worker_objects = db.relationship("WorkerObject", lazy="dynamic")

    def __repr__(self):
        return f"<Worker {self.id}>"


class TorchModel(db.Model):
    """ Database table that stores torch models.

        Collumns:
            id (primary key) : Model id, (UNIQUE).
            model : Serialized model.
    """

    __tablename__ = "ml_model"

    id = db.Column(db.String(64), primary_key=True)
    model = db.Column(db.LargeBinary(128))
    allow_download = db.Column(db.Boolean, default=False)
    allow_remote_inference = db.Column(db.Boolean, default=False)


class WorkerObject(db.Model):
    """ Database table that represents tensor objects.

        Collumns:
            id (primary key) : Pointer used to map tensor object (UNIQUE).
            description : Description of tensor object.
            data : Tensor value.
            worker_id : ID of worker that owns it (FOREIGN_KEY).
    """

    __tablename__ = "worker_objects"

    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(64), index=True)
    data = db.Column(db.LargeBinary(128))
    worker_id = db.Column(db.String(64), db.ForeignKey("workers.id"))

    def __init__(self, **kwargs):
        super(WorkerObject, self).__init__(**kwargs)

    @property
    def object(self):
        return sy.serde.deserialize(self.data)

    @object.setter
    def object(self, value):
        self.data = sy.serde.serialize(value, force_full_simplification=True)

    def __repr__(self):
        return f"<Tensor {self.id}>"


class TorchTensor(db.Model):
    """Database table that stores torch tensors."""

    __tablename__ = "torch_tensors"

    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.LargeBinary(128))

    @property
    def object(self):
        return sy.serde.deserialize(self.data)

    @object.setter
    def object(self, value):
        self.data = sy.serde.serialize(value)

    def __repr__(self):
        return f"<Tensor {self.id}>"
