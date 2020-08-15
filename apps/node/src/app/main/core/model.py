# External imports
import syft as sy

# Local imports
from ... import BaseModel, db


class Model(BaseModel):
    """Model table that represents the AI Models.

    Columns:
        id (Int, Primary Key) : Model's id, used to recover stored model.
        version (String) : Model version.
        checkpoints (ModelCheckPoint) : Model Checkpoints. (One to Many relationship)
        fl_process_id (Integer, ForeignKey) : FLProcess Foreign Key.
    """

    __tablename__ = "model_centric_model"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    version = db.Column(db.String())
    checkpoints = db.relationship("ModelCheckPoint", backref="model")
    fl_process_id = db.Column(
        db.Integer, db.ForeignKey("model_centric_fl_process.id"), unique=True
    )

    def __str__(self):
        return f"<Model  id: {self.id}, version: {self.version}>"


class ModelCheckPoint(BaseModel):
    """Model's save points.

    Columns:
        id (Integer, Primary Key): Checkpoint ID.
        value (Binary): Value of the model at a given checkpoint.
        model_id (String, Foreign Key): Model's ID.
    """

    __tablename__ = "model_centric_model_checkpoint"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    value = db.Column(db.LargeBinary)
    number = db.Column(db.Integer)
    alias = db.Column(db.String)
    model_id = db.Column(db.Integer, db.ForeignKey("model_centric_model.id"))

    @property
    def object(self):
        return sy.serde.deserialize(self.value)

    @object.setter
    def object(self):
        self.data = sy.serde.serialize(self.value)

    def __str__(self):
        return f"<CheckPoint id: {self.id}, number: {self.number}, alias: {self.alias}, model_id: {self.model_id}>"
