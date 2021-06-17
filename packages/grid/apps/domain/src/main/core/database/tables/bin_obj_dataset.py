from .. import BaseModel
from .. import db


class BinObjDataset(BaseModel):
    __tablename__ = "bin_obj_dataset"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    name = db.Column(db.String(256))
    obj = db.Column(db.String(256), db.ForeignKey("bin_object.id"))
    dataset = db.Column(db.String(256), db.ForeignKey("dataset.id"))
    dtype = db.Column(db.String(256))
    shape = db.Column(db.String(256))
