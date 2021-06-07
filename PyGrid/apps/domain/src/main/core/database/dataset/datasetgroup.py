# grid relative
from .. import BaseModel
from .. import db


class DatasetGroup(BaseModel):
    __tablename__ = "datasetgroup"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    bin_object = db.Column(db.String(), db.ForeignKey("bin_object.id"))
    dataset = db.Column(db.String(), db.ForeignKey("json_object.id"))

    def __str__(self):
        return (
            f"<DatasetGroup id: {self.id}, bin_object: {self.bin_object}, "
            f"dataset: {self.dataset}>"
        )


class Dataset(BaseModel):
    __tablename__ = "dataset"

    id = db.Column(db.String(256), primary_key=True)
    manifest = db.Column(db.String(2048))
    description = db.Column(db.String(2048))
    tags = db.Column(db.JSON())


class BinObjDataset(BaseModel):
    __tablename__ = "bin_obj_dataset"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    name = db.Column(db.String(256))
    obj = db.Column(db.String(256), db.ForeignKey("bin_object.id"))
    dataset = db.Column(db.String(256), db.ForeignKey("dataset.id"))
    dtype = db.Column(db.String(256))
    shape = db.Column(db.String(256))
