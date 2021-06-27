# grid relative
# syft relative
from .. import BaseModel
from .. import db


class Dataset(BaseModel):
    __tablename__ = "dataset"

    id = db.Column(db.String(256), primary_key=True)
    manifest = db.Column(db.String(2048))
    description = db.Column(db.String(2048))
    tags = db.Column(db.JSON())
