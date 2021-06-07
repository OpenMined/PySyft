# grid relative
from .. import BaseModel
from .. import db


class BinaryObject(BaseModel):
    __bind_key__ = "bin_store"
    __tablename__ = "binary_object"

    id = db.Column(db.String(), primary_key=True)
    binary = db.Column(db.LargeBinary())

    def __str__(self):
        return f"<BinaryObject id: {self.id}>"
