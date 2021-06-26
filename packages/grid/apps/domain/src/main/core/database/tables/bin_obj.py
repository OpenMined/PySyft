# third party
from .. import BaseModel
from .. import db
from syft import serialize
from syft import deserialize


class BinObject(BaseModel):
    __tablename__ = "bin_object"

    id = db.Column(db.String(3072), primary_key=True)
    binary = db.Column(db.LargeBinary(3072))

    @property
    def object(self):
        # storing DataMessage, we should unwrap
        return deserialize(self.binary, from_bytes=True)  # TODO: techdebt fix

    @object.setter
    def object(self, value):
        # storing DataMessage, we should unwrap
        self.binary = serialize(value, to_bytes=True)  # TODO: techdebt fix


class ObjectMetadata(BaseModel):
    __tablename__ = "obj_metadata"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # TODO: @Ionesio investigate the difference
    obj = db.Column(db.String(3072), db.ForeignKey("bin_object.id", ondelete="CASCADE"))
    # obj = db.Column(db.String(3072), db.ForeignKey("bin_object.id", ondelete='SET NULL'), nullable=True)

    tags = db.Column(db.JSON())
    description = db.Column(db.String())
    read_permissions = db.Column(db.JSON())
    search_permissions = db.Column(db.JSON())
