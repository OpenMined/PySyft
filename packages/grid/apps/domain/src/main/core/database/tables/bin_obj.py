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