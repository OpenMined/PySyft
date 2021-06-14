# third party
from syft import deserialize
from syft import serialize
from syft.proto.lib.pandas.frame_pb2 import PandasDataFrame as PandasDataFrame_PB
from syft.proto.lib.torch.module_pb2 import Module as Module_PB
from syft.proto.lib.torch.tensor_pb2 import TensorProto as TensorProto_PB

# grid relative
from .. import BaseModel
from .. import db


class BinObject(BaseModel):
    __tablename__ = "bin_object"

    id = db.Column(db.String(3072), primary_key=True)
    binary = db.Column(db.LargeBinary(3072))
    obj_name = db.Column(db.String(3072))

    @property
    def object(self):
        # storing DataMessage, we should unwrap
        return deserialize(self.binary, from_bytes=True)  # TODO: techdebt fix

    @object.setter
    def object(self, value):
        # storing DataMessage, we should unwrap
        self.binary = serialize(value, to_bytes=True)  # TODO: techdebt fix
        self.obj_name = type(value).__name__


class ObjectMetadata(BaseModel):
    __tablename__ = "obj_metadata"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    obj = db.Column(db.Integer, db.ForeignKey("bin_object.id"))
    tags = db.Column(db.JSON())
    description = db.Column(db.String())
    read_permissions = db.Column(db.JSON())
    search_permissions = db.Column(db.JSON())
