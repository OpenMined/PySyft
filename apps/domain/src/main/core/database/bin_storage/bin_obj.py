# third party
from syft import deserialize
from syft import serialize
from syft.proto.lib.pandas.frame_pb2 import PandasDataFrame as PandasDataFrame_PB
from syft.proto.lib.torch.tensor_pb2 import TensorProto as TensorProto_PB

# grid relative
from .. import BaseModel
from .. import db

bin_to_proto = {
    TensorProto_PB.__name__: TensorProto_PB,
    PandasDataFrame_PB.__name__: PandasDataFrame_PB,
}


class BinObject(BaseModel):
    __tablename__ = "bin_object"

    id = db.Column(db.String(3072), primary_key=True)
    binary = db.Column(db.LargeBinary(3072))
    protobuf_name = db.Column(db.String(3072))

    @property
    def object(self):
        _proto_struct = bin_to_proto[self.protobuf_name]()
        _proto_struct.ParseFromString(self.binary)
        _obj = deserialize(blob=_proto_struct)
        return _obj

    @object.setter
    def object(self, value):
        serialized_value = serialize(value)
        self.binary = serialized_value.SerializeToString()
        self.protobuf_name = serialized_value.__class__.__name__


class ObjectMetadata(BaseModel):
    __tablename__ = "obj_metadata"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    obj = db.Column(db.Integer, db.ForeignKey("bin_object.id"))
    tags = db.Column(db.JSON())
    description = db.Column(db.String())
    read_permissions = db.Column(db.JSON())
    search_permissions = db.Column(db.JSON())
