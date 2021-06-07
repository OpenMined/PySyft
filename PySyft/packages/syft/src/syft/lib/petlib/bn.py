# third party
import petlib as pl

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.petlib.bn_pb2 import Bn as Bn_PB


def object2proto_group(obj: pl.bn.Bn) -> Bn_PB:
    proto = Bn_PB()
    proto.hex = obj.hex()
    return proto


def proto2object_group(proto: Bn_PB) -> pl.bn.Bn:
    return pl.bn.Bn.from_hex(proto.hex)


GenerateWrapper(
    wrapped_type=pl.bn.Bn,
    import_path="petlib.bn.Bn",
    protobuf_scheme=Bn_PB,
    type_object2proto=object2proto_group,
    type_proto2object=proto2object_group,
)
