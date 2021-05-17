# third party
import petlib as pl

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.petlib.ecpt_group_pb2 import EcPtGroup as EcPtGroup_PB


def object2proto_group(obj: pl.ec.EcGroup) -> EcPtGroup_PB:
    proto = EcPtGroup_PB()
    proto.group_nid = obj.nid()
    return proto


def proto2object_group(proto: EcPtGroup_PB) -> pl.ec.EcGroup:
    ec_group = pl.ec.EcGroup(proto.group_nid)
    return ec_group


GenerateWrapper(
    wrapped_type=pl.ec.EcGroup,
    import_path="petlib.ec.EcGroup",
    protobuf_scheme=EcPtGroup_PB,
    type_object2proto=object2proto_group,
    type_proto2object=proto2object_group,
)
