# stdlib

# third party
import petlib

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.petlib.ec_obj_pb2 import EcPt_PB


def object2proto(obj: object) -> EcPt_PB:
    proto = EcPt_PB()
    proto.group_nid = obj.group.nid()  # type: ignore
    proto.data = obj.export()  # type: ignore
    return proto


def proto2object(proto: EcPt_PB) -> petlib.ec.EcPt:
    eg = petlib.ec.EcGroup(proto.group_nid)
    vec = petlib.ec.EcPt.from_binary(proto.data, eg)
    return vec


GenerateWrapper(
    wrapped_type=petlib.ec.EcPt,
    import_path="petlib.ec.EcPt",
    protobuf_scheme=EcPt_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)


def object2proto_group(obj: object) -> EcPt_PB:
    proto = EcPt_PB()
    proto.group_nid = obj.nid()  # type: ignore
    return proto


def proto2object_group(proto: EcPt_PB) -> petlib.ec.EcGroup:
    G2 = petlib.ec.EcGroup(proto.group_nid)
    return G2


GenerateWrapper(
    wrapped_type=petlib.ec.EcGroup,
    import_path="petlib.ec.EcGroup",
    protobuf_scheme=EcPt_PB,
    type_object2proto=object2proto_group,
    type_proto2object=proto2object_group,
)
