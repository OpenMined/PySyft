# stdlib

# third party
import petlib

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.petlib.ec_obj_pb2 import EC2_obj


def object2proto(obj: object) -> EC2_obj:
    proto = EC2_obj()
    proto.group_nid = obj.group.nid()  # type: ignore
    proto.data = obj.export()  # type: ignore
    return proto


def proto2object(proto: EC2_obj) -> petlib.ec.EcPt:

    vec = petlib.ec.EcPt(proto.data, petlib.ec.EcGroup(proto.group_nid))
    return vec


GenerateWrapper(
    wrapped_type=petlib.ec.EcPt,
    import_path="petlib.ec.EcPt",
    protobuf_scheme=EC2_obj,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)


def object2proto_group(obj: object) -> EC2_obj:
    proto = EC2_obj()
    proto.group_nid = obj.group.nid()  # type: ignore
    return proto


def proto2object_group(proto: EC2_obj) -> petlib.ec.EcGroup:
    G2 = petlib.ec.EcGroup(proto.group_nid)
    return G2


GenerateWrapper(
    wrapped_type=petlib.ec.EcGroup,
    import_path="petlib.ec.EcGroup",
    protobuf_scheme=EC2_obj,
    type_object2proto=object2proto_group,
    type_proto2object=proto2object_group,
)
