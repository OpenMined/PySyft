# stdlib

# third party
import petlib  # noqa: 401
from petlib.bn import Bn
import zksk

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.petlib.ec_obj_pb2 import EC2_obj


def object2proto(obj: object) -> EC2_obj:
    proto = EC2_obj()
    proto.obj_type = obj.name  # type: ignore
    if obj.value is not None:  # type: ignore
        proto.vector = obj.value.binary  # type: ignore # stores binary of BN
    else:
        proto.vector = b""
    return proto


def proto2object(proto: EC2_obj) -> zksk.expr.Secret:
    if proto.vector != b"":
        vec = zksk.expr.Secret(proto.obj_type, Bn.from_binary(proto.vector))
    else:
        vec = zksk.expr.Secret(proto.obj_type)

    return vec


GenerateWrapper(
    wrapped_type=zksk.expr.Secret,
    import_path="zksk.expr.Secret",
    protobuf_scheme=EC2_obj,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
