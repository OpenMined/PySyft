# stdlib

# third party
import petlib  # noqa: 401
from petlib.bn import Bn
import zksk

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.util.vendor_bytes_pb2 import VendorBytes as VendorBytes_PB


def object2proto(obj: object) -> VendorBytes_PB:
    proto = VendorBytes_PB()
    proto.obj_type = obj.name  # type: ignore
    if obj.value is not None:  # type: ignore
        proto.content = obj.value.binary  # type: ignore # stores binary of BN
    else:
        proto.content = b""
    return proto


def proto2object(proto: VendorBytes_PB) -> zksk.expr.Secret:
    if proto.content != b"":
        vec = zksk.expr.Secret(proto.obj_type, Bn.from_binary(proto.content))
    else:
        vec = zksk.expr.Secret(proto.obj_type)

    return vec


GenerateWrapper(
    wrapped_type=zksk.expr.Secret,
    import_path="zksk.expr.Secret",
    protobuf_scheme=VendorBytes_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
