# third party
import zksk as zk  # noqa: 401

# syft relative
from ...core.common.serde.deserialize import _deserialize as deserialize
from ...core.common.serde.serialize import _serialize as serialize
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.zksk.secret_pb2 import Secret as Secret_PB


def object2proto(obj: zk.expr.Secret) -> Secret_PB:
    proto = Secret_PB()
    proto.name = obj.name
    proto.value = serialize(obj.value, to_bytes=True)
    return proto


def proto2object(proto: Secret_PB) -> zk.expr.Secret:
    return zk.expr.Secret(
        name=proto.name, value=deserialize(proto.value, from_bytes=True)
    )


GenerateWrapper(
    wrapped_type=zk.expr.Secret,
    import_path="zksk.expr.Secret",
    protobuf_scheme=Secret_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
