# third party
import zksk as zk  # noqa: 401

# syft relative
from ...core.common.serde.deserialize import _deserialize as deserialize
from ...core.common.serde.serialize import _serialize as serialize
from ...generate_wrapper import GenerateWrapper
from ...lib.python import Tuple
from ...proto.lib.zksk.nizk_pb2 import NIZK as NIZK_PB


def object2proto(obj: zk.base.NIZK) -> NIZK_PB:
    return NIZK_PB(
        challenge=serialize(obj.challenge, to_proto=True),
        responses=serialize(Tuple(obj.responses), to_proto=True),
        stmt_hash=obj.stmt_hash,
    )


def proto2object(proto: NIZK_PB) -> zk.expr.Secret:
    return zk.base.NIZK(
        challenge=deserialize(proto.challenge, from_proto=True),
        responses=deserialize(proto.responses, from_proto=True).upcast(),
        stmt_hash=proto.stmt_hash,
    )


GenerateWrapper(
    wrapped_type=zk.base.NIZK,
    import_path="zksk.base.NIZK",
    protobuf_scheme=NIZK_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
