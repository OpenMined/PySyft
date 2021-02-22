# stdlib

# third party
from sympc.session import Session

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.sympc.session_pb2 import MPCSession as MPCSession_PB
from .session_util import protobuf_session_deserializer
from .session_util import protobuf_session_serializer


def object2proto(obj: object) -> MPCSession_PB:
    proto = protobuf_session_serializer(obj)
    return proto


def proto2object(proto: MPCSession_PB) -> Session:
    session = protobuf_session_deserializer(proto)
    return session


GenerateWrapper(
    wrapped_type=Session,
    import_path="sympc.session.Session",
    protobuf_scheme=MPCSession_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
