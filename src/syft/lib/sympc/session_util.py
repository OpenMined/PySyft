# stdlib
import dataclasses
from uuid import UUID

# third party
from sympc.config import Config
from sympc.session import Session

# syft relative
from ...proto.lib.sympc.session_pb2 import MPCSession as MPCSession_PB
from ..python import Dict
from ..python.primitive_factory import PrimitiveFactory


def protobuf_session_serializer(session: Session) -> MPCSession_PB:
    conf = PrimitiveFactory.generate_primitive(value=dataclasses.asdict(session.config))
    conf_proto = conf._object2proto()

    length_rs = session.ring_size.bit_length()
    rs_bytes = session.ring_size.to_bytes((length_rs + 7) // 8, byteorder="big")
    return MPCSession_PB(
        uuid=session.uuid.bytes,
        config=conf_proto,
        ring_size=rs_bytes,
        rank=session.rank,
    )


def protobuf_session_deserializer(proto: MPCSession_PB) -> Session:
    id_session = UUID(bytes=proto.uuid)
    rank = proto.rank
    conf_dict = Dict._proto2object(proto=proto.config)
    conf_dict = {key.data: value for key, value in conf_dict.items()}
    conf = Config(**conf_dict)
    ring_size = int.from_bytes(proto.ring_size, "big")

    session = Session(config=conf, uuid=id_session, ring_size=ring_size)
    session.rank = rank

    return session
