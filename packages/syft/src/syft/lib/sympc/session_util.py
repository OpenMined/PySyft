# stdlib
import dataclasses
from uuid import UUID

# third party
from sympc.config import Config
from sympc.session import Session
from sympc.store import CryptoStore

# syft relative
from ...logger import warning
from ...proto.lib.sympc.session_pb2 import MPCSession as MPCSession_PB
from ..python import Dict
from ..python.primitive_factory import PrimitiveFactory


def protobuf_session_serializer(session: Session) -> MPCSession_PB:
    conf = PrimitiveFactory.generate_primitive(value=dataclasses.asdict(session.config))
    conf_proto = conf._object2proto()

    length_rs = session.ring_size.bit_length()
    rs_bytes = session.ring_size.to_bytes((length_rs + 7) // 8, byteorder="big")

    length_nr_parties = session.ring_size.bit_length()
    nr_parties_bytes = session.nr_parties.to_bytes(
        (length_nr_parties + 7) // 8, byteorder="big"
    )

    protocol = session.protocol.__name__
    protocol_serialized = str.encode(protocol)

    return MPCSession_PB(
        uuid=session.uuid.bytes,
        config=conf_proto,
        ring_size=rs_bytes,
        nr_parties=nr_parties_bytes,
        rank=session.rank,
        protocol=protocol_serialized,
    )


def protobuf_session_deserializer(proto: MPCSession_PB) -> Session:
    id_session = UUID(bytes=proto.uuid)
    rank = proto.rank
    conf_dict = Dict._proto2object(proto=proto.config)
    _conf_dict = {key: value for key, value in conf_dict.items()}
    conf = Config(**_conf_dict)
    ring_size = int.from_bytes(proto.ring_size, "big")
    nr_parties = int.from_bytes(proto.nr_parties, "big")
    protocol_deserialized = proto.protocol.decode()

    session = Session(
        config=conf,
        uuid=id_session,
        ring_size=ring_size,
        protocol=protocol_deserialized,
    )
    session.rank = rank
    session.crypto_store = CryptoStore()
    session.nr_parties = nr_parties

    if "session" in globals():
        warning("Overwritting session for MPC")
        globals()["session"] = session

    return session
