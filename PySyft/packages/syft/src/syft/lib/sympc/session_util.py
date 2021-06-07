# stdlib
import dataclasses
from typing import Optional
from uuid import UUID

# third party
import sympc
from sympc.config import Config
from sympc.protocol.protocol import Protocol
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

    uuid = str(session.uuid)

    session_pb = MPCSession_PB(
        uuid=uuid,
        config=conf_proto,
        ring_size=rs_bytes,
        nr_parties=nr_parties_bytes,
        rank=session.rank,
    )

    session_pb.protocol.name = type(session.protocol).__name__
    session_pb.protocol.security_type = session.protocol.security_type

    return session_pb


def protobuf_session_deserializer(proto: MPCSession_PB) -> Session:

    id_session: Optional[str] = None

    if proto.uuid:
        id_session = proto.uuid
        saved_session = sympc.session.get_session(id_session)
        if saved_session and id_session == str(saved_session.uuid):
            return saved_session

    rank = proto.rank
    conf_dict = Dict._proto2object(proto=proto.config)
    _conf_dict = {key: value for key, value in conf_dict.items()}
    conf = Config(**_conf_dict)
    ring_size = int.from_bytes(proto.ring_size, "big")
    nr_parties = int.from_bytes(proto.nr_parties, "big")
    protocol_deserialized = Protocol.registered_protocols[proto.protocol.name]()
    protocol_deserialized.security_type = proto.protocol.security_type

    session = Session(
        config=conf,
        ring_size=ring_size,
        protocol=protocol_deserialized,
    )

    session.rank = rank
    session.crypto_store = CryptoStore()
    session.nr_parties = nr_parties

    if id_session is not None:
        session.uuid = UUID(id_session)
        if saved_session and id_session != str(saved_session.uuid):
            warning("Changing already set session")
        sympc.session.set_session(session)

    return session
