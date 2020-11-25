# stdlib
import dataclasses
from typing import List as TypedList
from typing import Optional
from uuid import UUID

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from sympc.config import Config
from sympc.session import Session

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...proto.lib.sympc.session_pb2 import MPCSession as MPCSession_PB
from ...util import aggressive_set_attr
from ..python import Dict
from ..python.primitive_factory import PrimitiveFactory


class SySessionWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> MPCSession_PB:
        session: Session = self.value
        conf = PrimitiveFactory.generate_primitive(
            value=dataclasses.asdict(session.config)
        )

        # Can be found from the ring size
        conf.pop("max_value")
        conf.pop("min_value")

        conf_proto = conf._object2proto()

        return MPCSession_PB(
            uuid=session.uuid.bytes, config=conf_proto, rank=session.rank
        )

    @staticmethod
    def _data_proto2object(proto: MPCSession_PB) -> "SySessionWrapper":
        rank = proto.rank
        conf_dict = Dict._proto2object(proto=proto.config)
        conf_dict = {key.data: value for key, value in conf_dict.items()}
        conf = Config(**conf_dict)
        id_session = UUID(bytes=proto.uuid)

        session = Session(config=conf, uuid=id_session)
        session.rank = rank

        return session

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return MPCSession_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Session

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[TypedList[str]],
    ) -> StorableObject:
        data.id = id
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=Session, name="serializable_wrapper_type", attr=SySessionWrapper
)
