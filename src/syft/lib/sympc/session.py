# stdlib
import dataclasses
from typing import Any
from typing import List as TypedList
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from sympc.config import Config
from sympc.session import Session

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...decorators import syft_decorator
from ...proto.lib.sympc.session_pb2 import MPCSession as MPCSession_PB
from ...util import aggressive_set_attr
from ..python import Dict
from ..python.primitive_factory import PrimitiveFactory


class SySession(Session):
    @syft_decorator(typechecking=True)
    def __init__(
        self,
        config: Optional[Config] = None,
        parties: Optional[TypedList[Any]] = None,
        ttp: Optional[Any] = None,
        id_obj: Optional[UID] = None,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, parties=parties, ttp=ttp, uuid=id_obj)
        self._id: UID = id_obj if id_obj else UID()
        self.rank = rank

    @syft_decorator(typechecking=True)
    def setup_mpc(self) -> None:
        for rank, party in enumerate(self.parties):
            self.rank = rank
            self.session_ptr.append(self.send(party))

    def _object2proto(self) -> MPCSession_PB:
        id_obj_proto = serialize(obj=self._id)
        conf = PrimitiveFactory.generate_primitive(
            value=dataclasses.asdict(self.config)
        )

        # Can be found from the ring size
        conf.pop("max_value")
        conf.pop("min_value")

        conf_proto = conf._object2proto()

        return MPCSession_PB(id=id_obj_proto, config=conf_proto, rank=self.rank)

    @staticmethod
    def _proto2object(proto: MPCSession_PB) -> "SySession":
        id_obj = deserialize(blob=proto.id)
        rank = proto.rank
        conf_dict = Dict._proto2object(proto=proto.config)

        conf_dict = {key.data: value for key, value in conf_dict.items()}
        conf = Config(**conf_dict)
        session = SySession(config=conf, id_obj=id_obj, rank=rank)
        return session


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
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: MPCSession_PB) -> "SySessionWrapper":
        return SySession._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return MPCSession_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return SySession

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[TypedList[str]],
    ) -> StorableObject:
        setattr(data, "_id", id)
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(
    obj=SySession, name="serializable_wrapper_type", attr=SySessionWrapper
)
