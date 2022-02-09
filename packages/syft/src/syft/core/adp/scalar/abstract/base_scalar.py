# future
from __future__ import annotations

# stdlib
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft as sy

# relative
from .....proto.core.adp.scalar_pb2 import BaseScalar as BaseScalar_PB
from ....common import UID
from ....common.serde.serializable import serializable
from ...entity import DataSubjectGroup
from ...entity import Entity
from .scalar import Scalar


@serializable()
class BaseScalar(Scalar):
    """
    A scalar which stores the root polynomial values. When this is a superclass of
    PhiScalar it represents data that was loaded in by a data owner. When this is a
    superclass of GammaScalar this represents the NODE at which point data from multiple
    entities was combined.
    """

    def __init__(
        self,
        min_val: float,
        value: Optional[float],
        max_val: float,
        entity: Optional[Union[Entity, DataSubjectGroup]] = None,
        id: Optional[UID] = None,
    ) -> None:
        self.id = id if id else UID()
        self._min_val: Optional[float] = float(min_val)
        self._value = float(value) if value is not None else None
        self._max_val: Optional[float] = float(max_val)
        self.entity = entity if entity is not None else Entity()

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def max_val(self) -> Optional[float]:
        return self._max_val

    @property
    def min_val(self) -> Optional[float]:
        return self._min_val

    def _object2proto(self) -> BaseScalar_PB:
        kwargs = {
            "id": sy.serialize(self.id, to_proto=True),
            "entity": sy.serialize(self.entity, to_proto=True),
        }

        for field in ["max_val", "min_val", "value"]:
            if getattr(self, field) is not None:
                kwargs[field] = getattr(self, field)
        pb = BaseScalar_PB(**kwargs)
        return pb

    @staticmethod
    def _proto2object(proto: BaseScalar_PB) -> BaseScalar:
        return BaseScalar(
            min_val=proto.min_val if proto.HasField("min_val") else None,
            max_val=proto.max_val if proto.HasField("max_val") else None,
            value=proto.value if proto.HasField("value") else None,
            entity=sy.deserialize(proto.entity, from_proto=True),
            id=sy.deserialize(proto.id, from_proto=True),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return BaseScalar_PB
