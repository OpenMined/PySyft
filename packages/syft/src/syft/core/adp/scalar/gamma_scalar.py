# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from pymbolic import var

# syft absolute
import syft as sy

# relative
from ....proto.core.adp.scalar_pb2 import GammaScalar as GammaScalar_PB
from ...common import UID
from ...common.serde.serializable import serializable
from ..entity import DataSubjectGroup
from ..entity import Entity
from ..search import ssid2obj
from .abstract.base_scalar import BaseScalar
from .intermediate_gamma_scalar import IntermediateGammaScalar


@serializable()
class GammaScalar(BaseScalar, IntermediateGammaScalar):
    """
    A scalar over data from multiple entities. Uses all the operators from IntermediateGammaScalar.
    Uses BaseScalar to represent the node at which point data from multiple entities was combined.
    Finally, adds SSIDs to allow the underlying polynomial libraries to work and run.
    """

    def __init__(
        self,
        min_val: float,
        value: float,
        max_val: float,
        prime: int,
        entity: Optional[Union[Entity, DataSubjectGroup]] = None,
        id: Optional[UID] = None,
        ssid: Optional[str] = None,
    ) -> None:
        super().__init__(
            min_val=min_val, value=value, max_val=max_val, entity=entity, id=id
        )

        self.prime = prime

        # The scalar string identifier (SSID) - because we're using polynomial libraries
        # we need to be able to reference this object in string form. The library
        # doesn't know how to process things that aren't strings
        if ssid is None:
            ssid = "_" + self.id.no_dash + "_" + self.entity.to_string()
        self.ssid = ssid

        IntermediateGammaScalar.__init__(
            self, poly=var(self.ssid), min_val=min_val, max_val=max_val, id=self.id
        )

        ssid2obj[self.ssid] = self

    def _object2proto(self) -> GammaScalar_PB:
        kwargs = {
            "id": sy.serialize(self.id, to_proto=True),
            "prime": self.prime,
        }
        entity_data = sy.serialize(self.entity, to_proto=True)
        if isinstance(self.entity, Entity):
            kwargs["entity"] = entity_data
        elif isinstance(self.entity, DataSubjectGroup):
            kwargs["dsg"] = entity_data
        else:
            raise ValueError(
                f"Invalid type: {type(self.entity)} in GammaScalar Entity field."
            )

        for field in ["max_val", "min_val", "value"]:
            if getattr(self, field) is not None:
                kwargs[field] = getattr(self, field)

        return GammaScalar_PB(**kwargs)

    @staticmethod
    def _proto2object(proto: GammaScalar_PB) -> "GammaScalar":
        if proto.HasField("entity"):
            entity_data = sy.deserialize(proto.entity)
        elif proto.HasField("dsg"):
            entity_data = sy.deserialize(proto.dsg)
        return GammaScalar(
            id=sy.deserialize(proto.id, from_proto=True),
            min_val=proto.min_val if proto.HasField("min_val") else None,
            max_val=proto.max_val if proto.HasField("max_val") else None,
            value=proto.value if proto.HasField("value") else None,
            entity=entity_data,
            prime=proto.prime,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GammaScalar_PB

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GammaScalar):
            return (
                self.min_val == other.min_val
                and self.max_val == other.max_val
                and self.value == other.value
                and self.entity == other.entity
                and self.prime == other.prime
            )
        return self == other

    def __hash__(self) -> int:
        return (
            hash(self.min_val)
            + hash(self.max_val)
            + hash(self.prime)
            + hash(self.entity)
            + hash(self.value)
        )
