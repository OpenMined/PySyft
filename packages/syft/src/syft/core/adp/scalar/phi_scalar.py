# future
from __future__ import annotations

# stdlib
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from pymbolic import var

# syft absolute
import syft as sy

# relative
from ....proto.core.adp.scalar_pb2 import PhiScalar as PhiScalar_PB
from ...common import UID
from ...common.serde.serializable import serializable
from ..entity import DataSubjectGroup
from ..entity import Entity
from ..search import ssid2obj
from .abstract.base_scalar import BaseScalar
from .intermediate_phi_scalar import IntermediatePhiScalar


@serializable()
class PhiScalar(BaseScalar, IntermediatePhiScalar):
    """
    Scalar with data from a single entity. Uses all the operations implemented in
    IntermediatePhiScalar to let the user perform operations on the data. Uses the
    BaseScalar class attributes are used to represent the data loaded in by a data owner.
    Builds on both of the above by adding SSIDs to allow the object to be referenced in
    string form and thus use the underlying polynomial libraries.
    """

    def __init__(
        self,
        min_val: Optional[float],
        value: Optional[float],
        max_val: Optional[float],
        entity: Optional[Union[Entity, DataSubjectGroup]] = None,
        id: Optional[UID] = None,
        ssid: Optional[str] = None,
    ) -> None:
        super().__init__(
            min_val=min_val, value=value, max_val=max_val, entity=entity, id=id  # type: ignore
        )
        # The scalar string identifier (SSID) - because we're using polynomial libraries
        # we need to be able to reference this object in string form. The library
        # doesn't know how to process things that aren't strings
        if ssid is None:
            ssid = "_" + self.id.no_dash + "_" + self.entity.to_string()

        self.ssid = ssid

        IntermediatePhiScalar.__init__(
            self, poly=var(self.ssid), entity=self.entity, id=self.id
        )

        ssid2obj[self.ssid] = self

    def _object2proto(self) -> PhiScalar_PB:
        kwargs = {
            "id": sy.serialize(self.id, to_proto=True),
            "entity": sy.serialize(self.entity, to_proto=True),
        }

        for field in ["max_val", "min_val", "value"]:
            if getattr(self, field) is not None:
                kwargs[field] = getattr(self, field)

        return PhiScalar_PB(**kwargs)

    @staticmethod
    def _proto2object(proto: PhiScalar_PB) -> "PhiScalar":
        return PhiScalar(
            id=sy.deserialize(proto.id),
            entity=sy.deserialize(proto.entity),
            min_val=proto.min_val if proto.HasField("min_val") else None,
            max_val=proto.max_val if proto.HasField("max_val") else None,
            value=proto.value if proto.HasField("value") else None,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return PhiScalar_PB
