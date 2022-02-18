# stdlib
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from sympy.core.basic import Basic as BasicSymbol

# syft absolute
import syft as sy

# relative
from ....proto.core.adp.scalar_pb2 import (
    IntermediatePhiScalar as IntermediatePhiScalar_PB,
)
from ...common import UID
from ...common.serde.serializable import serializable
from ..entity import DataSubjectGroup
from ..entity import Entity
from .abstract.intermediate_scalar import IntermediateScalar
from .gamma_scalar import GammaScalar


@serializable()
class IntermediatePhiScalar(IntermediateScalar):
    """
    Serializable superclass for PhiScalars (Scalars with data from one entity).
    This is where all the functionality of a PhiScalar is implemented,
    such as searching for the max Lipshitz value.
    """

    def __init__(
        self,
        poly: BasicSymbol,
        entity: Union[Entity, DataSubjectGroup],
        id: Optional[UID] = None,
    ) -> None:
        super().__init__(poly=poly, id=id)
        self._gamma: Optional[GammaScalar] = None
        self.entity = entity

    def max_lipschitz_wrt_entity(
        self,
        entity: Entity,
    ) -> float:
        """Perform the search for max Lipshitz with respect to the current PhiScalar's entity"""
        return self.gamma.max_lipschitz_wrt_entity(entity=entity)

    @property
    def max_lipschitz(self) -> float:
        """Perform the search for max Lipschitz"""
        return self.gamma.max_lipschitz

    def __mul__(self, other: IntermediateScalar) -> IntermediateScalar:

        # relative
        from .intermediate_gamma_scalar import IntermediateGammaScalar

        if isinstance(
            other, IntermediateGammaScalar
        ):  # PhiScalar * GammaScalar = GammaScalar
            return self.gamma * other

        if not isinstance(
            other, IntermediatePhiScalar
        ):  # PhiScalar * Int/Float/etc = PhiScalar
            return IntermediatePhiScalar(poly=self.poly * other, entity=self.entity)

        # If the entities match, output is PhiScalar, otherwise it should be a GammaScalar
        if self.entity == other.entity:
            return IntermediatePhiScalar(
                poly=self.poly * other.poly, entity=self.entity
            )

        return (
            self.gamma * other.gamma
        )  # Phi(E1) * Phi(E2) = Gamma(E1, E2) | E1,E2 = Entities

    def __add__(self, other: IntermediateScalar) -> IntermediateScalar:

        # relative
        from .intermediate_gamma_scalar import IntermediateGammaScalar

        if isinstance(
            other, IntermediateGammaScalar
        ):  # PhiScalar + GammaScalar = GammaScalar
            return self.gamma + other

        if not isinstance(other, IntermediatePhiScalar):  # PhiScalar + 5 = PhiScalar
            return IntermediatePhiScalar(poly=self.poly + other, entity=self.entity)

        # if other is referencing the same individual
        if self.entity == other.entity:
            return IntermediatePhiScalar(
                poly=self.poly + other.poly, entity=self.entity
            )
        return (
            self.gamma + other.gamma
        )  # Phi(E1) * Phi(E2) = Gamma(E1, E2) | E1,E2 = Entities

    def __sub__(self, other: IntermediateScalar) -> IntermediateScalar:

        # relative
        from .intermediate_gamma_scalar import IntermediateGammaScalar

        if isinstance(other, IntermediateGammaScalar):
            return self.gamma - other

        # if other is a public value
        if not isinstance(other, IntermediatePhiScalar):
            return IntermediatePhiScalar(poly=self.poly - other, entity=self.entity)

        # if other is referencing the same individual
        if self.entity == other.entity:
            return IntermediatePhiScalar(
                poly=self.poly - other.poly, entity=self.entity
            )

        return (
            self.gamma - other.gamma
        )  # Phi(E1) * Phi(E2) = Gamma(E1, E2) | E1,E2 = Entities

    @property
    def gamma(self) -> GammaScalar:
        """Turn the PhiScalar into a GammaScalar, if another entity is involved"""
        if self._gamma is None:
            if (
                self.min_val is not None
                and self.value is not None
                and self.max_val is not None
            ):
                # TODO: Add prime to GammaScalar init
                self._gamma = GammaScalar(
                    min_val=self.min_val,
                    value=self.value,
                    max_val=self.max_val,
                    entity=self.entity,
                    prime=-1,  # TODO: shouldn't we be passing in some kind of prime here?
                )
            else:
                raise Exception("GammaScalar requires min_val, value and max_val")
        return self._gamma

    def _object2proto(self) -> IntermediatePhiScalar_PB:
        return IntermediatePhiScalar_PB(
            id=sy.serialize(self.id, to_proto=True),
            # gamma=sy.serialize(self._gamma, to_proto=True),
            # poly=self._poly if self._poly is not None else None,
            entity=sy.serialize(self.entity, to_proto=True),
        )

    @staticmethod
    def _proto2object(proto: IntermediatePhiScalar_PB) -> "IntermediatePhiScalar":
        intermediate_phi_scalar = IntermediatePhiScalar(
            id=sy.deserialize(proto.id, from_proto=True),
            entity=sy.deserialize(blob=proto.entity, from_proto=True),
            poly=None
            # poly=sy.deserialize(proto.poly)
        )
        # intermediate_phi_scalar._gamma = sy.deserialize(proto.gamma)
        return intermediate_phi_scalar

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return IntermediatePhiScalar_PB
