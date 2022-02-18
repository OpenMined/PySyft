# future
from __future__ import annotations

# stdlib
from typing import List as TypeList
from typing import Optional
from typing import Set as TypeSet
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from pymbolic.interop.sympy import PymbolicToSympyMapper
from pymbolic.mapper.evaluator import EvaluationMapper as EM
from sympy.core.basic import Basic as BasicSymbol

# relative
from .....proto.core.adp.scalar_pb2 import IntermediateScalar as IntermediateScalar_PB
from ....common import UID
from ....common.serde.serializable import serializable
from ...entity import Entity
from ...search import GetSymbolsMapper
from ...search import flatten_and_maximize_poly
from ...search import ssid2obj
from .scalar import Scalar


@serializable()
class IntermediateScalar(Scalar):
    """
    Serializable Scalar class that supports polynomial representations of data.
    It is assumed that IntermediateScalar is immutable and therefore we will cache
    the sympoly creation because poly wont change
    """

    def __init__(self, poly: BasicSymbol, id: Optional[UID] = None) -> None:
        self.poly = poly
        self.id = id if id else UID()
        self._sympoly: Optional[BasicSymbol] = None

        # only initialize these if they aren't already set
        if not hasattr(self, "_min_val"):
            self._min_val: Optional[float] = None
        if not hasattr(self, "_max_val"):
            self._max_val: Optional[float] = None

    @property
    def sympoly(self) -> BasicSymbol:
        """Sympy version of self.poly"""
        if self._sympoly is None:
            self._sympoly = PymbolicToSympyMapper()(self.poly)
        return self._sympoly

    def __mul__(self, other: IntermediateScalar) -> IntermediateScalar:
        raise NotImplementedError

    def __rmul__(self, other: IntermediateScalar) -> IntermediateScalar:
        return self * other

    def __add__(self, other: IntermediateScalar) -> IntermediateScalar:
        raise NotImplementedError

    def __radd__(self, other: IntermediateScalar) -> IntermediateScalar:
        return self + other

    def __sub__(self, other: IntermediateScalar) -> IntermediateScalar:
        raise NotImplementedError

    def __rsub__(self, other: IntermediateScalar) -> IntermediateScalar:
        return other - self  # subtraction is not commutative

    @property
    def input_scalars(self) -> TypeList[Scalar]:
        """Return a list of the PhiScalar & GammaScalar objects used to create this Scalar."""
        phi_gamma_scalars: TypeList[Scalar] = list()
        for free_symbol in self.input_polys:
            ssid = str(free_symbol)
            phi_gamma_scalars.append(ssid2obj[ssid])
        return phi_gamma_scalars

    @property
    def input_entities(self) -> Union[TypeList[Entity], TypeList[None]]:
        """Return a list of the entities involved in the creation of this scalar object."""
        return list(set([x.entity for x in self.input_scalars]))  # type: ignore

    @property
    def input_polys(self) -> TypeSet[BasicSymbol]:
        """Use a mapper object to return the unique set of polynomials"""
        mapper = GetSymbolsMapper()  # type: ignore
        mapper(self.poly)
        return mapper.free_symbols

    @property
    def max_val(self) -> Optional[float]:
        # TODO: Verify that his doesnt change anything with budget spend
        if self._max_val is not None:
            return self._max_val

        if self.poly is not None:
            results = flatten_and_maximize_poly(-self.poly)
            if len(results) >= 1:
                self._max_val = float(-results[-1].fun)
                return self._max_val
        return None

    @property
    def min_val(self) -> Optional[float]:
        # TODO: Verify that his doesnt change anything with budget spend
        if self._min_val is not None:
            return self._min_val

        if self.poly is not None:
            results = flatten_and_maximize_poly(self.poly)
            if len(results) >= 1:
                self._min_val = float(results[-1].fun)
                return self._min_val
        return None

    @property
    def value(self) -> Optional[float]:

        if hasattr(self, "_value_cache"):
            return self._value_cache  # type: ignore

        if self.poly is not None:
            result = EM(
                context={obj.poly.name: obj.value for obj in self.input_scalars}  # type: ignore
            )(self.poly)
            return float(result)
        return None

    def _object2proto(self) -> IntermediateScalar_PB:
        return IntermediateScalar_PB(
            id=self.id._object2proto(),
            # poly=self._poly if self._poly is not None else None,
        )

    @staticmethod
    def _proto2object(proto: IntermediateScalar_PB) -> "IntermediateScalar":
        intermediate_scalar = IntermediateScalar(
            id=UID._proto2object(proto.id),
            poly=None
            # poly=sy.deserialize(proto.poly)
        )
        return intermediate_scalar

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return IntermediateScalar_PB
