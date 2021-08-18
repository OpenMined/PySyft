# CLEANUP NOTES (for ISHAN):
# - remove unused comments
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)

# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import List as TypeList
from typing import Optional
from typing import Set as TypeSet
from typing import Tuple as TypeTuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
import numpy as np
from pymbolic import var
from pymbolic.interop.sympy import PymbolicToSympyMapper
from pymbolic.mapper.evaluator import EvaluationMapper as EM
from scipy import optimize
from sympy.core.basic import Basic as BasicSymbol

# relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.common.serde.serializable import Serializable
from ...core.common.serde.serializable import bind_protobuf
from ...proto.core.adp.scalar_pb2 import (
    IntermediatePhiScalar as IntermediatePhiScalar_PB,
)
from ...proto.core.adp.scalar_pb2 import BaseScalar as BaseScalar_PB
from ...proto.core.adp.scalar_pb2 import GammaScalar as GammaScalar_PB
from ...proto.core.adp.scalar_pb2 import IntermediateScalar as IntermediateScalar_PB
from ...proto.core.adp.scalar_pb2 import PhiScalar as PhiScalar_PB
from .entity import Entity
from .search import GetSymbolsMapper
from .search import create_lookup_tables_for_symbol
from .search import create_searchable_function_from_polynomial
from .search import flatten_and_maximize_poly
from .search import max_lipschitz_via_jacobian
from .search import minimize_function
from .search import ssid2obj


# the most generic class
class Scalar(Serializable):
    def publish(
        self, acc: Any, user_key: VerifyKey, sigma: float = 1.5
    ) -> TypeList[Any]:
        # relative
        from .publish import publish

        return publish([self], acc=acc, sigma=sigma, user_key=user_key)

    @property
    def max_val(self) -> Optional[np.float64]:
        raise NotImplementedError

    @property
    def min_val(self) -> Optional[np.float64]:
        raise NotImplementedError

    @property
    def value(self) -> Optional[np.float64]:
        raise NotImplementedError

    def __str__(self) -> str:
        return (
            "<"
            + str(type(self).__name__)
            + ": ("
            + str(self.min_val)
            + " < "
            + str(self.value)
            + " < "
            + str(self.max_val)
            + ")>"
        )

    def __repr__(self) -> str:
        return str(self)


@bind_protobuf
class IntermediateScalar(Scalar):
    def __init__(self, poly: BasicSymbol, id: Optional[UID] = None) -> None:
        self.poly = poly
        self.id = id if id else UID()

    @property
    def sympoly(self) -> BasicSymbol:
        """Sympy version of self.poly"""
        return PymbolicToSympyMapper()(self.poly)

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
    def input_scalars(self) -> TypeList[Union[PhiScalar, GammaScalar]]:
        phi_gamma_scalars: TypeList[Union[PhiScalar, GammaScalar]] = list()
        for free_symbol in self.input_polys:
            ssid = str(free_symbol)
            phi_gamma_scalars.append(ssid2obj[ssid])
        return phi_gamma_scalars

    @property
    def input_entities(self) -> TypeList[Entity]:
        return list(set([x.entity for x in self.input_scalars]))

    @property
    def input_polys(self) -> TypeSet[BasicSymbol]:
        mapper = GetSymbolsMapper()  # type: ignore
        mapper(self.poly)
        return mapper.free_symbols

    @property
    def max_val(self) -> Optional[np.float64]:
        if self.poly is not None:
            results = flatten_and_maximize_poly(-self.poly)
            if len(results) >= 1:
                return -results[-1].fun
        return None

    @property
    def min_val(self) -> Optional[np.float64]:
        if self.poly is not None:
            results = flatten_and_maximize_poly(self.poly)
            if len(results) >= 1:
                return results[-1].fun
        return None

    @property
    def value(self) -> Optional[np.float64]:
        if self.poly is not None:
            result = EM(
                context={obj.poly.name: obj.value for obj in self.input_scalars}
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
            # poly=deserialize(proto.poly)
        )
        return intermediate_scalar

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return IntermediateScalar_PB


@bind_protobuf
class IntermediatePhiScalar(IntermediateScalar):
    def __init__(
        self, poly: BasicSymbol, entity: Entity, id: Optional[UID] = None
    ) -> None:
        super().__init__(poly=poly, id=id)
        self._gamma: Optional[GammaScalar] = None
        self.entity = entity

    def max_lipschitz_wrt_entity(
        self,
        entity: Entity,
    ) -> float:
        return self.gamma.max_lipschitz_wrt_entity(entity=entity)

    @property
    def max_lipschitz(self) -> float:
        return self.gamma.max_lipschitz

    def __mul__(self, other: IntermediateScalar) -> IntermediateScalar:

        if isinstance(other, IntermediateGammaScalar):
            return self.gamma * other

        if not isinstance(other, IntermediatePhiScalar):
            return IntermediatePhiScalar(poly=self.poly * other, entity=self.entity)

        # if other is referencing the same individual
        if self.entity == other.entity:
            return IntermediatePhiScalar(
                poly=self.poly * other.poly, entity=self.entity
            )

        return self.gamma * other.gamma

    def __add__(self, other: IntermediateScalar) -> IntermediateScalar:

        if isinstance(other, IntermediateGammaScalar):
            return self.gamma + other

        # if other is a public value
        if not isinstance(other, IntermediatePhiScalar):
            return IntermediatePhiScalar(poly=self.poly + other, entity=self.entity)

        # if other is referencing the same individual
        if self.entity == other.entity:
            return IntermediatePhiScalar(
                poly=self.poly + other.poly, entity=self.entity
            )
        return self.gamma + other.gamma

    def __sub__(self, other: IntermediateScalar) -> IntermediateScalar:

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

        return self.gamma - other.gamma

    @property
    def gamma(self) -> GammaScalar:
        if self._gamma is None:
            self._gamma = GammaScalar(
                min_val=self.min_val,
                value=self.value,
                max_val=self.max_val,
                entity=self.entity,
            )
        return self._gamma

    def _object2proto(self) -> IntermediatePhiScalar_PB:
        return IntermediatePhiScalar_PB(
            id=serialize(self.id, to_proto=True),
            # gamma=serialize(self._gamma, to_proto=True),
            # poly=self._poly if self._poly is not None else None,
            entity=serialize(self.entity, to_proto=True),
        )

    @staticmethod
    def _proto2object(proto: IntermediatePhiScalar_PB) -> "IntermediatePhiScalar":
        intermediate_phi_scalar = IntermediatePhiScalar(
            entity=deserialize(blob=proto.entity, from_proto=True),
            poly=None
            # poly=deserialize(proto.poly)
        )
        intermediate_phi_scalar.id = deserialize(proto.id, from_proto=True)
        # intermediate_phi_scalar._gamma = deserialize(proto.gamma)
        return intermediate_phi_scalar

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return IntermediatePhiScalar_PB


@bind_protobuf
class BaseScalar(Scalar):
    """A scalar which stores the root polynomial values. When this is a superclass of
    PhiScalar it represents data that was loaded in by a data owner. When this is a
    superclass of GammaScalar this represents the node at which point data from multiple
    entities was combined."""

    def __init__(
        self,
        min_val: Optional[float],
        value: Optional[float],
        max_val: Optional[float],
        entity: Optional[Entity] = None,
        id: Optional[UID] = None,
    ) -> None:
        self.id = id if id else UID()
        self._min_val = float(min_val) if min_val is not None else None
        self._value = float(value) if value is not None else None
        self._max_val = float(max_val) if max_val is not None else None
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
            "id": serialize(self.id, to_proto=True),
            "entity": serialize(self.entity, to_proto=True),
        }

        for field in ["max_val", "min_val", "value"]:
            if getattr(self, field):
                kwargs[field] = getattr(self, field)

        return BaseScalar_PB(**kwargs)

    @staticmethod
    def _proto2object(proto: BaseScalar_PB) -> BaseScalar:
        return BaseScalar(
            min_val=proto.min_val if proto.HasField("min_val") else None,
            max_val=proto.max_val if proto.HasField("max_val") else None,
            value=proto.value if proto.HasField("value") else None,
            entity=deserialize(proto.entity, from_proto=True),
            id=deserialize(proto.id, from_proto=True),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return BaseScalar_PB


@bind_protobuf
class PhiScalar(BaseScalar, IntermediatePhiScalar):
    """A scalar over data from a single entity"""

    def __init__(
        self,
        min_val: Optional[float],
        value: Optional[float],
        max_val: Optional[float],
        entity: Optional[Entity] = None,
        id: Optional[UID] = None,
        ssid: Optional[str] = None,
    ) -> None:
        super().__init__(
            min_val=min_val, value=value, max_val=max_val, entity=entity, id=id
        )
        # The scalar string identifier (SSID) - because we're using polynomial libraries
        # we need to be able to reference this object in string form. The library
        # doesn't know how to process things that aren't strings
        if ssid is None:
            ssid = "_" + self.id.no_dash + "_" + self.entity.id.no_dash

        self.ssid = ssid

        IntermediatePhiScalar.__init__(
            self, poly=var(self.ssid), entity=self.entity, id=id
        )

        ssid2obj[self.ssid] = self

    def _object2proto(self) -> PhiScalar_PB:
        kwargs = {
            "id": serialize(self.id, to_proto=True),
            "entity": serialize(self.entity, to_proto=True),
        }

        for field in ["max_val", "min_val", "value"]:
            if getattr(self, field):
                kwargs[field] = getattr(self, field)

        return PhiScalar_PB(**kwargs)

    @staticmethod
    def _proto2object(proto: PhiScalar_PB) -> "PhiScalar":
        return PhiScalar(
            id=deserialize(proto.id),
            entity=deserialize(proto.entity),
            min_val=proto.min_val if proto.HasField("min_val") else None,
            max_val=proto.max_val if proto.HasField("max_val") else None,
            value=proto.value if proto.HasField("value") else None,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return PhiScalar_PB


class IntermediateGammaScalar(IntermediateScalar):
    """ """

    def __add__(self, other) -> IntermediateGammaScalar:
        if isinstance(other, Scalar):
            if isinstance(other, IntermediatePhiScalar):
                other = other.gamma
            return IntermediateGammaScalar(poly=self.poly + other.poly)
        return IntermediateGammaScalar(poly=self.poly + other)

    def __sub__(self, other) -> IntermediateGammaScalar:
        if isinstance(other, Scalar):
            if isinstance(other, IntermediatePhiScalar):
                other = other.gamma
            return IntermediateGammaScalar(poly=self.poly - other.poly)
        return IntermediateGammaScalar(poly=self.poly - other)

    def __mul__(self, other) -> IntermediateGammaScalar:
        if isinstance(other, Scalar):
            if isinstance(other, IntermediatePhiScalar):
                other = other.gamma
            return IntermediateGammaScalar(poly=self.poly * other.poly)
        return IntermediateGammaScalar(poly=self.poly * other)

    def max_lipschitz_via_explicit_search(
        self, force_all_searches: bool = False
    ) -> TypeTuple[TypeList[optimize.OptimizeResult], np.float64]:

        r1 = np.array([x.poly for x in self.input_scalars])

        r2_diffs = np.array(
            [
                GammaScalar(x.min_val, x.value, x.max_val, entity=x.entity).poly
                for x in self.input_scalars
            ]
        )
        r2 = r1 + r2_diffs

        fr1 = self.poly
        fr2 = self.poly.copy().subs({x[0]: x[1] for x in list(zip(r1, r2))})

        left = np.sum(np.square(fr1 - fr2)) ** 0.5
        right = np.sum(np.square(r1 - r2)) ** 0.5

        C = -left / right

        i2s, s2i = create_lookup_tables_for_symbol(C)
        search_fun = create_searchable_function_from_polynomial(
            poly=C, symbol2index=s2i
        )

        r1r2diff_zip = list(zip(r1, r2_diffs))

        s2range = {}
        for _input_scalar, _additive_counterpart in r1r2diff_zip:
            input_scalar = ssid2obj[_input_scalar.name]
            additive_counterpart = ssid2obj[_additive_counterpart.name]

            s2range[input_scalar.ssid] = (input_scalar.min_val, input_scalar.max_val)
            s2range[additive_counterpart.ssid] = (
                input_scalar.min_val,
                input_scalar.max_val,
            )

        rranges = list()
        for _, symbol in enumerate(i2s):
            rranges.append(s2range[symbol])

        r2_indices_list = list()
        min_max_list = list()
        for r2_val in r2:
            r2_syms = [ssid2obj[x.name] for x in r2_val.free_symbols]
            r2_indices = [s2i[x.ssid] for x in r2_syms]

            r2_indices_list.append(r2_indices)
            min_max_list.append((r2_syms[0].min_val, r2_syms[0].max_val))

        functions = list()
        for i in range(2):
            f1 = (
                lambda x, i=i: x[r2_indices_list[i][0]]
                + x[r2_indices_list[i][1]]
                + min_max_list[i][0]
            )
            f2 = (
                lambda x, i=i: -(x[r2_indices_list[i][0]] + x[r2_indices_list[i][1]])
                + min_max_list[i][1]
            )

            functions.append(f1)
            functions.append(f2)

        constraints = [{"type": "ineq", "fun": f} for f in functions]

        def non_negative_additive_terms(symbol_vector: np.ndarray) -> np.float64:
            out = 0
            for index in [s2i[x.name] for x in r2_diffs]:
                out += symbol_vector[index] ** 2
            # theres a small bit of rounding error from this constraint - this should
            # only be used as a double check or as a backup!!!
            return out ** 0.5 - 1 / 2 ** 16

        constraints.append({"type": "ineq", "fun": non_negative_additive_terms})
        results = minimize_function(
            f=search_fun,
            rranges=rranges,
            constraints=constraints,
            force_all_searches=force_all_searches,
        )

        return results, C

    def max_lipschitz_via_jacobian(
        self,
        input_entity: Optional[Entity] = None,
        data_dependent: bool = True,
        force_all_searches: bool = False,
        try_hessian_shortcut: bool = False,
    ) -> TypeList[optimize.OptimizeResult]:
        return max_lipschitz_via_jacobian(
            scalars=[self],
            input_entity=input_entity,
            data_dependent=data_dependent,
            force_all_searches=force_all_searches,
            try_hessian_shortcut=try_hessian_shortcut,
        )

    @property
    def max_lipschitz(self) -> float:
        result = self.max_lipschitz_via_jacobian()[0][-1]
        if isinstance(result, float):
            return -result
        else:
            return -float(result.fun)

    def max_lipschitz_wrt_entity(self, entity: Entity) -> float:
        result = self.max_lipschitz_via_jacobian(input_entity=entity)[0][-1]
        if isinstance(result, float):
            return -result
        else:
            return -float(result.fun)


@bind_protobuf
class GammaScalar(BaseScalar, IntermediateGammaScalar):
    """A scalar over data from multiple entities"""

    def __init__(
        self,
        min_val: float,
        value: float,
        max_val: float,
        entity: Optional[Entity] = None,
        id: Optional[UID] = None,
        ssid: Optional[str] = None,
    ) -> None:
        super().__init__(
            min_val=min_val, value=value, max_val=max_val, entity=entity, id=id
        )

        # The scalar string identifier (SSID) - because we're using polynomial libraries
        # we need to be able to reference this object in string form. The library
        # doesn't know how to process things that aren't strings
        if ssid is None:
            ssid = "_" + self.id.no_dash + "_" + self.entity.id.no_dash

        self.ssid = ssid

        IntermediateGammaScalar.__init__(self, poly=var(self.ssid), id=id)

        ssid2obj[self.ssid] = self

    def _object2proto(self) -> GammaScalar_PB:
        kwargs = {
            "id": serialize(self.id, to_proto=True),
            "entity": serialize(self.entity, to_proto=True),
        }

        for field in ["max_val", "min_val", "value"]:
            if getattr(self, field):
                kwargs[field] = getattr(self, field)
        return GammaScalar_PB(**kwargs)

    @staticmethod
    def _proto2object(proto: GammaScalar_PB) -> GammaScalar:
        scalar = GammaScalar(
            min_val=proto.min_val if proto.HasField("min_val") else None,
            max_val=proto.max_val if proto.HasField("max_val") else None,
            value=proto.value if proto.HasField("value") else None,
            entity=deserialize(proto.entity),
        )
        scalar.id = deserialize(proto.id, from_proto=True)
        return scalar

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GammaScalar_PB
