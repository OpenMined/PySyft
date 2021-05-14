# stdlib
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import random
from string import ascii_letters
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult
from sympy import Symbol
from sympy import diff
from sympy import symbols

# syft absolute
from syft.core.common.serde import Serializable

# syft relative
from ...core.common import UID
from ...core.common.serde.serializable import bind_protobuf
from ...proto.lib.adp.scalar_pb2 import Scalar as Scalar_PB
from .entity import Entity
from .idp_gaussian_mechanism import iDPGaussianMechanism

# scalar_name2obj is used to look and extract value, min_val and max_val
# if we are able to store these in the name string instead we could extract them
# at the point of use instead of lookup
scalar_name2obj = {}


@lru_cache(maxsize=None)
def search(
    run_specific_args: Callable, rranges: Tuple, binary: bool = True
) -> OptimizeResult:

    if binary:

        slices = list()
        for r in rranges:
            slices.append(slice(r[0], r[1] + 0.00001, (r[1] - r[0])))
        all_brute_results = optimize.brute(
            func=run_specific_args, ranges=slices, finish=None, full_output=True
        )
        # print(all_brute_results[0])

        brute_results = float(all_brute_results[1])
        return brute_results

    else:

        shgo_results = float(optimize.shgo(run_specific_args, rranges).fun)
        return shgo_results

def create_lookup_tables_for_symbol(polynomial):

    index2symbol = [str(x) for x in polynomial.free_symbols]
    symbol2index = {sym: i for i, sym in enumerate(index2symbol)}

    return index2symbol, symbol2index


@bind_protobuf
class Scalar(Serializable):
    def __init__(
        self,
        *,
        value: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        entity: Optional[Union[str, Entity]] = None,
        poly: Optional[Symbol] = None,
        name: Optional[str] = None,
        id: Optional[UID] = None,
        is_discrete: bool = False
    ):
        self.id = id if id else UID()
        if name is None:
            name = "".join([random.choice(ascii_letters) for _ in range(5)])

        self.name = name
        self._value = value
        self._min_val = min_val
        self._max_val = max_val
        self.entity = entity
        self.is_discrete = is_discrete

        if poly is not None:
            # if this Scalar is being formed as a function of other Scalar objects
            self._poly = poly
            self.entity_name = None
        elif entity is not None:
            self.entity_name = entity.name if isinstance(entity, Entity) else entity
            self.scalar_name = self.name + "_" + self.entity_name
            self._poly = symbols(self.scalar_name)
            scalar_name2obj[self.scalar_name] = self
        else:
            raise Exception("Poly or entity must be not None")

    @property
    def poly(self) -> Symbol:
        return self._poly

    @property
    def value(self) -> float:
        if self._value is not None:
            return self._value
        index2symbol, symbol2index = create_lookup_tables_for_symbol(self.poly)
        run_specific_args = self.create_run_specific_args(f=self.poly, symbol2index=symbol2index)
        inputs = [scalar_name2obj[sym]._value for sym in index2symbol]
        return float(run_specific_args(inputs))

    @property
    def min_val(self) -> Optional[float]:
        return self._min_val

    @property
    def max_val(self) -> Optional[float]:
        return self._max_val

    def __rmul__(self, other: "Scalar") -> "Scalar":
        return self * other

    def __mul__(self, other: "Scalar") -> "Scalar":

        if hasattr(other, "poly"):
            result_poly = self.poly * other.poly
        else:
            result_poly = self.poly * other

        result = Scalar(value=None, poly=result_poly)
        return result

    def __radd__(self, other: "Scalar") -> "Scalar":
        return self + other

    def __add__(self, other: "Scalar") -> "Scalar":

        if hasattr(other, "poly"):
            result_poly = self.poly + other.poly
        else:
            result_poly = self.poly + other

        result = Scalar(value=None, poly=result_poly)
        return result

    def __sub__(self, other: "Scalar") -> "Scalar":
        result_poly = self.poly - other.poly
        result = Scalar(value=None, poly=result_poly)
        return result

    def __str__(self) -> str:
        return str(self.poly) + "=" + str(self.value)

    def __repr__(self) -> str:
        return str(self)

    @property
    def sens(self) -> Optional[float]:
        if self.min_val and self.max_val:
            return self.max_val - self.min_val
        return None

    # def neg_deriv_wrt_input(self, input_name: str) -> Symbol:
    #     obj = scalar_name2obj[input_name]
    #     derivative = -diff(self.poly, obj.poly)
    #     return derivative

    def create_run_specific_args(
        self, f: Symbol, symbol2index:dict
    ) -> Tuple[Callable, List[str], Dict[str, int]]:

        # Tudor: Here you weren't using *params
        # Tudor: If I understand correctly, .subs returns
        def _run_specific_args(tuple_of_args: tuple) -> Any:
            kwargs = {sym: tuple_of_args[i] for sym, i in symbol2index.items()}
            return f.subs(kwargs)

        return _run_specific_args

    def get_mechanism_for_entity(
        self, entity_name: str, input_symbols: List["Scalar"], sigma: float = 0.1
    ) -> iDPGaussianMechanism:
        print(entity_name)

        print(input_symbols)
        print()
        # Step 1: create derivative function wrt all inputs
        all_input_polys = [x.poly for x in input_symbols]
        print(all_input_polys)
        print()
        print(self.poly)
        print()
        # NOTE: self.poly is the L2 norm... but since it's only one variable we don't have to
        # do anything to it (the l2 norm of a variable is itself). Also we negate the derivative
        # because the search algorithms we use find the global minimum not maximum.
        output_deriv_wrt_all_inputs = -diff(self.poly, *all_input_polys)
        print(output_deriv_wrt_all_inputs)
        print()
        # Step 2: fix all inputs which come from entity_name
        output_deriv_wrt_all_inputs.subs({sym.poly: sym.value for sym in input_symbols})
        print(output_deriv_wrt_all_inputs)
        print()

        # Step 3: wrap in a function so that scipy optimizer can use it
        index2symbol, symbol2index = create_lookup_tables_for_symbol(output_deriv_wrt_all_inputs)
        search_lib_compatible_output_deriv = self.create_run_specific_args(
            f=output_deriv_wrt_all_inputs,
            symbol2index=symbol2index
        )

        # Step 4: Determine search ranges (also check to see if we can use discrete search)
        discrete_search = True
        rranges = list()
        for i, sym in enumerate(index2symbol):
            obj = scalar_name2obj[sym]
            if not obj.is_discrete:
                discrete_search = False
            rranges.append((obj.min_val, obj.max_val))

        # Step 5: Search over all possible inputs to find the maximum derivative
        L = search(search_lib_compatible_output_deriv, tuple(rranges), binary=discrete_search)

        # Step 6: Return gaussian mechanism object
        gm1 = iDPGaussianMechanism(
            sigma=sigma,
            value=self.value,
            L=L,
            entity=entity_name,
            name="gm_" + self.symbol_name,
        )
        return gm1



    def get_mechanism_for_symbol(
        self, symbol_name: str = "b", sigma: float = 0.1
    ) -> iDPGaussianMechanism:

        symbol = scalar_name2obj[symbol_name]

        # Step 1: get derivative we want to maximize
        z = self.neg_deriv(symbol_name)

        # Step 2: get lookup tables
        index2symbol, symbol2index = create_lookup_tables_for_symbol(self.poly)

        # # Step 3: substitute out all variables with the same entity as symbol
        # for i, symbol_name in enumerate(index2symbol):
        #     sym = scalar_name2obj[symbol_name]
        #     if sym.entity == symbol.entity:
        #         z = z.subs(sym.poly, sym.value)
        #     # else:
        #     #     print("Entity Mismatch:" + str(sym.split("_")[1]) + " != " + str(symbol.entity.name))

        # index2symbol, symbol2index = create_lookup_tables_for_symbol(z)

        run_specific_args = self.create_run_specific_args(
            f=z,
            symbol2index=symbol2index
        )

        discrete_search = True

        rranges = list()
        for i, sym in enumerate(index2symbol):
            obj = scalar_name2obj[sym]
            if not obj.is_discrete:
                discrete_search = False
            rranges.append((obj.min_val, obj.max_val))

        if not symbol.is_discrete:
            discrete_search = False

        # Step 3: maximize the derivative over a bounded range of <entity_name>
        L = search(run_specific_args, tuple(rranges), binary=discrete_search)

        # print("New Mechanism(sigma=" + str(sigma) + " value=" + str(symbol.value) + "L=" + str(L) + ")")
        # Step 4: create the gaussian mechanism object
        gm1 = iDPGaussianMechanism(
            sigma=sigma,
            value=symbol.value,
            L=L,
            entity=symbol_name.split("_")[1],
            name="gm_" + symbol_name,
        )

        return gm1

    def get_all_entity_mechanisms(
        self, sigma: float = 0.1
    ) -> Dict[str, List[iDPGaussianMechanism]]:
        sy_names = self.poly.free_symbols

        entity2symbols = defaultdict(list)
        for sy_name in sy_names:
            symbol = scalar_name2obj[str(sy_name)]
            entity_name = symbol.entity.name

            if entity_name not in entity2symbols:
                entity2symbols[entity_name] = list()

            entity2symbols[entity_name].append(symbol)

        entity2mechanisms = defaultdict(list)
        for entity_name, symbols in entity2symbols.items():
            entity2mechanisms[entity_name] = [self.get_mechanism_for_entity(entity_name=entity_name, input_symbols=symbols, sigma=sigma)]

        # for sy_name in sy_names:
        #     mechanism = self.get_mechanism_for_symbol(symbol_name=str(sy_name), sigma=sigma)
        #     split_name = str(sy_name).split("_")
        #     entity_name = split_name[1]
        #     entity2mechanisms[entity_name].append(mechanism)

        return entity2mechanisms

    @property
    def entities(self) -> Set[str]:
        return {str(sy_name).split("_")[1] for sy_name in self.poly.free_symbols}

    # Tudor: remove the typing from Any to an accountant type
    def publish(self, acc: Any, sigma: float = 1.5) -> float:
        acc_original = acc

        assert sigma > 0

        acc_temp = deepcopy(acc_original)

        # get mechanisms for new publish event
        ms = self.get_all_entity_mechanisms(sigma=sigma)
        acc_temp.append(ms)

        overbudgeted_entities = acc_temp.overbudgeted_entities

        sample = random.gauss(0, sigma)

        while len(overbudgeted_entities) > 0:
            for sy_name in self.poly.free_symbols:
                entity_name = str(sy_name).split("_")[1]
                if entity_name in overbudgeted_entities:
                    sym = scalar_name2obj[str(sy_name)]
                    self._poly = self.poly.subs(sym.poly, 0)

            acc_temp = deepcopy(acc_original)

            # get mechanisms for new publish event
            ms = self.get_all_entity_mechanisms(sigma=sigma)
            acc_temp.append(ms)

            overbudgeted_entities = acc_temp.overbudgeted_entities

        output = self.value + sample

        acc_original.entity2ledger = deepcopy(acc_temp.entity2ledger)

        return output

    def _object2proto(self) -> Scalar_PB:
        return Scalar_PB(
            id=self.id._object2proto(),
            has_name=True if self.name is not None else False,
            name=self.name if self.name is not None else "",
            has_value=True if self._value is not None else False,
            value=self._value if self._value is not None else 0,
            has_min_val=True if self._min_val is not None else False,
            min_val=self._min_val if self._min_val is not None else 0,
            has_max_val=True if self._max_val is not None else False,
            max_val=self._max_val if self._max_val is not None else 0,
            # has_poly=True if self._poly is not None else False,
            # poly=self._poly if self._poly is not None else None,
            has_entity_name=True if self.entity_name is not None else False,
            entity_name=self.entity_name if self.entity_name is not None else "",
        )

    @staticmethod
    def _proto2object(proto: Scalar_PB) -> "Scalar":
        name: Optional[str] = None
        if proto.has_name:
            name = proto.name

        value: Optional[float] = None
        if proto.has_value:
            value = proto.value

        min_val: Optional[float] = None
        if proto.has_min_val:
            min_val = proto.min_val

        max_val: Optional[float] = None
        if proto.has_max_val:
            max_val = proto.max_val

        entity_name: Optional[str] = None
        if proto.has_entity_name:
            entity_name = proto.entity_name

        return Scalar(
            id=UID._proto2object(proto.id),
            name=name,
            value=value,
            min_val=min_val,
            max_val=max_val,
            entity=entity_name,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Scalar_PB
