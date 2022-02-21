# stdlib
from typing import Any
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple

# third party
import numpy as np
from scipy import optimize
from sympy.core.basic import Basic as BasicSymbol

# relative
from ...common import UID
from ..entity import Entity
from ..search import create_lookup_tables_for_symbol
from ..search import create_searchable_function_from_polynomial
from ..search import max_lipschitz_via_jacobian
from ..search import minimize_function
from ..search import ssid2obj
from .abstract.intermediate_scalar import IntermediateScalar
from .abstract.scalar import Scalar


class IntermediateGammaScalar(IntermediateScalar):
    """
    A Superclass for Scalars with data from multiple entities (GammaScalars).
    Most importantly, this is where all of the operations (+/-/*/div) are implemented,
    as well as the various methods with which to perform the search for the max Lipschitz.
    """

    def __init__(
        self,
        poly: BasicSymbol,
        min_val: float,
        max_val: float,
        id: Optional[UID] = None,
    ) -> None:
        super().__init__(poly=poly, id=id)
        self._min_val = float(min_val)
        self._max_val = float(max_val)
        self.is_linear: Optional[
            bool
        ] = None  # None means skip performance optimization

    # GammaScalar +/-/*/div other ---> GammaScalar
    def __add__(self, other: Any) -> IntermediateScalar:
        if isinstance(other, Scalar):
            # relative
            from .intermediate_phi_scalar import IntermediatePhiScalar

            if isinstance(other, IntermediatePhiScalar):
                other = other.gamma
            return IntermediateGammaScalar(
                poly=self.poly + other.poly,
                min_val=self.min_val + other.min_val,
                max_val=self.max_val + other.max_val,
            )
        return IntermediateGammaScalar(
            poly=self.poly + other,
            min_val=self.min_val + other,
            max_val=self.max_val + other,
        )

    def __sub__(self, other: Any) -> IntermediateScalar:
        if isinstance(other, Scalar):
            # relative
            from .intermediate_phi_scalar import IntermediatePhiScalar

            if isinstance(other, IntermediatePhiScalar):
                other = other.gamma
            return IntermediateGammaScalar(
                poly=self.poly - other.poly,
                min_val=self.min_val - other.min_val,
                max_val=self.max_val - other.max_val,
            )
        return IntermediateGammaScalar(
            poly=self.poly - other,
            min_val=self.min_val - other,
            max_val=self.max_val - other,
        )

    def __mul__(self, other: Any) -> IntermediateScalar:
        if isinstance(other, Scalar):
            # relative
            from .intermediate_phi_scalar import IntermediatePhiScalar

            if isinstance(other, IntermediatePhiScalar):
                other = other.gamma

            max_val = max(
                self.min_val * other.min_val,
                self.min_val * other.max_val,
                self.max_val * other.min_val,
                self.max_val * other.max_val,
            )

            min_val = min(
                self.min_val * other.min_val,
                self.min_val * other.max_val,
                self.max_val * other.min_val,
            )

            return IntermediateGammaScalar(
                poly=self.poly * other.poly, max_val=max_val, min_val=min_val
            )

        max_val = max(
            self.min_val * other,
            self.max_val * other,
        )

        min_val = min(
            self.min_val * other,
            self.max_val * other,
        )

        return IntermediateGammaScalar(
            poly=self.poly * other, min_val=min_val, max_val=max_val
        )

    def max_lipschitz_via_explicit_search(
        self, force_all_searches: bool = False
    ) -> TypeTuple[TypeList[optimize.OptimizeResult], np.float64]:

        r1 = np.array([x.poly for x in self.input_scalars])  # type: ignore

        # relative
        from .gamma_scalar import GammaScalar

        r2_diffs = np.array(
            [
                GammaScalar(x.min_val, x.value, x.max_val, entity=x.entity, prime=x.prime).poly  # type: ignore
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
            # there's a small bit of rounding error from this constraint - this should
            # only be used as a double check or as a backup!!!
            return out**0.5 - 1 / 2**16

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
        )  # type: ignore

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
