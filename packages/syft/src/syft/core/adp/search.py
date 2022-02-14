# CLEANUP NOTES:
# - remove unused comments
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)

# stdlib
from functools import lru_cache
from typing import Any
from typing import Callable
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from typing import Set
from typing import Tuple as TypeTuple
from typing import Union

# third party
import numpy as np
from pymbolic.mapper import WalkMapper
from pymbolic.mapper.evaluator import EvaluationMapper as EM
from pymbolic.primitives import Variable
from scipy import optimize
import sympy as sym
from sympy.core.basic import Basic
from sympy.solvers import solve

# relative
from .entity import Entity

# Leaving this commented out here because I'm pretty sure we can get the
# lru_cache to be WAY faster through this approach but I can't seem to
# get it to work (it adds about 10% perf loss).
# ordered_symbols = list()
# for i in range(100):
#     ordered_symbols.append(symbols("s"+str(i)))

# ssid2obj is used to lookup value, min_val and max_val. if we are able to store these
# in the name string instead we could extract them at the point of use instead of lookup
# TypeDict[str, Union[PhiScalar, GammaScalar]]
ssid2obj: TypeDict[str, Any] = {}  # TODO: Fix types in circular deps


class GetSymbolsMapper(WalkMapper):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.free_symbols: Set[Any] = set()

    def visit(self, *args: Any, **kwargs: Any) -> bool:
        for arg in args:
            if isinstance(arg, Variable):
                self.free_symbols.add(arg)
        return True


def create_searchable_function_from_polynomial(
    poly: Basic, symbol2index: dict
) -> Callable:
    """Wrap polynomial execution logic in a function usable for scipy.optimize

    scipy.optimize functions expect an ordered list of args as input whereas
    sympy passes dictionaries into a .subs() function. This method just
    wraps sympy's approach in a method which accepts a tuple of args ordered
    according to symbol2index lookup table
    """
    if "pymbolic" in str(type(poly)):

        def _run_specific_args(tuple_of_args: TypeTuple) -> EM:
            kwargs = {sym: tuple_of_args[i] for sym, i in symbol2index.items()}
            output = EM(context=kwargs)(poly)
            return output

    else:

        def _run_specific_args(
            tuple_of_args: TypeTuple,
        ) -> EM:
            kwargs = {sym: tuple_of_args[i] for sym, i in symbol2index.items()}
            output = poly.subs(kwargs)

            return output

    return _run_specific_args


@lru_cache(maxsize=None)
def minimize_poly(
    poly: Basic, *rranges: TypeTuple, force_all_searches: bool = False, **s2i: TypeDict
) -> TypeList[optimize.OptimizeResult]:
    """Minimizes a polynomial using types basic enough for lru_cache"""

    # convert polynomial to function object with API necessary for scipy optimizer
    search_fun = create_searchable_function_from_polynomial(poly, s2i)

    # search over rranges for the set of inputs which minimizes function "f"
    return minimize_function(f=search_fun, rranges=rranges, force_all_searches=False)


def flatten_and_maximize_sympoly(
    poly: Basic, force_all_searches: bool = False
) -> TypeList[optimize.OptimizeResult]:

    i2s = list(poly.free_symbols)
    s2i = {s: i for i, s in enumerate(i2s)}

    # this code seems to make things slower - although there might be a memory
    # improvement (i haven't checked)
    # flattened_poly = poly.copy().subs({k:v for k,v in zip(i2s, ordered_symbols[0:len(i2s)])})
    # flattened_s2i = {str(ordered_symbols[i]):i for s,i in s2i.items()}

    flattened_poly = poly
    flattened_s2i = {str(s): i for s, i in s2i.items()}

    rranges = [
        (ssid2obj[i2s[i].name].min_val, ssid2obj[i2s[i].name].max_val)
        for i in range(len(s2i))
    ]

    return minimize_poly(
        flattened_poly, *rranges, force_all_searches=force_all_searches, **flattened_s2i
    )


def flatten_and_maximize_poly(
    poly: Any, force_all_searches: bool = False
) -> TypeList[optimize.OptimizeResult]:

    mapper = GetSymbolsMapper()
    mapper(poly)

    i2s = list(mapper.free_symbols)
    s2i = {s: i for i, s in enumerate(i2s)}

    # this code seems to make things slower - although there might be a memory
    # improvement (i haven't checked)
    # flattened_poly = poly.copy().subs({k:v for k,v in zip(i2s, ordered_symbols[0:len(i2s)])})
    # flattened_s2i = {str(ordered_symbols[i]):i for s,i in s2i.items()}

    flattened_poly = poly
    flattened_s2i = {str(s): i for s, i in s2i.items()}

    rranges = [
        (ssid2obj[i2s[i].name].min_val, ssid2obj[i2s[i].name].max_val)
        for i in range(len(s2i))
    ]

    return minimize_poly(
        flattened_poly, *rranges, force_all_searches=force_all_searches, **flattened_s2i
    )


def create_lookup_tables_for_symbol(
    polynomial: Any,
) -> TypeTuple[TypeList[str], TypeDict[str, int]]:

    mapper = GetSymbolsMapper()
    mapper(polynomial)

    index2symbol = [str(x) for x in mapper.free_symbols]
    symbol2index = {sym: i for i, sym in enumerate(index2symbol)}

    return index2symbol, symbol2index


def minimize_function(
    f: Any,
    rranges: Any,
    constraints: TypeList[TypeDict[str, Any]] = [],
    force_all_searches: bool = False,
) -> TypeList[optimize.OptimizeResult]:
    results = list()

    # Step 1: try simplicial
    shgo_results = optimize.shgo(
        f, rranges, sampling_method="simplicial", constraints=constraints
    )
    results.append(shgo_results)

    if not shgo_results.success or force_all_searches:
        # sometimes simplicial has trouble as a result of initialization
        # see: https://github.com/scipy/scipy/issues/10429 for details
        shgo_results = optimize.shgo(
            f, rranges, sampling_method="sobol", constraints=constraints
        )
        results.append(shgo_results)

    # if not shgo_results.success:
    #     raise Exception("Search algorithm wasn't solvable... abort")
    print("Search algorithm wasn't solvable... abort")

    return results


def max_lipschitz_wrt_entity(scalars: Any, entity: Entity) -> float:
    result: Union[float, optimize.OptimizeResult] = max_lipschitz_via_jacobian(
        scalars, input_entity=entity
    )[0][-1]
    if not isinstance(result, float):
        return -float(result.fun)
    else:
        return -result

    # if isinstance(result, float):
    #     return -result
    # else:
    #     return -float(result.fun)


def max_lipschitz_via_jacobian(
    scalars: TypeList[Any],  # TODO: Fix Scalar type circular
    input_entity: Optional[Entity] = None,
    data_dependent: bool = True,
    force_all_searches: bool = False,
    try_hessian_shortcut: bool = False,
) -> TypeTuple[TypeList[float], Any]:
    # scalars = R^d` representing the d' dimensional output of g
    # input_entity = the 'i'th entity for which we want to compute a lipschitz bound

    input_scalars = set()
    for s in scalars:
        for i_s in s.input_scalars:
            input_scalars.add(i_s)

    # R^d` representing the d' dimensional output of g
    # the numberator of the partial derivative
    out = sym.Matrix([x.sympoly for x in scalars])

    if input_entity is None:
        # X_1 through X_n flattened into a single vector
        j = out.jacobian([x.sympoly for x in input_scalars])
    else:

        # In general it doesn't make sense to consider the max partial derivative over
        # all inputs because we don't want the Lipschitz bound of the entire jacobian,
        # we want the Lipschitz bound with respect to entity "i" (see Individual Privacy
        # Accounting via a Renyi Filter: https://arxiv.org/abs/2008.11193). For example,
        # if I had a polynomial y = a + b**2 + c**3 + d**4 where each a,b,c,d variable
        # was from a different entity, the fact that d has a big derivative should
        # change the max Lipschitz bound of y with respect to "a". Thus, we're only
        # interested in searching for the maximum partial derivative with respect to the
        # variables from the focus entity "i".

        # And if we're looking to compute the max parital derivative with respect to
        # input scalars from only one entity, then we select only the variables
        # corresponding to that entity here.

        # X_i scalars
        relevant_scalars = list(
            filter(lambda s: s.entity == input_entity, input_scalars)
        )
        relevant_inputs = [x.sympoly for x in relevant_scalars]

        # jacobian ONLY with respect to X_i
        j = out.jacobian(relevant_inputs)

        # For higher order functions - it's possible that some of the partial
        # derivatives are conditioned on data from the input entity. The philosophy of
        # input DP is that when producing an epsilon guarantee for entity[i] that we
        # don't need to search over the possible range of data for that entity but can
        # instead use the data itself - this results in an epsilon for each entity which
        # is private but it also means the bound is tighter. So we could skip this step
        # but it would in some cases make the bound looser than it needs to be.
        if data_dependent:
            j = j.subs({x.sympoly: x.value for x in relevant_scalars})

    neg_l2_j = -((np.sum(np.square(j))) ** 0.5)

    # if the L2 norm is a constant (flat derivative) - just return it
    if len(np.sum(j).free_symbols) == 0:
        result = -float(np.max(j))
        return [result], neg_l2_j

    if try_hessian_shortcut:
        h = j.jacobian([x.sympoly for x in input_scalars])
        if (
            len(solve(np.sum(h**2), *[x.sympoly for x in input_scalars], dict=True))
            == 0
        ):
            print(
                "Gradient is linear - solve with brute force search over edges of domain"
            )

            i2s, s2i = create_lookup_tables_for_symbol(neg_l2_j)
            search_fun = create_searchable_function_from_polynomial(
                poly=neg_l2_j, symbol2index=s2i
            )

            constant = 0.000001
            rranges = [
                (x.min_val, x.max_val, x.max_val - x.min_val) for x in input_scalars
            ]
            skewed_results = optimize.brute(
                search_fun, rranges, finish=None, full_output=False
            )
            result_inputs = skewed_results + constant
            result_output = search_fun(result_inputs)
            return [float(result_output)], neg_l2_j

    return flatten_and_maximize_sympoly(neg_l2_j), neg_l2_j
