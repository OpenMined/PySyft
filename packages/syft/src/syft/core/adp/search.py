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
    constraints: Optional[TypeList[TypeDict[str, Any]]] = None,
    force_all_searches: bool = False,
) -> TypeList[optimize.OptimizeResult]:
    constraints = constraints if constraints is not None else []

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

    if not shgo_results.success:
        raise Exception("Search algorithm wasn't solvable... abort")

    return results


def max_lipschitz_wrt_entity(scalars: Any, entity: Entity) -> float:
    # if all scalars have is_linear = True, we will skip the search below
    can_skip = True
    for i in scalars:
        if getattr(i, "is_linear", None) is not True:
            can_skip = False
            break
    if can_skip:
        return 1.0

    result: Union[float, optimize.OptimizeResult] = max_lipschitz_via_jacobian(
        scalars, input_entity=entity
    )[0][-1]
    if not isinstance(result, float):
        return -float(result.fun)
    else:
        return -result


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

    # WARNING: in the Feldman paper (https://arxiv.org/pdf/2008.11193.pdf) in
    # Example 2.8, which is the basis for this logic, the paper says the following,
    # "Suppose that g : (Rd)n → Rd` is Li-Lipschitz in coordinate i (in L2-norm)."
    # This is a somewhat ambiguous statement. After much thought, we are confident
    # that 'i' refers to the index of a specific entity, but it's unclear exactly
    # what is being L2-normed. There are two plausible options.

    # - Option A: each output value is L2 normed before the derivative is taken, meaning
    # that Li-Lipschitz is a bound the absolute value of the derivative of g wrt each input
    # from entity i. Note that this means that we took a vector of polynomials, L2normed htem,
    # took the derivative, and then the Li-lipschiktz bound is wrt each of these derivatives.
    # This seems a bit odd... because why L2 norm something when Lipschitz is going to calculate
    # the absolute value anyway (it's an identical operation... although absolute value isn't
    # technically continuous and differentiable in all places, so it could be that this is
    # express as L2 norm because that's the mathematically elegant way to do it. Also it makes more
    # sense why it would be a paranthetical given that it's just a small explanation as to why
    # the derivaitve of the L-Lipschitz bound can be taken). Perhaps this paraenthetical was
    # just a throwaway.

    # - Option B: more likely, because the Li-lipschitz constraint is referring to a vector
    # of derivatives corresponding to the input->output relationship between a single output
    # scalar and all of the inputs from that entity, the L2 norm is being used to compress
    # all of these derivatives into one scalar, which is bounded. That is to say, Option A
    # is bound on the derivative of the L2 norm of each value, whereas Option B is a bound
    # on the L2 norm of the derivatives, which is a scalar. We think that it's the latter
    # because it makes sense why the L2 norm is being applied (not redundant). The bigger reason
    # why Option B makes more sense is that with option A, if we publish an out value with
    # multiple inputs coming from one entity, the privacy budget is only scaled WRT the sharpest
    # derivative of one input as opposed to the combined effect of all of them. This follows
    # the differential privacy intuition that we care about the overall effect of removing
    # all inputs toa  function from entity, not just the max effect of any one of htose inputs.
    # Thus, Option B *should* be the right answer, and if it's not, then we probably end
    # up writing a paper for why Vitaly Feldman's paper is broken.

    # ON THE CONTRARY: in Example 2.8, the L term is just supposed to be a worst case scalar
    # on the function, which gets multipled by a (separate) L2 norm of the underlying data
    # values, so it makes sense that we wouldn't scale by the combined factor of all inputs, but
    # instead scale by the "worst case relationship" between any input and any output.

    # ON THE CONTRARY (Again): L_i is supposed to be the maximum slope WRT all contributions
    # from this entity, meaning the worst case bound of removing all inputs to this function.
    # thus it follows that the combined inputs could be worse than any particular input?...

    # CONSERVATIVE VIEW (and interim conclusion): Option B will always be higher than option A,
    # because after ensuring
    # that everything is positive value (by squaring it), Option A calculates an max across values
    # whereas option B sums them. This means that Option B will always generate at least the same
    # if not higher value than Option A, so given the uncertainty we're going with Option B.
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
