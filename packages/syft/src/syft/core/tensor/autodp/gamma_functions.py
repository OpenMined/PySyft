# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union
from typing import ValuesView

# third party
import jax
from jax import numpy as jnp
import numpy as np

# relative
from .gamma_tensor_ops import GAMMA_TENSOR_OP


def generate_ops() -> Dict[GAMMA_TENSOR_OP, Callable]:
    # third party
    from numpy.typing import ArrayLike

    # relative
    from . import gamma_tensor as GT
    from ..passthrough import AcceptableSimpleType  # type: ignore

    JaxableType = Union[GT.GammaTensor, ArrayLike, AcceptableSimpleType]

    def reconstruct_from_inputs(values: ValuesView[JaxableType]) -> List[JaxableType]:
        return [i.reconstruct() if isinstance(i, GT.GammaTensor) else i for i in values]

    def unpack_reconstruct_with_func(jnp_func: Callable) -> Callable:
        def wrapped_jnp(state: dict) -> jax.numpy.DeviceArray:
            return jnp_func(*reconstruct_from_inputs(state.values()))

        return wrapped_jnp

    def _reciprocal(state: dict) -> jax.numpy.DeviceArray:
        return jnp.divide(1, *reconstruct_from_inputs(state.values()))

    VALID_FLATTEN_TYPES = ["C", "F", "A", "K"]

    def get_flatten_type(order: str) -> Callable:
        if order.upper() not in VALID_FLATTEN_TYPES:
            raise Exception(f"Invalid flatten order. {order}")

        def flatten_op(value: JaxableType) -> Callable:
            # TODO: figure out why JAX jnp.ndarray.flatten does nothing?
            # perhaps replace with jnp.ravel
            return np.array(value).flatten(order=order)

        return flatten_op

    # given an infix operation with left and right
    # return a wrapper function which swaps the inputs before calling the original op
    # l_infix_op(100, 1) = 100 / 1
    # r_infix_op(100, 1) = 1 / 100
    def make_r_infix_op(l_infix_op: Callable) -> Callable:
        def r_infix_op(left: Any, right: Any) -> Any:
            return l_infix_op(right, left)

        return r_infix_op

    ops = {
        GAMMA_TENSOR_OP.NOOP: lambda x: x,
        GAMMA_TENSOR_OP.ADD: jnp.add,
        GAMMA_TENSOR_OP.SUBTRACT: jnp.subtract,
        GAMMA_TENSOR_OP.MULTIPLY: jnp.multiply,
        GAMMA_TENSOR_OP.TRUE_DIVIDE: jnp.true_divide,
        GAMMA_TENSOR_OP.MATMUL: jnp.matmul,
        GAMMA_TENSOR_OP.RMATMUL: make_r_infix_op(jnp.matmul),
        GAMMA_TENSOR_OP.GREATER: jnp.greater,
        GAMMA_TENSOR_OP.GREATER_EQUAL: jnp.greater_equal,
        GAMMA_TENSOR_OP.EQUAL: jnp.equal,
        GAMMA_TENSOR_OP.NOT_EQUAL: jnp.not_equal,
        GAMMA_TENSOR_OP.LESS: jnp.less,
        GAMMA_TENSOR_OP.LESS_EQUAL: jnp.less_equal,
        GAMMA_TENSOR_OP.EXP: jnp.exp,
        GAMMA_TENSOR_OP.LOG: jnp.log,
        GAMMA_TENSOR_OP.TRANSPOSE: jnp.transpose,
        GAMMA_TENSOR_OP.SUM: jnp.sum,
        GAMMA_TENSOR_OP.ONES_LIKE: jnp.ones_like,
        GAMMA_TENSOR_OP.ZEROS_LIKE: jnp.zeros_like,
        GAMMA_TENSOR_OP.RAVEL: jnp.ravel,
        GAMMA_TENSOR_OP.RESIZE: jnp.resize,
        GAMMA_TENSOR_OP.COMPRESS: jnp.compress,
        GAMMA_TENSOR_OP.SQUEEZE: jnp.squeeze,
        GAMMA_TENSOR_OP.ANY: jnp.any,
        GAMMA_TENSOR_OP.ALL: jnp.all,
        GAMMA_TENSOR_OP.LOGICAL_AND: jnp.logical_and,
        GAMMA_TENSOR_OP.LOGICAL_OR: jnp.logical_or,
        GAMMA_TENSOR_OP.POSITIVE: jnp.positive,
        GAMMA_TENSOR_OP.NEGATIVE: jnp.negative,
        GAMMA_TENSOR_OP.MEAN: jnp.mean,
        GAMMA_TENSOR_OP.STD: jnp.std,
        GAMMA_TENSOR_OP.DOT: jnp.dot,
        GAMMA_TENSOR_OP.SQRT: jnp.sqrt,
        GAMMA_TENSOR_OP.ABS: jnp.abs,
        GAMMA_TENSOR_OP.CLIP: jnp.clip,
        GAMMA_TENSOR_OP.RECIPROCAL: _reciprocal,
        GAMMA_TENSOR_OP.FLATTEN_C: get_flatten_type(order="C"),
        GAMMA_TENSOR_OP.FLATTEN_F: get_flatten_type(order="F"),
        GAMMA_TENSOR_OP.FLATTEN_A: get_flatten_type(order="A"),
        GAMMA_TENSOR_OP.FLATTEN_K: get_flatten_type(order="K"),
    }

    non_generic_funcs = [GAMMA_TENSOR_OP.NOOP, GAMMA_TENSOR_OP.RECIPROCAL]

    mapper = dict()

    for op, func in ops.items():
        if op in non_generic_funcs:
            mapper[op] = func
        mapper[op] = unpack_reconstruct_with_func(func)
    return mapper


GAMMA_FUNC_MAPPER = generate_ops()
