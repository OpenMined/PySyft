# stdlib
from enum import Enum

# third party
import jax
import jax.numpy as jnp


class GAMMA_TENSOR_OP(Enum):
    # Numpy ArrayLike
    ABS = "abs"
    ADD = "add"
    ALL = "all"
    ANY = "any"
    ARGMAX = "argmax"
    ARGMIN = "argmin"
    ARGSORT = "argsort"
    BITWISE_AND = "bitwise_and"
    BITWISE_OR = "bitwise_or"
    BITWISE_XOR = "bitwise_xor"
    CHOOSE = "choose"
    CLIP = "clip"
    COMPRESS = "compress"
    COPY = "copy"
    CUMPROD = "cumprod"
    CUMSUM = "cumsum"
    DIAGONAL = "diagonal"
    DIVMOD = "divmod"
    DOT = "dot"
    EQUAL = "equal"
    EXP = "exp"
    FLOOR_DIVIDE = "floor_divide"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    LOG = "log"
    LSHIFT = "left_shift"
    MATMUL = "matmul"
    MAX = "max"
    MEAN = "mean"
    MIN = "min"
    MOD = "mod"
    MULTIPLY = "multiply"
    NEGATIVE = "negative"
    NONZERO = "nonzero"
    NOOP = "noop"
    NOT_EQUAL = "not_equal"
    ONES_LIKE = "ones_like"
    POSITIVE = "positive"
    POWER = "power"
    PROD = "prod"
    PTP = "ptp"
    PUT = "put"
    RAVEL = "ravel"
    REPEAT = "repeat"
    RESHAPE = "reshape"
    RESIZE = "resize"
    RMATMUL = "rmatmul"
    ROUND = "round"
    RSHIFT = "right_shift"
    SORT = "sort"
    SQRT = "sqrt"
    SQUEEZE = "squeeze"
    STD = "std"
    SUBTRACT = "subtract"
    SUM = "sum"
    SWAPAXES = "swapaxes"
    TAKE = "take"
    TRACE = "trace"
    TRANSPOSE = "transpose"
    TRUE_DIVIDE = "true_divide"
    VAR = "var"
    ZEROS_LIKE = "zeros_like"
    # Our Methods
    RECIPROCAL = "reciprocal"
    FLATTEN = "flatten"
    PY_GETITEM = "py_getitem"


GAMMA_TENSOR_OP_FUNC = {
    GAMMA_TENSOR_OP.ABS: jnp.abs,
    GAMMA_TENSOR_OP.ADD: jnp.add,
    GAMMA_TENSOR_OP.ALL: jnp.all,
    GAMMA_TENSOR_OP.ANY: jnp.any,
    GAMMA_TENSOR_OP.ARGMAX: jnp.argmax,
    GAMMA_TENSOR_OP.ARGMIN: jnp.argmin,
    GAMMA_TENSOR_OP.ARGSORT: jnp.argsort,
    GAMMA_TENSOR_OP.BITWISE_AND: jnp.bitwise_and,
    GAMMA_TENSOR_OP.BITWISE_OR: jnp.bitwise_or,
    GAMMA_TENSOR_OP.BITWISE_XOR: jnp.bitwise_xor,
    GAMMA_TENSOR_OP.CHOOSE: jnp.choose,
    GAMMA_TENSOR_OP.CLIP: jnp.clip,
    GAMMA_TENSOR_OP.COMPRESS: lambda x, y, axis: jnp.compress(y, x, axis=axis),
    GAMMA_TENSOR_OP.COPY: jnp.copy,
    GAMMA_TENSOR_OP.CUMPROD: jnp.cumprod,
    GAMMA_TENSOR_OP.CUMSUM: jnp.cumsum,
    GAMMA_TENSOR_OP.DIAGONAL: jnp.diag,
    GAMMA_TENSOR_OP.DIVMOD: jnp.divmod,
    GAMMA_TENSOR_OP.DOT: jnp.dot,
    GAMMA_TENSOR_OP.EQUAL: jnp.equal,
    GAMMA_TENSOR_OP.FLOOR_DIVIDE: jnp.floor_divide,
    GAMMA_TENSOR_OP.GREATER_EQUAL: jax.lax.ge,
    GAMMA_TENSOR_OP.GREATER: jax.lax.gt,
    GAMMA_TENSOR_OP.LESS_EQUAL: jax.lax.le,
    GAMMA_TENSOR_OP.LESS: jax.lax.lt,
    GAMMA_TENSOR_OP.LOG: jnp.log,
    GAMMA_TENSOR_OP.LSHIFT: jnp.left_shift,
    GAMMA_TENSOR_OP.MATMUL: jnp.matmul,
    GAMMA_TENSOR_OP.MAX: jnp.max,
    GAMMA_TENSOR_OP.MEAN: jnp.mean,
    GAMMA_TENSOR_OP.MIN: jnp.min,
    GAMMA_TENSOR_OP.MOD: jnp.mod,
    GAMMA_TENSOR_OP.MULTIPLY: jnp.multiply,
    GAMMA_TENSOR_OP.NEGATIVE: jnp.negative,
    GAMMA_TENSOR_OP.NONZERO: jnp.nonzero,
    GAMMA_TENSOR_OP.NOOP: lambda x: x,
    GAMMA_TENSOR_OP.NOT_EQUAL: jax.lax.ne,
    GAMMA_TENSOR_OP.ONES_LIKE: jnp.ones_like,
    GAMMA_TENSOR_OP.POSITIVE: jnp.positive,
    GAMMA_TENSOR_OP.POWER: jnp.power,
    GAMMA_TENSOR_OP.PROD: jnp.prod,
    GAMMA_TENSOR_OP.PTP: jnp.ptp,
    GAMMA_TENSOR_OP.PUT: jnp.put,
    GAMMA_TENSOR_OP.RAVEL: jnp.ravel,
    GAMMA_TENSOR_OP.REPEAT: jnp.repeat,
    GAMMA_TENSOR_OP.RESHAPE: jnp.reshape,
    GAMMA_TENSOR_OP.RESIZE: jnp.resize,
    GAMMA_TENSOR_OP.RMATMUL: lambda x, y: jnp.matmul(y, x),
    GAMMA_TENSOR_OP.ROUND: jnp.round,
    GAMMA_TENSOR_OP.RSHIFT: jnp.right_shift,
    GAMMA_TENSOR_OP.SORT: jnp.sort,
    GAMMA_TENSOR_OP.SQRT: jnp.sqrt,
    GAMMA_TENSOR_OP.SQUEEZE: jnp.squeeze,
    GAMMA_TENSOR_OP.STD: jnp.std,
    GAMMA_TENSOR_OP.SUM: jnp.sum,
    GAMMA_TENSOR_OP.SWAPAXES: jnp.swapaxes,
    GAMMA_TENSOR_OP.SUBTRACT: jnp.subtract,
    GAMMA_TENSOR_OP.TAKE: jnp.take,
    GAMMA_TENSOR_OP.TRACE: jnp.trace,
    GAMMA_TENSOR_OP.TRANSPOSE: jnp.transpose,
    GAMMA_TENSOR_OP.TRUE_DIVIDE: jnp.true_divide,
    GAMMA_TENSOR_OP.VAR: jnp.var,
    GAMMA_TENSOR_OP.ZEROS_LIKE: jnp.zeros_like,
    # GAMMA_TENSOR_OP.FLATTEN: jnp.flatten,
    GAMMA_TENSOR_OP.PY_GETITEM: lambda x, y: x.__getitem__(y),
}
