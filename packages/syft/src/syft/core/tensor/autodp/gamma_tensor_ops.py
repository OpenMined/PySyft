# stdlib
from enum import Enum
import operator

# third party
import jax.numpy as jnp


class GAMMA_TENSOR_OP(Enum):
    # Numpy ArrayLike
    ABS = "abs"
    ADD = "add"
    ALL = "all"
    ANY = "any"
    ARGMAX = "argmax"
    ARGMhi = "argmin"
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
    FLATTEN_C = "flatten_c"
    FLATTEN_A = "flatten_a"
    FLATTEN_F = "flatten_f"
    FLATTEN_K = "flatten_k"


GAMMA_TENSOR_OP_FUNC = {
    GAMMA_TENSOR_OP.ADD: operator.add,
    GAMMA_TENSOR_OP.BITWISE_AND: operator.and_,
    GAMMA_TENSOR_OP.BITWISE_OR: operator.or_,
    GAMMA_TENSOR_OP.BITWISE_XOR: operator.xor,
    GAMMA_TENSOR_OP.DOT: jnp.dot,
    GAMMA_TENSOR_OP.EQUAL: operator.eq,
    GAMMA_TENSOR_OP.FLOOR_DIVIDE: operator.floordiv,
    GAMMA_TENSOR_OP.GREATER_EQUAL: operator.ge,
    GAMMA_TENSOR_OP.GREATER: operator.gt,
    GAMMA_TENSOR_OP.LESS_EQUAL: operator.le,
    GAMMA_TENSOR_OP.LESS: operator.lt,
    GAMMA_TENSOR_OP.LSHIFT: operator.lshift,
    GAMMA_TENSOR_OP.MATMUL: operator.matmul,
    GAMMA_TENSOR_OP.MOD: operator.mod,
    GAMMA_TENSOR_OP.MULTIPLY: operator.mul,
    GAMMA_TENSOR_OP.NOT_EQUAL: operator.ne,
    GAMMA_TENSOR_OP.RSHIFT: operator.rshift,
    GAMMA_TENSOR_OP.SUBTRACT: operator.sub,
    GAMMA_TENSOR_OP.TRUE_DIVIDE: operator.truediv,
}
