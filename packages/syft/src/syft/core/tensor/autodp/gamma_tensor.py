# future
from __future__ import annotations

# stdlib
from collections import deque
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

# from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union

# third party
import flax
import jax
from jax import numpy as jnp
import numpy as np

# from numpy.random import randint
from numpy.typing import ArrayLike

# from scipy import optimize
from scipy.optimize import shgo

# relative
from ...adp.data_subject_ledger import DataSubjectLedger

# from ...adp.data_subject_list import DataSubjectList
# from ...adp.data_subject_list import DataSubjectArray
# from ...adp.data_subject_list import dslarraytonumpyutf8
# from ...adp.data_subject_list import numpyutf8todslarray
from ...adp.vectorized_publish import publish
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ..fixed_precision_tensor import FixedPrecisionTensor
from ..passthrough import PassthroughTensor  # type: ignore
from .gamma_tensor_ops import GAMMA_TENSOR_OP
from .gamma_tensor_ops import GAMMA_TENSOR_OP_FUNC
from .jax_ops import SyftJaxInfixOp
from .jax_ops import SyftJaxOp
from .jax_ops import SyftJaxUnaryOp
from .tensor_wrapper_gamma_tensor_pointer import TensorWrappedGammaTensorPointer

if TYPE_CHECKING:
    # stdlib
    from dataclasses import dataclass
else:
    # third party
    from flax.struct import dataclass


INPLACE_OPS = {"resize", "sort"}


def debox_other(other: Any, attr: str) -> Any:
    if not isinstance(other, GammaTensor):
        return other
    return getattr(other, attr)


def debox_child(other: Any) -> Any:
    return debox_other(other, "child")


def debox_linear(other: Any) -> Any:
    if not isinstance(other, GammaTensor):
        return True
    return debox_other(other, "is_linear")


def debox_phi(other: Any) -> Any:
    # relative
    from .phi_tensor import PhiTensor

    if not isinstance(other, PhiTensor):
        return other
    return other.gamma


def update_state(state: Dict, other: Any) -> Dict:
    if isinstance(other, GammaTensor):
        state.update(other.sources)
    return state


SingleOrTupleInt = Union[int, Tuple[int, ...]]
OptionalAxisArg = Optional[SingleOrTupleInt]


def create_lookup_tables(dictionary: dict) -> Tuple[List[str], dict, List[dict]]:
    index2key: List = [str(x) for x in dictionary.keys()]
    key2index: dict = {key: i for i, key in enumerate(index2key)}
    # Note this maps to GammaTensor, not to GammaTensor.child as name may imply
    index2values: List = [dictionary[i] for i in index2key]

    return index2key, key2index, index2values


def create_new_lookup_tables(
    dictionary: dict,
) -> Tuple[Deque[str], dict, Deque[dict], Deque[int]]:
    index2key: Deque = deque()
    key2index: dict = {}
    index2values: Deque = (
        deque()
    )  # Note this maps to GammaTensor, not to GammaTensor.child as name may imply
    index2size: Deque = deque()
    for index, key in enumerate(dictionary.keys()):
        key = str(key)
        index2key.append(key)
        key2index[key] = index
        index2values.append(dictionary[key])
        index2size.append(len(dictionary[key]))

    return index2key, key2index, index2values, index2size


def jax2numpy(value: jnp.array, dtype: np.dtype) -> np.array:
    # are we incurring copying here?
    return np.asarray(value, dtype=dtype)


def numpy2jax(value: np.array, dtype: np.dtype) -> jnp.array:
    return jnp.asarray(value, dtype=dtype)


# ATTENTION: Shouldn't this be a subclass of some kind of base tensor so all the numpy
# methods and properties don't need to be re-implemented on it?
@dataclass
@serializable(recursive_serde=True)
class GammaTensor:
    child: jnp.array
    jax_op: SyftJaxOp = flax.struct.field(pytree_node=False)
    sources: dict = flax.struct.field(pytree_node=False)
    is_linear: bool = False
    id: str = flax.struct.field(pytree_node=False, default_factory=lambda: UID())

    __attr_allowlist__ = (
        "child",
        "jax_op",
        "sources",
        "is_linear",
        "id",
    )

    @classmethod
    def serde_constructor(cls, kwargs: Dict[str, Any]) -> GammaTensor:
        return GammaTensor(**kwargs)

    """
    A differential privacy tensor that contains data belonging to atleast 2 or more unique data subjects.

    Attributes:
        child: jnp.array
            The private data itself.
        min_vals: lazyrepeatarray
            (DP Metadata) A custom class that keeps track of (data-independent) minimum values for this tensor.
        max_vals: lazyrepeatarray
            (DP Metadata) A custom class that keeps track of (data-independent) maximum values for this tensor.
        jax_op: SyftJaxOp
        is_linear: bool
            Whether the "func_str" for this tensor is a linear query or not. This impacts the epsilon calculations
            when publishing.
        sources: dict
            A dictionary containing all the Tensors, integers, etc that were used to create this tensor.
            It maps an integer to each input object.
        id: int
            A 32-bit integer that is used when this GammaTensor needs to be added to the "sources" dictionary.

    Methods:
        All efforts were made to make this tensor's API as similar to the NumPy API as possible.
        Special, unique methods are listed below:

        reconstruct(sources: Optional[dict]):
            rebuilds the tensor from the sources dictionary provided, or from the current self.sources.
            This is exclusively used when adding DP Noise, if the data scientist doesn't have enough privacy budget to
            use one of the input tensors, thus requiring that tensor's data to be removed from the computation.

        swap_state(sources: Optional[Dict]):
            calls reconstruct() and populates the rest of the GammaTensor's attributes based on the current tensor.
            Used exclusively when adding DP Noise.



        decode():
            occasionally the use of a FixedPrecisionTensor (FPT) is needed during SMPC[1]. This helps convert back from
            FPT to regular numpy/jax arrays.

            (https://en.wikipedia.org/wiki/Secure_multi-party_computation)




    """

    PointerClassOverride = TensorWrappedGammaTensorPointer
    __array_ufunc__ = None

    child: jnp.array
    jax_op: SyftJaxOp = flax.struct.field(pytree_node=False)
    sources: dict = flax.struct.field(pytree_node=False)
    is_linear: bool = False
    id: str = flax.struct.field(pytree_node=False, default_factory=lambda: UID())

    def decode(self) -> np.ndarray:
        if isinstance(self.child, FixedPrecisionTensor):
            return self.child.decode()
        else:
            return self.child

    @property
    def proxy_public_kwargs(self) -> Dict[str, Any]:
        return {
            "min_vals": self.min_vals,
            "max_vals": self.max_vals,
            "data_subjects": self.data_subjects,
        }  # TODO 0.7: maybe this is obsolete now?

    def reconstruct(self, state: Dict) -> GammaTensor:
        return self.func(state)

    def swap_state(self, state: dict) -> GammaTensor:
        return GammaTensor(
            child=self.reconstruct(state),
            sources=state,
            jax_op=self.jax_op,
            is_linear=self.is_linear,
        )

    def astype(self, new_type: str) -> GammaTensor:
        return GammaTensor(
            child=self.child.astype(new_type),
            jax_op=self.jax_op,
            sources=self.sources,
            is_linear=self.is_linear,
            id=self.id,
        )

    @property
    def size(self) -> int:
        if (
            isinstance(self.child, float)
            or isinstance(self.child, int)
            or isinstance(self.child, bool)
        ):
            return 1

        if hasattr(self.child, "size"):
            return self.child.size
        elif hasattr(self.child, "shape"):
            return np.prod(self.child.shape)

        raise Exception(f"{type(self)} has no attribute size.")

    def func(self, state: Dict) -> GammaTensor:
        return self.jax_op.func(state)

    # infix operations

    @staticmethod
    def _infix_func(
        left: Any, right: Any, gamma_op: GAMMA_TENSOR_OP, is_linear_op: bool
    ) -> GammaTensor:
        left = debox_phi(left)
        right = debox_phi(right)
        state = left.sources.copy() if hasattr(left, "sources") else {}
        output_state = update_state(state, right)
        child = GAMMA_TENSOR_OP_FUNC[gamma_op](debox_child(left), debox_child(right))
        is_linear = debox_linear(left) and debox_linear(right) and is_linear_op
        jax_op = SyftJaxInfixOp(jax_op=gamma_op, left=left, right=right)

        return GammaTensor(
            child=child, jax_op=jax_op, sources=output_state, is_linear=is_linear
        )

    def _rinfix(
        self, other: Any, gamma_op: GAMMA_TENSOR_OP, is_linear_op: bool
    ) -> GammaTensor:
        return self._infix_func(
            left=other, right=self, gamma_op=gamma_op, is_linear_op=is_linear_op
        )

    def _infix(
        self, other: Any, gamma_op: GAMMA_TENSOR_OP, is_linear_op: bool
    ) -> GammaTensor:
        return self._infix_func(
            left=self, right=other, gamma_op=gamma_op, is_linear_op=is_linear_op
        )

    def _unary_op(
        self,
        gamma_op: GAMMA_TENSOR_OP,
        is_linear: bool = False,
        args: Optional[Any] = None,
        kwargs: Optional[Any] = None,
    ) -> GammaTensor:
        args = (
            args if args is not None else []
        )  # can't use collections in default params
        kwargs = (
            kwargs if kwargs is not None else {}
        )  # can't use collections in default params
        output_state = self.sources.copy()
        child = GAMMA_TENSOR_OP_FUNC[gamma_op](self.child, *args, **kwargs)
        jax_op = SyftJaxUnaryOp(jax_op=gamma_op, operand=self, args=args, kwargs=kwargs)
        return GammaTensor(
            child=child,
            jax_op=jax_op,
            sources=output_state,
            is_linear=is_linear and self.is_linear,
        )

    def __add__(self, other: Any) -> GammaTensor:
        return self._infix(other, gamma_op=GAMMA_TENSOR_OP.ADD, is_linear_op=True)

    def __sub__(self, other: Any) -> GammaTensor:
        return self._infix(other, gamma_op=GAMMA_TENSOR_OP.SUBTRACT, is_linear_op=True)

    def __mod__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(other, gamma_op=GAMMA_TENSOR_OP.MOD, is_linear_op=False)

    def __mul__(self, other: Any) -> GammaTensor:
        return self._infix(other, gamma_op=GAMMA_TENSOR_OP.MULTIPLY, is_linear_op=True)

    def __truediv__(self, other: Any) -> GammaTensor:
        return self._infix(
            other, gamma_op=GAMMA_TENSOR_OP.TRUE_DIVIDE, is_linear_op=True
        )

    def __floordiv__(self, other: Any) -> GammaTensor:
        return self._infix(
            other, gamma_op=GAMMA_TENSOR_OP.FLOOR_DIVIDE, is_linear_op=True
        )

    def __matmul__(self, other: Any) -> GammaTensor:
        return self._infix(other, gamma_op=GAMMA_TENSOR_OP.MATMUL, is_linear_op=False)

    def __gt__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(other, gamma_op=GAMMA_TENSOR_OP.GREATER, is_linear_op=False)

    def __ge__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(
        #     other, gamma_op=GAMMA_TENSOR_OP.GREATER_EQUAL, is_linear_op=False
        # )

    def __lt__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(other, gamma_op=GAMMA_TENSOR_OP.LESS, is_linear_op=False)

    def __le__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(
        #     other, gamma_op=GAMMA_TENSOR_OP.LESS_EQUAL, is_linear_op=False
        # )

    def __eq__(self, other: Any) -> GammaTensor:  # type: ignore
        raise NotImplementedError
        # return self._infix(other, gamma_op=GAMMA_TENSOR_OP.EQUAL, is_linear_op=False)

    def __ne__(self, other: Any) -> GammaTensor:  # type: ignore
        raise NotImplementedError
        # return self._infix(
        #     other, gamma_op=GAMMA_TENSOR_OP.NOT_EQUAL, is_linear_op=False
        # )

    def __and__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(
        #     other, gamma_op=GAMMA_TENSOR_OP.BITWISE_AND, is_linear_op=False
        # )

    def __or__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(
        #     other, gamma_op=GAMMA_TENSOR_OP.BITWISE_OR, is_linear_op=False
        # )

    def __lshift__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(other, gamma_op=GAMMA_TENSOR_OP.LSHIFT, is_linear_op=False)

    def __rshift__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(other, gamma_op=GAMMA_TENSOR_OP.RSHIFT, is_linear_op=False)

    def __xor__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._infix(
        #     other, gamma_op=GAMMA_TENSOR_OP.BITWISE_XOR, is_linear_op=False
        # )

    def dot(self, other: Union[np.ndarray, GammaTensor]) -> GammaTensor:
        # QUESTION: is there a reason other can't be a non gamma tensor numpy array?
        return self._infix(other, gamma_op=GAMMA_TENSOR_OP.DOT, is_linear_op=False)

    #  __r*__ infix operations

    def __radd__(self, other: Any) -> GammaTensor:
        # return self.__add__(other)
        return self._rinfix(other, gamma_op=GAMMA_TENSOR_OP.ADD, is_linear_op=True)

    def __rsub__(self, other: Any) -> GammaTensor:
        # return (self.__sub__(other)) * -1
        return self._rinfix(other, gamma_op=GAMMA_TENSOR_OP.SUBTRACT, is_linear_op=True)

    def __rmod__(self, other: Any) -> GammaTensor:
        raise NotImplementedError
        # return self._rinfix(other, gamma_op=GAMMA_TENSOR_OP.MOD, is_linear_op=False)

    def __rmul__(self, other: Any) -> GammaTensor:
        return self._rinfix(other, gamma_op=GAMMA_TENSOR_OP.MULTIPLY, is_linear_op=True)

    def __rtruediv__(self, other: Any) -> GammaTensor:
        return self._rinfix(
            other, gamma_op=GAMMA_TENSOR_OP.TRUE_DIVIDE, is_linear_op=True
        )

    def __rfloordiv__(self, other: Any) -> GammaTensor:
        return self._rinfix(
            other, gamma_op=GAMMA_TENSOR_OP.FLOOR_DIVIDE, is_linear_op=True
        )

    def __rmatmul__(self, other: Any) -> GammaTensor:
        return self._rinfix(other, gamma_op=GAMMA_TENSOR_OP.MATMUL, is_linear_op=False)

    def __divmod__(self, other: Any) -> GammaTensor:
        return self._rinfix(other, gamma_op=GAMMA_TENSOR_OP.DIVMOD, is_linear_op=False)

    def divmod(self, other: Any) -> GammaTensor:
        return self.__divmod__(other)

    #  unary operations

    def __abs__(self) -> GammaTensor:
        return self._unary_op(gamma_op=GAMMA_TENSOR_OP.ABS, is_linear=False)

    def argmax(self, axis: Optional[int] = None) -> GammaTensor:
        raise NotImplementedError
        # return self._unary_op(
        #     gamma_op=GAMMA_TENSOR_OP.ARGMAX, is_linear=False, args=[axis]
        # )

    def argmin(self, axis: Optional[int] = None) -> GammaTensor:
        raise NotImplementedError
        # return self._unary_op(
        #     gamma_op=GAMMA_TENSOR_OP.ARGMIN, is_linear=False, args=[axis]
        # )

    def log(self) -> GammaTensor:  # TODO 0.7: this needs a test
        return self._unary_op(gamma_op=GAMMA_TENSOR_OP.LOG, is_linear=False)

    def flatten(self, order: str = "C") -> GammaTensor:  # TODO 0.7: this needs a test
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.FLATTEN, is_linear=True, args=order
        )

    def transpose(self, *args: Any, **kwargs: Any) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.TRANSPOSE, is_linear=True, args=args, kwargs=kwargs
        )

    @property
    def T(self) -> GammaTensor:
        return self.transpose()

    def sum(self, *args: Any, **kwargs: Any) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.SUM, is_linear=True, args=args, kwargs=kwargs
        )

    def __pow__(self, *args: Any, **kwargs: Any) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.POWER, is_linear=False, args=args, kwargs=kwargs
        )

    def ones_like(self, *args: Any, **kwargs: Any) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.ONES_LIKE, is_linear=True, args=args, kwargs=kwargs
        )

    def zeros_like(self, *args: Any, **kwargs: Any) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.ZEROS_LIKE,
            is_linear=True,
            args=args,
            kwargs=kwargs,
        )

    def filtered(self, *args: Any, **kwargs: Any) -> GammaTensor:  # TODO
        raise NotImplementedError
        # return GammaTensor(
        #     child=jnp.zeros_like(self.child), jax_op=, sources=self.sources.copy()
        # )

    # def filtered(self) -> GammaTensor:
    #     # This is only used during publish to filter out data in GammaTensors with no_op. It serves no other purpose.
    #     def _filtered(state: Dict) -> GammaTensor:
    #         return self.reconstruct(state)

    #     func = _filtered

    #     return GammaTensor(
    #         child=jnp.zeros_like(self.child),
    #         func=func,
    #         sources=self.sources.copy(),
    #     )

    def __round__(self, n: int = 0) -> GammaTensor:
        return self._unary_op(gamma_op=GAMMA_TENSOR_OP.ROUND, is_linear=False, args=[n])

    def round(self, n: int = 0) -> GammaTensor:
        return self.__round__(n)

    def squeeze(self, axis: OptionalAxisArg = None) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.SQUEEZE, is_linear=True, args=[axis]
        )

    def mean(self, axis: OptionalAxisArg = None, **kwargs: Any) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.MEAN, is_linear=True, args=[axis], kwargs=kwargs
        )

    def ravel(self, order: Optional[str] = "C") -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.RAVEL, is_linear=True, args=[order]
        )

    def resize(self, new_shape: SingleOrTupleInt) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.RESIZE, is_linear=True, args=[new_shape]
        )

    def compress(
        self, condition: List[bool], axis: Optional[int] = None
    ) -> GammaTensor:
        output_state = self.sources.copy()
        child = jnp.compress(condition, self.child, axis=axis)
        jax_op = SyftJaxUnaryOp(
            jax_op=GAMMA_TENSOR_OP.COMPRESS,
            operand=self,
            args=[condition],
            kwargs={"axis": axis},
            operand_before=False,
        )
        return GammaTensor(
            child=child,
            jax_op=jax_op,
            sources=output_state,
            is_linear=self.is_linear,
        )

    def any(
        self,
        axis: OptionalAxisArg = None,
        keepdims: Optional[bool] = None,
    ) -> GammaTensor:
        raise NotImplementedError
        # return self._unary_op(
        #     gamma_op=GAMMA_TENSOR_OP.ANY,
        #     is_linear=False,
        #     kwargs={"axis": axis, "keepdims": keepdims},
        # )

    def all(
        self,
        axis: OptionalAxisArg = None,
        keepdims: Optional[bool] = None,
    ) -> GammaTensor:
        raise NotImplementedError
        # return self._unary_op(
        #     gamma_op=GAMMA_TENSOR_OP.ALL,
        #     is_linear=False,
        #     kwargs={"axis": axis, "keepdims": keepdims},
        # )

    def __pos__(self) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.POSITIVE,
            is_linear=True,
        )

    def __neg__(self) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.NEGATIVE,
            is_linear=True,
        )

    def reshape(
        self, newshape: SingleOrTupleInt, order: Optional[str] = "C"
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.RESHAPE,
            is_linear=True,
            args=[newshape],
            kwargs={"order": order},
        )

    def std(
        self,
        axis: OptionalAxisArg = None,
        dtype: Optional[np.dtype] = None,
        ddof: Optional[int] = 0,
        keepdims: Optional[bool] = None,
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.STD,
            is_linear=True,
            kwargs={
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
            },
        )

    def var(
        self,
        axis: OptionalAxisArg = None,
        dtype: Optional[np.dtype] = None,
        ddof: Optional[int] = 0,
        keepdims: Optional[bool] = None,
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.VAR,
            is_linear=True,
            kwargs={
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
            },
        )

    def sqrt(self, *args: Any, **kwargs: Any) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.SQRT,
            is_linear=False,
            args=args,
            kwargs=kwargs,
        )

    def abs(self, *args: Any, **kwargs: Any) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.ABS,
            is_linear=False,
            args=args,
            kwargs=kwargs,
        )

    def clip(
        self,
        a_min: Optional[ArrayLike] = None,
        a_max: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.CLIP,
            is_linear=False,
            args=[a_min, a_max],
            kwargs=kwargs,
        )

    def nonzero(self) -> GammaTensor:
        raise NotImplementedError
        # return self._unary_op(gamma_op=GAMMA_TENSOR_OP.NONZERO, is_linear=False)

    def swapaxes(self, axis1: int, axis2: int) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.SWAPAXES, is_linear=True, args=[axis1, axis2]
        )

    def __len__(self) -> int:
        if not hasattr(self.child, "__len__"):
            if self.child is None:
                return 0
            return 1
        try:
            return len(self.child)
        except Exception:  # nosec
            return self.child.size

    def __getitem__(self, key: Union[int, slice, ArrayLike]) -> GammaTensor:
        # TODO: I think we can move the mapping of the final getattr(jnp, "op") to be more general
        # to also accommodate this kind of pattern where its a lambda or whever
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.PY_GETITEM, is_linear=True, args=[key]
        )

    def __setitem__(
        self, key: Union[int, slice, ArrayLike], value: Union[GammaTensor, ArrayLike]
    ) -> None:
        raise NotImplementedError
        # QUESTION: Is mutation allowed and if so how do we trace that in the graph?
        # self._unary_op(gamma_op=GAMMA_TENSOR_OP.PY_GETITEM, is_linear=True, args=[key])
        # # relative
        # from .phi_tensor import PhiTensor

        # # TODO: fix this
        # if isinstance(value, (PhiTensor, GammaTensor)):
        #     self.child[key] = value.child
        # elif isinstance(value, np.ndarray):
        #     self.child[key] = value
        # else:
        #     raise NotImplementedError

    def copy(self, order: str = "C") -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.COPY, is_linear=True, args=[order]
        )

    def ptp(self, axis: OptionalAxisArg = None) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.PTP, is_linear=False, args=[axis]
        )

    def take(
        self,
        indices: ArrayLike,
        axis: Optional[int] = None,
        mode: str = "clip",
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.TAKE,
            is_linear=True,
            args=[indices],
            kwargs={"axis": axis, "mode": mode},
        )

    def put(
        self,
        ind: ArrayLike,
        v: ArrayLike,
        mode: str = "raise",
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.PUT,
            is_linear=True,
            args=[ind, v],
            kwargs={"mode": mode},
        )

    def repeat(
        self, repeats: SingleOrTupleInt, axis: Optional[int] = None
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.REPEAT,
            is_linear=True,
            args=[repeats],
            kwargs={"axis": axis},
        )

    def cumsum(
        self, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.CUMSUM,
            is_linear=False,
            kwargs={"axis": axis, "dtype": dtype},
        )

    def cumprod(
        self, axis: Optional[int] = None, dtype: Optional[np.dtype] = None
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.CUMPROD,
            is_linear=False,
            kwargs={"axis": axis, "dtype": dtype},
        )

    @property
    def lipschitz_bound(self) -> float:
        if self.is_linear:
            return 1.0

        def convert_array_to_dict_state(array_state: Dict, input_sizes: Dict) -> Dict:
            start_id = 0
            state = {}

            for id, shape in input_sizes.items():
                total_size = 1
                for size in shape:
                    total_size *= size
                state[id] = np.reshape(
                    array_state[start_id : start_id + total_size], shape  # noqa: E203
                )
                start_id += total_size

            return state

        def convert_state_to_bounds(input_sizes: Dict, input_states: Dict) -> List:
            bounds = []
            for id in input_sizes:
                bounds.extend(
                    list(
                        zip(
                            input_states[id].min_vals.to_numpy().flatten(),
                            input_states[id].max_vals.to_numpy().flatten(),
                        )
                    )
                )
            return bounds

        grad_fn = jax.grad(jax.jit(lambda state: jnp.sum(self.func(state))))

        input_sizes = {tensor.id: tensor.shape for tensor in self.sources.values()}
        bounds = convert_state_to_bounds(input_sizes, self.sources)

        def search(array_state: Dict) -> jnp.DeviceArray:
            dict_state = convert_array_to_dict_state(array_state, input_sizes)
            grads = grad_fn(dict_state)
            return -jnp.max(jnp.array(list(grads.values())))

        return -shgo(search, bounds=bounds, sampling_method="simplicial").fun

    def prod(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.PROD, is_linear=False, kwargs={"axis": axis}
        )

    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.TRACE,
            is_linear=True,
            kwargs={"offset": offset, "axis1": axis1, "axis2": axis2},
        )

    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.DIAGONAL,
            is_linear=True,
            kwargs={"offset": offset, "axis1": axis1, "axis2": axis2},
        )

    def min(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[float] = None,
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.MIN,
            is_linear=False,
            kwargs={"axis": axis, "keepdims": keepdims, "initial": initial},
        )

    def max(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[float] = None,
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.MAX,
            is_linear=False,
            kwargs={"axis": axis, "keepdims": keepdims, "initial": initial},
        )

    def sort(self, axis: int = -1, kind: Optional[str] = None) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.SORT,
            is_linear=False,
            kwargs={"axis": axis, "kind": kind},
        )

    def argsort(self, axis: int = -1, kind: Optional[str] = None) -> GammaTensor:
        raise NotImplementedError
        # return self._unary_op(
        #     gamma_op=GAMMA_TENSOR_OP.ARGSORT,
        #     is_linear=False,
        #     kwargs={"axis": axis, "kind": kind},
        # )

    def choose(
        self,
        choices: Union[Sequence, np.ndarray, PassthroughTensor],
        mode: Optional[str] = "raise",
    ) -> GammaTensor:
        return self._unary_op(
            gamma_op=GAMMA_TENSOR_OP.CHOOSE,
            is_linear=True,
            kwargs={"choices": choices, "mode": mode},
        )

    # @staticmethod
    # def convert_dsl(
    #     state: dict, new_state: Optional[dict] = None
    # ) -> Dict:  # TODO 0.7: maybe this is not required?
    #     if new_state is None:
    #         new_state = dict()
    #     if state:
    #         for tensor in list(state.values()):
    #             if isinstance(tensor.data_subjects, np.ndarray):
    #                 new_tensor = GammaTensor(
    #                     child=tensor.child,
    #                     func=tensor.func,
    #                     sources=GammaTensor.convert_dsl(tensor.sources, {}),
    #                 )
    #             else:

    #                 new_tensor = tensor
    #             new_state[new_tensor.id] = new_tensor
    #         return new_state
    #     else:
    #         return {}

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: float,
        private: bool,
    ) -> np.ndarray:
        return publish(
            tensor=self,
            ledger=ledger,
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            sigma=sigma,
            is_linear=self.is_linear,
            private=private,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.child.shape

    @property
    def dtype(self) -> np.dtype:
        return self.child.dtype

    # def _object2bytes(self) -> bytes:
    #     # TODO Tudor: fix this
    #     schema = get_capnp_schema(schema_file="gamma_tensor.capnp")

    #     gamma_tensor_struct: CapnpModule = schema.GammaTensor  # type: ignore
    #     gamma_msg = gamma_tensor_struct.new_message()
    #     # this is how we dispatch correct deserialization of bytes
    #     gamma_msg.magicHeader = serde_magic_header(type(self))

    #     # do we need to serde func? if so how?
    #     # what about the state dict?

    #     if isinstance(self.child, np.ndarray) or np.isscalar(self.child):
    #         chunk_bytes(capnp_serialize(np.array(self.child), to_bytes=True), "child", gamma_msg)  # type: ignore
    #         gamma_msg.isNumpy = True
    #     elif isinstance(self.child, jnp.ndarray):
    #         chunk_bytes(
    #             capnp_serialize(jax2numpy(self.child, self.child.dtype), to_bytes=True),
    #             "child",
    #             gamma_msg,
    #         )
    #         gamma_msg.isNumpy = True
    #     else:
    #         chunk_bytes(serialize(self.child, to_bytes=True), "child", gamma_msg)  # type: ignore
    #         gamma_msg.isNumpy = False

    #     gamma_msg.sources = serialize(self.sources, to_bytes=True)
    #     gamma_msg.isLinear = self.is_linear
    #     gamma_msg.id = self.id.to_string()
    #     gamma_msg.jaxOp = serialize(self.jax_op, to_bytes=True)

    #     # return gamma_msg.to_bytes_packed()
    #     return gamma_msg.to_bytes()

    # @staticmethod
    # def _bytes2object(buf: bytes) -> GammaTensor:
    #     # TODO Tudor: fix this
    #     schema = get_capnp_schema(schema_file="gamma_tensor.capnp")
    #     gamma_struct: CapnpModule = schema.GammaTensor  # type: ignore
    #     # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
    #     MAX_TRAVERSAL_LIMIT = 2**64 - 1
    #     # capnp from_bytes is now a context
    #     with gamma_struct.from_bytes(
    #         buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
    #     ) as gamma_msg:

    #         if gamma_msg.isNumpy:
    #             child = capnp_deserialize(
    #                 combine_bytes(gamma_msg.child), from_bytes=True
    #             )
    #         else:
    #             child = deserialize(combine_bytes(gamma_msg.child), from_bytes=True)

    #         state = deserialize(gamma_msg.sources, from_bytes=True)
    #         is_linear = gamma_msg.isLinear
    #         id_str = UID.from_string(gamma_msg.id)
    #         jax_op = deserialize(gamma_msg.jaxOp, from_bytes=True)

    #         return GammaTensor(
    #             child=child,
    #             is_linear=is_linear,
    #             sources=state,
    #             id=id_str,
    #             jax_op=jax_op,
    #         )
