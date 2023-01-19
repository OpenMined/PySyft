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
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs

# from ....core.adp.data_subject import DataSubject
from ....core.node.common.action.get_or_set_property_action import (
    GetOrSetPropertyAction,
)
from ....core.node.common.action.get_or_set_property_action import PropertyActions
from ....lib.numpy.array import capnp_deserialize
from ....lib.numpy.array import capnp_serialize
from ....lib.python.util import upcast
from ....util import inherit_tags
from ...adp.data_subject_ledger import DataSubjectLedger

# from ...adp.data_subject_list import DataSubjectList
# from ...adp.data_subject_list import DataSubjectArray
# from ...adp.data_subject_list import dslarraytonumpyutf8
# from ...adp.data_subject_list import numpyutf8todslarray
from ...adp.vectorized_publish import publish
from ...common.serde.capnp import CapnpModule
from ...common.serde.capnp import chunk_bytes
from ...common.serde.capnp import combine_bytes
from ...common.serde.capnp import get_capnp_schema
from ...common.serde.capnp import serde_magic_header
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize as serialize
from ...common.uid import UID
from ...node.abstract.node import AbstractNodeClient
from ...node.common.action.run_class_method_action import RunClassMethodAction
from ...node.enums import PointerStatus
from ...pointer.pointer import Pointer
from ..config import DEFAULT_INT_NUMPY_TYPE
from ..fixed_precision_tensor import FixedPrecisionTensor
from ..passthrough import PassthroughTensor  # type: ignore
from ..smpc import utils
from ..smpc.mpc_tensor import MPCTensor
from ..smpc.utils import TYPE_TO_RING_SIZE
from ..util import implements
from .gamma_tensor_ops import GAMMA_TENSOR_OP
from .gamma_tensor_ops import GAMMA_TENSOR_OP_FUNC
from .jax_ops import SyftJaxInfixOp
from .jax_ops import SyftJaxOp
from .jax_ops import SyftJaxUnaryOp

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


@serializable(recursive_serde=True)
class TensorWrappedGammaTensorPointer(Pointer):
    __name__ = "TensorWrappedGammaTensorPointer"
    __module__ = "syft.core.tensor.autodp.gamma_tensor"
    __attr_allowlist__ = [
        # default pointer attrs
        "client",
        "id_at_location",
        "object_type",
        "tags",
        "description",
        # phi_tensor attrs
        "public_dtype",
        "public_shape",
    ]

    __serde_overrides__ = {
        "client": [lambda x: x.address, lambda y: y],
        "public_shape": [lambda x: x, lambda y: upcast(y)],
        # "data_subjects": [dslarraytonumpyutf8, numpyutf8todslarray],
        "public_dtype": [lambda x: str(x), lambda y: np.dtype(y)],
    }
    _exhausted = False
    is_enum = False
    PUBLISH_POINTER_TYPE = "numpy.ndarray"
    __array_ufunc__ = None

    def __init__(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
        public_shape: Optional[Tuple[int, ...]] = None,
        public_dtype: Optional[np.dtype] = None,
    ):
        super().__init__(
            client=client,
            id_at_location=id_at_location,
            object_type=object_type,
            tags=tags,
            description=description,
        )
        self.public_shape = public_shape
        self.public_dtype = public_dtype

    # TODO: Modify for large arrays
    @property
    def synthetic(self) -> np.ndarray:
        public_dtype_func = getattr(
            self.public_dtype, "upcast", lambda: self.public_dtype
        )
        return (
            np.random.rand(*list(self.public_shape))  # type: ignore
            * (self.max_vals.to_numpy() - self.min_vals.to_numpy())
            + self.min_vals.to_numpy()
        ).astype(public_dtype_func())

    def __repr__(self) -> str:
        repr_string = f"PointerId: {self.id_at_location.no_dash}"
        if hasattr(self.client, "obj_exists"):
            _ptr_status = (
                PointerStatus.READY.value
                if self.exists
                else PointerStatus.PROCESSING.value
            )
            repr_string += f"\nStatus: {_ptr_status}"
        repr_string += f"\nRepresentation: {self.synthetic.__repr__()}"
        repr_string += "\n\n(The data printed above is synthetic - it's an imitation of the real data.)"
        return repr_string

    def share(self, *parties: Tuple[AbstractNodeClient, ...]) -> MPCTensor:
        all_parties = list(parties) + [self.client]
        ring_size = TYPE_TO_RING_SIZE.get(self.public_dtype, None)
        self_mpc = MPCTensor(
            secret=self,
            shape=self.public_shape,
            ring_size=ring_size,
            parties=all_parties,
        )
        return self_mpc

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        if hasattr(self, "public_shape"):
            return self.public_shape
        else:
            return None

    def _apply_tensor_op(self, other: Any, op_str: str) -> Any:
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass
        # We always maintain a Tensor hierarchy Tensor ---> PT--> Actual Data
        attr_path_and_name = f"syft.core.tensor.tensor.Tensor.{op_str}"

        result = TensorWrappedGammaTensorPointer(
            client=self.client,
        )

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)

        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=[other], kwargs={})

            # then we convert anything which isnt a pointer into a pointer
            pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
                args=downcast_args,
                kwargs=downcast_kwargs,
                client=self.client,
                gc_enabled=False,
            )

            cmd = RunClassMethodAction(
                path=attr_path_and_name,
                _self=self,
                args=pointer_args,
                kwargs=pointer_kwargs,
                id_at_location=result_id_at_location,
                address=self.client.address,
            )
            self.client.send_immediate_msg_without_reply(msg=cmd)

        inherit_tags(
            attr_path_and_name=attr_path_and_name,
            result=result,
            self_obj=self,
            args=[other],
            kwargs={},
        )

        result_public_shape = None

        if isinstance(other, TensorWrappedGammaTensorPointer):
            other_shape = other.public_shape
            other_dtype = other.public_dtype
        elif isinstance(other, (int, float)):
            other_shape = (1,)
            other_dtype = DEFAULT_INT_NUMPY_TYPE
        elif isinstance(other, bool):
            other_shape = (1,)
            other_dtype = np.dtype("bool")
        elif isinstance(other, np.ndarray):
            other_shape = other.shape
            other_dtype = other.dtype
        else:
            raise ValueError(
                f"Invalid Type for TensorWrappedGammaTensorPointer:{type(other)}"
            )

        if self.public_shape is not None and other_shape is not None:
            result_public_shape = utils.get_shape(
                op_str, self.public_shape, other_shape
            )

        if self.public_dtype is None or other_dtype is None:
            if self.public_dtype != other_dtype:
                raise ValueError(
                    f"Dtype for self: {self.public_dtype} and other :{other_dtype} should not be None"
                )

        # calculate the dtype of the result based on the op_str
        result_public_dtype = utils.get_dtype(
            op_str, self.public_shape, other_shape, self.public_dtype, other_dtype
        )

        result.public_shape = result_public_shape
        result.public_dtype = result_public_dtype

        result.client.processing_pointers[result.id_at_location] = True

        return result

    @staticmethod
    def _apply_op(
        self: TensorWrappedGammaTensorPointer,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
        op_str: str,
    ) -> Union[MPCTensor, TensorWrappedGammaTensorPointer]:
        """Performs the operation based on op_str

        Args:
            other (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]): second operand.

        Returns:
            Tuple[MPCTensor,Union[MPCTensor,int,float,np.ndarray]] : Result of the operation
        """
        # relative
        from ..autodp.phi_tensor import TensorWrappedPhiTensorPointer

        if isinstance(other, TensorWrappedPhiTensorPointer):
            other = other.gamma

        if (
            isinstance(other, TensorWrappedGammaTensorPointer)
            and self.client != other.client
        ):

            parties = [self.client, other.client]

            self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
            other_mpc = MPCTensor(
                secret=other, shape=other.public_shape, parties=parties
            )

            return getattr(self_mpc, op_str)(other_mpc)

        elif isinstance(other, MPCTensor):

            return getattr(other, op_str)(self)

        return self._apply_tensor_op(other=other, op_str=op_str)

    def _apply_self_tensor_op(self, op_str: str, *args: Any, **kwargs: Any) -> Any:
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass

        # We always maintain a Tensor hierarchy Tensor ---> PT--> Actual Data
        attr_path_and_name = f"syft.core.tensor.tensor.Tensor.{op_str}"
        result = TensorWrappedGammaTensorPointer(
            client=self.client,
        )

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)

        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=args, kwargs=kwargs)

            # then we convert anything which isnt a pointer into a pointer
            pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
                args=downcast_args,
                kwargs=downcast_kwargs,
                client=self.client,
                gc_enabled=False,
            )

            cmd = RunClassMethodAction(
                path=attr_path_and_name,
                _self=self,
                args=pointer_args,
                kwargs=pointer_kwargs,
                id_at_location=result_id_at_location,
                address=self.client.address,
            )
            self.client.send_immediate_msg_without_reply(msg=cmd)

        inherit_tags(
            attr_path_and_name=attr_path_and_name,
            result=result,
            self_obj=self,
            args=args,
            kwargs=kwargs,
        )

        # relative
        from ..autodp.phi_tensor import TensorWrappedPhiTensorPointer

        if op_str == "choose":
            dummy_res = np.ones(self.public_shape, dtype=np.int64)
            if isinstance(
                args[0],
                (TensorWrappedPhiTensorPointer, TensorWrappedGammaTensorPointer),
            ):
                temp_args = (np.ones(args[0].shape, dtype=np.int64), *args[1:])
                dummy_res = getattr(dummy_res, op_str)(*temp_args, **kwargs)
            else:
                dummy_res = getattr(dummy_res, op_str)(*args, **kwargs)
        else:
            dummy_res = np.empty(self.public_shape)
            if hasattr(dummy_res, op_str):
                if op_str in INPLACE_OPS:
                    getattr(dummy_res, op_str)(*args, **kwargs)
                else:
                    dummy_res = getattr(dummy_res, op_str)(*args, **kwargs)
            elif hasattr(np, op_str):
                dummy_res = getattr(np, op_str)(dummy_res, *args, *kwargs)
            else:
                raise ValueError(f"Invalid Numpy Operation: {op_str} for Pointer")

        result.public_shape = dummy_res.shape
        result.public_dtype = dummy_res.dtype

        return result

    def copy(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
        return self._apply_self_tensor_op("copy", *args, **kwargs)

    def __add__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__add__")

    def __radd__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "radd" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__radd__")

    def __sub__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "sub" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__sub__")

    def __rsub__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "rsub" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__rsub__")

    def __mul__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__mul__")

    def __rmul__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "rmul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__rmul__")

    def __matmul__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "matmul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__matmul__")

    def __rmatmul__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "rmatmul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__rmatmul__")

    def __lt__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "lt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__lt__")

    def __gt__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "gt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__gt__")

    def __ge__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "ge" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__ge__")

    def __le__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "le" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__le__")

    def __eq__(  # type: ignore
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "eq" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__eq__")

    def __ne__(  # type: ignore
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "ne" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__ne__")

    def concatenate(
        self,
        other: TensorWrappedGammaTensorPointer,
        *args: Any,
        **kwargs: Any,
    ) -> MPCTensor:
        """Apply the "concatenate" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.


        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        if not isinstance(other, TensorWrappedGammaTensorPointer):
            raise ValueError(
                f"Concatenate works only for TensorWrappedGammaTensorPointer got type: {type(other)}"
            )

        if self.client != other.client:

            parties = [self.client, other.client]

            self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
            other_mpc = MPCTensor(
                secret=other, shape=other.public_shape, parties=parties
            )

            return self_mpc.concatenate(other_mpc, *args, **kwargs)

        else:
            raise ValueError(
                "Concatenate method currently works only between two different clients."
            )

    def __truediv__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__truediv__")

    def __rtruediv__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "rtruediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__rtruediv__")

    def __mod__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__mod__")

    def __and__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "and" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__and__")

    def __or__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "or" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__or__")

    def __floordiv__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "floordiv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__floordiv__")

    def __rfloordiv__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "rfloordiv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__rfloordiv__")

    def __divmod__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Tuple[
        Union[TensorWrappedGammaTensorPointer, MPCTensor],
        Union[TensorWrappedGammaTensorPointer, MPCTensor],
    ]:
        """Apply the "divmod" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self.divmod(other)

    def divmod(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Tuple[
        Union[TensorWrappedGammaTensorPointer, MPCTensor],
        Union[TensorWrappedGammaTensorPointer, MPCTensor],
    ]:
        """Apply the "divmod" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(
            self, other, "__floordiv__"
        ), TensorWrappedGammaTensorPointer._apply_op(self, other, "__mod__")

    def sum(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Sum of array elements over a given axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which a sum is performed.
                The default, axis=None, will sum all of the elements of the input array.
                If axis is negative it counts from the last to the first axis.
                If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a
                single axis or all the axes as before.
            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the sum method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.
            initial: scalar, optional
                Starting value for the sum. See reduce for details.
            where: array_like of bool, optional
                Elements to include in the sum. See reduce for details.
        """
        return self._apply_self_tensor_op("sum", *args, **kwargs)

    def ptp(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "ptp" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("ptp", *args, **kwargs)

    def __lshift__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "lshift" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__lshift__")

    def argmax(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "argmax" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return self._apply_self_tensor_op("argmax", *args, **kwargs)

    def __rshift__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "rshift" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__rshift__")

    def argmin(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "argmin" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return self._apply_self_tensor_op("argmin", *args, **kwargs)

    def __abs__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "abs" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("__abs__", *args, **kwargs)

    def all(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "all" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return self._apply_self_tensor_op("all", *args, **kwargs)

    def any(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "any" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return self._apply_self_tensor_op("any", *args, **kwargs)

    def round(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
        return self._apply_self_tensor_op("round", *args, **kwargs)

    def __round__(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
        return self.round(*args, **kwargs)

    def __pos__(self) -> TensorWrappedGammaTensorPointer:
        """Apply the pos (+) operator  on self.

        Returns:
            Union[TensorWrappedGammaTensorPointer] : Result of the operation.
        """
        return self._apply_self_tensor_op(op_str="__pos__")

    def var(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Compute the variance along the specified axis of the array elements, a measure of the spread of a distribution.
        The variance is computed for the flattened array by default, otherwise over the specified axis.

        Parameters

            axis: None or int or tuple of ints, optional
                Axis or axes along which the variance is computed.
                The default is to compute the variance of the flattened array.
                If this is a tuple of ints, a variance is performed over multiple axes, instead of a single axis or all
                the axes as before.

            ddof: int, optional
                “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof, where N represents the
                number of elements. By default ddof is zero.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the var method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            where: array_like of bool, optional
                Elements to include in the variance. See reduce for details.
        """
        return self._apply_self_tensor_op("var", *args, **kwargs)

    def cumsum(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """ "
        Return the cumulative sum of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative sum is computed. The default (None) is to compute the cumsum over the
                flattened array.
        Returns
            cumsum_along_axis: PhiTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        return self._apply_self_tensor_op("cumsum", *args, **kwargs)

    def cumprod(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return the cumulative product of the elements along a given axis.

        Parameters
            axis: int, optional
                Axis along which the cumulative product is computed. The default (None) is to compute the cumprod over
                the flattened array.
        Returns
            cumprod_along_axis: PhiTensor
                A new array holding the result is returned. The result has the same size as input, and the same shape as
                 a if axis is not None or a is 1-d.
        """
        return self._apply_self_tensor_op("cumprod", *args, **kwargs)

    def prod(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return the product of array elements over a given axis.
        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which a product is performed.
                The default, axis=None, will calculate the product of all the elements in the input array.
                If axis is negative it counts from the last to the first axis.
                If axis is a tuple of ints, a product is performed on all of the axes specified in the tuple instead of
                a single axis or all the axes as before.
            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the prod method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.
            initial: scalar, optional
                The starting value for this product. See reduce for details.
            where: array_like of bool, optional
                Elements to include in the product. See reduce for details.
        """
        return self._apply_self_tensor_op("prod", *args, **kwargs)

    def __xor__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "xor" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        raise NotImplementedError
        # return TensorWrappedGammaTensorPointer._apply_op(self, other, "__xor__")

    def __pow__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        First array elements raised to powers from second array, element-wise.

        Raise each base in x1 to the positionally-corresponding power in x2.
        x1 and x2 must be broadcastable to the same shape.
        An integer type raised to a negative integer power will raise a ValueError.
        Negative values raised to a non-integral value will return nan.

        Parameters
            x2: array_like

                The exponents. If self.shape != x2.shape, they must be broadcastable to a common shape.

            where: array_like, optional

                This condition is broadcast over the input. At locations where the condition is True, the out array will
                 be set to the ufunc result.
                 Elsewhere, the out array will retain its original value.

            **kwargs
                For other keyword-only arguments, see the ufunc docs.

        Returns
            y: PhiTensorPointer
                The bases in the tensor raised to the exponents in x2. This is a scalar if both self and x2 are scalars.
        """
        return self._apply_self_tensor_op("__pow__", *args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
        """
        Compute the arithmetic mean along the specified axis.

        Returns the average of the array elements. The average is taken over the flattened array by default, otherwise
        over the specified axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to compute the mean of the flattened
                array.
        """
        return self._apply_self_tensor_op("mean", *args, **kwargs)

    def std(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Compute the standard deviation along the specified axis.
        Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
        The standard deviation is computed for the flattened array by default, otherwise over the specified axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which the standard deviation is computed.
                The default is to compute the standard deviation of the flattened array.
                If this is a tuple of ints, a standard deviation is performed over multiple axes, instead of a single
                axis or all the axes as before.

            out: ndarray, optional
                Alternative output array in which to place the result. It must have the same shape as the expected
                output but the type (of the calculated values) will be cast if necessary.

            ddof: int, optional
                ddof = Delta Degrees of Freedom. By default ddof is zero.
                The divisor used in calculations is N - ddof, where N represents the number of elements.

            keepdims: bool, optional
                If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.

                If the default value is passed, then keepdims will not be passed through to the std method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised.

            where: array_like of bool, optional
                Elements to include in the standard deviation. See reduce for details.

        Returns

            standard_deviation: PhiTensor
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.std"
        result = TensorWrappedGammaTensorPointer(
            client=self.client,
        )

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)

        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=args, kwargs=kwargs)

            # then we convert anything which isnt a pointer into a pointer
            pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
                args=downcast_args,
                kwargs=downcast_kwargs,
                client=self.client,
                gc_enabled=False,
            )

            cmd = RunClassMethodAction(
                path=attr_path_and_name,
                _self=self,
                args=pointer_args,
                kwargs=pointer_kwargs,
                id_at_location=result_id_at_location,
                address=self.client.address,
            )
            self.client.send_immediate_msg_without_reply(msg=cmd)

        inherit_tags(
            attr_path_and_name=attr_path_and_name,
            result=result,
            self_obj=self,
            args=[],
            kwargs={},
        )
        result.public_shape = self.client.shape  # data_subjects.shape
        result.public_dtype = self.public_dtype

        return result

    def trace(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return the sum along diagonals of the array.

        If a is 2-D, the sum along its diagonal with the given offset is returned, i.e., the sum of elements
        a[i,i+offset] for all i.

        If a has more than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
        sub-arrays whose traces are returned. The shape of the resulting array is the same as that of a with axis1 and
        axis2 removed.

        Parameters

            offset: int, optional
                Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.

            axis1, axis2: int, optional
                Axes to be used as the first and second axis of the 2-D sub-arrays from which the diagonals should be
                taken. Defaults are the first two axes of a.

        Returns

            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
                If a is 2-D, the sum along the diagonal is returned.
                If a has larger dimensions, then an array of sums along diagonals is returned.

        """
        return self._apply_self_tensor_op("trace", *args, **kwargs)

    def sort(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
        """
        Return a sorted copy of an array.

        Parameters

            a: array_like
                Array to be sorted.

            axis: int or None, optional
                Axis along which to sort. If None, the array is flattened before sorting.
                The default is -1, which sorts along the last axis.

            kind{‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional
                Sorting algorithm. The default is ‘quicksort’.
                Note that both ‘stable’ and ‘mergesort’ use timsort or radix sort under the covers and, in general,
                the actual implementation will vary with data type. The ‘mergesort’ option is retained for backwards
                compatibility.

                Changed in version 1.15.0.: The ‘stable’ option was added.

            order: str or list of str, optional
                When a is an array with fields defined, this argument specifies which fields to compare first, second,
                etc. A single field can be specified as a string, and not all fields need be specified, but unspecified
                 fields will still be used, in the order in which they come up in the dtype, to break ties.

        Please see docs here: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """
        return self._apply_self_tensor_op("sort", *args, **kwargs)

    def argsort(self, *args: Any, **kwargs: Any) -> TensorWrappedGammaTensorPointer:
        """
        Returns the indices that would sort an array.

        Perform an indirect sort along the given axis using the algorithm specified by the kind keyword.
        It returns an array of indices of the same shape as a that index data along the given axis in sorted order.

        Parameters
            axis: int or None, optional
                Axis along which to sort. The default is -1 (the last axis). If None, the flattened array is used.
            kind: {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional
                Sorting algorithm. The default is ‘quicksort’. Note that both ‘stable’ and ‘mergesort’ use timsort
                under the covers and, in general, the actual implementation will vary with data type. The ‘mergesort’
                option is retained for backwards compatibility.
            order: str or list of str, optional
                When a is an array with fields defined, this argument specifies which fields to compare 1st, 2nd, etc.
                A single field can be specified as a string, and not all fields need be specified, but unspecified
                fields will still be used, in the order in which they come up in the dtype, to break ties.

        Returns
            index_array: ndarray, int
                Array of indices that sort a along the specified axis. If a is one-dimensional, a[index_array] yields a
                sorted a. More generally, np.take_along_axis(a, index_array, axis=axis) always yields the sorted a,
                irrespective of dimensionality.
        """
        raise NotImplementedError
        # return self._apply_self_tensor_op("argsort", *args, **kwargs)

    def min(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return the minimum of an array or minimum along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

        Returns
            a_min: PhiTensor
                Minimum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """
        return self._apply_self_tensor_op("min", *args, **kwargs)

    def max(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return the maximum of an array or along an axis.

        Parameters
            axis: None or int or tuple of ints, optional
                Axis or axes along which to operate. By default, flattened input is used.
                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.

        Returns
            a_max: PhiTensor
                Maximum of a.
                If axis is None, the result is a scalar value.
                If axis is given, the result is an array of dimension a.ndim - 1.
        """
        return self._apply_self_tensor_op("max", *args, **kwargs)

    def compress(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return selected slices of an array along given axis.

        When working along a given axis, a slice along that axis is returned in output for each index
        where condition evaluates to True. When working on a 1-D array, compress is equivalent to extract.

        Parameters
            condition: 1-D array of bools
            Array that selects which entries to return. If len(condition) is less than the size of
            a along the given axis,then output is truncated to the length of the condition array.

            axis: int, optional
            Axis along which to take slices. If None (default), work on the flattened array.

        Returns:
            compressed_array: PhiTensor
            A copy of a without the slices along axis for which condition is false.
        """
        return self._apply_self_tensor_op("compress", *args, **kwargs)

    def squeeze(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Remove axes of length one from a.

        Parameters
            axis: None or int or tuple of ints, optional
                Selects a subset of the entries of length one in the shape.
                If an axis is selected with shape entry greater than one, an error is raised.

        Returns:
            squeezed: PhiTensor
                The input array, but with all or a subset of the dimensions of length 1 removed.
                This is always a itself or a view into a.
                Note that if all axes are squeezed, the result is a 0d array and not a scalar.
        """
        return self._apply_self_tensor_op("squeeze", *args, **kwargs)

    def __getitem__(
        self, key: Union[int, bool, slice]
    ) -> TensorWrappedGammaTensorPointer:
        """Return self[key].
        Args:
            y (Union[int,bool,slice]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer] : Result of the operation.
        """
        return self._apply_self_tensor_op("__getitem__", key)

    def zeros_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "zeros_like" operation on "self"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("zeros_like", *args, **kwargs)

    def ones_like(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "ones_like" operation on "self"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("ones_like", *args, **kwargs)

    def transpose(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Reverse or permute the axes of an array; returns the modified array.

        Returns
            p: ndarray
                array with its axes permuted. A view is returned whenever possible.
        """

        return self._apply_self_tensor_op("transpose", *args, **kwargs)

    def resize(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:

        """
        Return a new array with the specified shape.

        Parameters
            new_shape: int or tuple of int
                Shape of resized array.

        Returns
            reshaped_array: ndarray
                The new array is formed from the data in the old array,
                repeated if necessary to fill out the required number of elements.
                The data are repeated iterating over the array in C-order.

        """
        return self._apply_self_tensor_op("resize", *args, **kwargs)

    def reshape(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:

        """
        Gives a new shape to an array without changing its data.

        Parameters
            new_shape: int or tuple of int
                The new shape should be compatible with the original shape. If an integer, then the result will
                be a 1-D array of that length. One shape dimension can be -1. In this case,
                the value is inferred from the length of the array and remaining dimensions.

        Returns
            reshaped_array: ndarray
                This will be a new view object if possible; otherwise, it will be a copy.
                Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.
        """
        return self._apply_self_tensor_op("reshape", *args, **kwargs)

    def repeat(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the repeat" operation

        Args:
            y (Union[TensorWrappedPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("repeat", *args, **kwargs)

    def diagonal(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return specified diagonals.
        If a is 2-D, returns the diagonal of a with the given offset, i.e., the collection of elements
        of the form a[i, i+offset].

        If a has more than two dimensions, then the axes specified by axis1 and axis are used to determine
        the 2-D sub-array whose diagonal is returned.  The shape of the resulting array can be determined by
        removing axis1 and axis2 and appending an index to the right equal to the size of the resulting diagonals.

        Parameters

            offset: int, optional
                Offset of the diagonal from the main diagonal.  Can be positive or negative.
                Defaults to main diagonal (0).
            axis1, axis2: int, optional
                Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken.
                Defaults are the first two axes of a.

        Returns
            array_of_diagonals : Union[TensorWrappedPhiTensorPointer,MPCTensor]
                If a is 2-D, then a 1-D array containing the diagonal and of the same type as a is returned unless
                a is a matrix, in which case
                a 1-D array rather than a (2-D) matrix is returned in order to maintain backward compatibility.

                If a.ndim > 2, then the dimensions specified by axis1 and axis2 are removed, and a new axis
                inserted at the end corresponding to the diagonal.
        """
        return self._apply_self_tensor_op("diagonal", *args, **kwargs)

    def flatten(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return a copy of the array collapsed into one dimension.

        Parameters
            order: {‘C’, ‘F’, ‘A’, ‘K’}, optional
            ‘C’ means to flatten in row-major (C-style) order.
            ‘F’ means to flatten in column-major (Fortran- style) order.
            ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory, row-major order otherwise.
            ‘K’ means to flatten a in the order the elements occur in memory. The default is ‘C’.

        Returns
            y: PhiTensor
                A copy of the input array, flattened to one dimension.
        """
        return self._apply_self_tensor_op("flatten", *args, **kwargs)

    def ravel(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Return a contiguous flattened array.

        A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.

        As of NumPy 1.10, the returned array will have the same type as the input array.
        (for example, a masked array will be returned for a masked array input)
        Parameters
            order: {‘C’,’F’, ‘A’, ‘K’}, optional
            The elements of a are read using this index order.
            ‘C’ means to index the elements in row-major,
            C-style order, with the last axis index changing fastest, back to the first axis index changing slowest.
            ‘F’ means to index the elements in column-major, Fortran-style order, with the first index changing fastest,
             and the last index changing slowest.
            Note that the ‘C’ and ‘F’ options take no account of the memory layout of the underlying array,
             and only refer to the order of axis indexing.
            ‘A’ means to read the elements in Fortran-like index order if a is Fortran contiguous in memory,
             C-like order otherwise.
            ‘K’ means to read the elements in the order they occur in memory, except for reversing the data
             when strides are negative.
            By default, ‘C’ index order is used.

        Returns:
            y: PhiTensor
                y is an array of the same subtype as a, with shape (a.size,).
                Note that matrices are special cased for backward compatibility,
                if a is a matrix, then y is a 1-D ndarray.
        """
        return self._apply_self_tensor_op("ravel", *args, **kwargs)

    def take(
        self, *args: Any, **kwargs: Any
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Take elements from an array along an axis.

        When axis is not None, this function does the same thing as “fancy” indexing (indexing arrays using arrays);
        however, it can be easier to use if you need elements along a given axis.
        A call such as np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...].

        Explained without fancy indexing, this is equivalent to the following use of ndindex, \
        which sets each of ii, jj, and kk to a tuple of indices:

            Ni, Nk = a.shape[:axis], a.shape[axis+1:]
            Nj = indices.shape
            for ii in ndindex(Ni):
                for jj in ndindex(Nj):
                    for kk in ndindex(Nk):
                        out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

        Parameters
            indices: array_like (Nj…)
                The indices of the values to extract.

            axis: int, optional
                The axis over which to select values. By default, the flattened input array is used.

            mode: {‘raise’, ‘wrap’, ‘clip’}, optional
                Specifies how out-of-bounds indices will behave.

                * ‘raise’ – raise an error (default)

                * ‘wrap’ – wrap around

                * ‘clip’ – clip to the range

                ‘clip’ mode means that all indices that are too large are replaced by the index
                that addresses the last element along that axis.
                Note that this disables indexing with negative numbers.

        Returns
            out: PhiTensor
                The returned array has the same type as a.
        """
        return self._apply_self_tensor_op("take", *args, **kwargs)

    def clip(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Clip (limit) the values in an array.

        Parameters
            a : array_like
                Array containing elements to clip.
            a_min, a_max : array_like or None
                Minimum and maximum value. If None, clipping is not performed on
                the corresponding edge. Only one of a_min and a_max may be
                None. Both are broadcast against a.
        Returns:
            Union[TensorWrappedPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return self._apply_self_tensor_op("clip", *args, **kwargs)

    def choose(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """
        Construct an array from an index array and a list of arrays to choose from.

        First of all, if confused or uncertain, definitely look at the Examples - in its full generality,
        this function is less simple than it might seem from the following code description
        (below ndi = numpy.lib.index_tricks):

        np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)]).

        But this omits some subtleties. Here is a fully general summary:

        Given an “index” array (a) of integers and a sequence of n arrays (choices), a and each choice array are first
        broadcast, as necessary, to arrays of a common shape; calling these Ba and Bchoices[i], i = 0,…,n-1 we have that
         necessarily, Ba.shape == Bchoices[i].shape for each i. Then, a new array with shape Ba.shape is created
         as follows:

            if mode='raise' (the default), then, first of all, each element of a (and thus Ba) must be in the range
            [0, n-1]; now, suppose that i (in that range) is the value at the (j0, j1, ..., jm) position in Ba -
            then the value at the same position in the new array is the value in Bchoices[i] at that same position;

            if mode='wrap', values in a (and thus Ba) may be any (signed) integer; modular arithmetic is used to map
            integers outside the range [0, n-1] back into that range; and then the new array is constructed as above;

            if mode='clip', values in a (and thus Ba) may be any (signed) integer; negative integers are mapped to 0;
            values greater than n-1 are mapped to n-1; and then the new array is constructed as above.

        Parameters

            choices: sequence of arrays

                Choice arrays. a and all of the choices must be broadcastable to the same shape. If choices is itself an
                 array (not recommended), then its outermost dimension (i.e., the one corresponding to choices.shape[0])
                  is taken as defining the “sequence”.

            out: array, optional

                If provided, the result will be inserted into this array. It should be of the appropriate shape and
                dtype. Note that out is always buffered if mode='raise'; use other modes for better performance.

            mode{‘raise’ (default), ‘wrap’, ‘clip’}, optional

                Specifies how indices outside [0, n-1] will be treated:

                        ‘raise’ : an exception is raised

                        ‘wrap’ : value becomes value mod n

                        ‘clip’ : values < 0 are mapped to 0, values > n-1 are mapped to n-1

        Returns
            merged_array: PhiTensor
                The merged result.

        Raises
            ValueError: shape mismatch
                If a and each choice array are not all broadcastable to the same shape.

        """
        return self._apply_self_tensor_op("choose", *args, **kwargs)

    @property
    def T(self) -> TensorWrappedGammaTensorPointer:
        # We always maintain a Tensor hierarchy Tensor ---> PT--> Actual Data
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.T"

        result = TensorWrappedGammaTensorPointer(
            client=self.client,
        )

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)

        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=[], kwargs={})

            # then we convert anything which isnt a pointer into a pointer
            pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
                args=downcast_args,
                kwargs=downcast_kwargs,
                client=self.client,
                gc_enabled=False,
            )

            cmd = GetOrSetPropertyAction(
                path=attr_path_and_name,
                id_at_location=result_id_at_location,
                address=self.client.address,
                _self=self,
                args=pointer_args,
                kwargs=pointer_kwargs,
                action=PropertyActions.GET,
                map_to_dyn=False,
            )
            self.client.send_immediate_msg_without_reply(msg=cmd)

        inherit_tags(
            attr_path_and_name=attr_path_and_name,
            result=result,
            self_obj=self,
            args=[],
            kwargs={},
        )

        result_public_shape = np.empty(self.public_shape).T.shape

        result.public_shape = result_public_shape
        result.public_dtype = self.public_dtype

        return result

    def to_local_object_without_private_data_child(self) -> GammaTensor:
        """Convert this pointer into a partial version of the GammaTensor but without
        any of the private data therein."""
        # relative
        from ..tensor import Tensor

        public_shape = getattr(self, "public_shape", None)
        public_dtype = getattr(self, "public_dtype", None)
        return Tensor(
            child=GammaTensor(
                child=FixedPrecisionTensor(value=None),  # TODO 0.7 fix this
                sources={},
                jax_op=None,
                is_linear=False,
            ),
            public_shape=public_shape,
            public_dtype=public_dtype,
        )


@implements(TensorWrappedGammaTensorPointer, np.zeros_like)
def zeros_like(
    tensor: TensorWrappedGammaTensorPointer,
    *args: Any,
    **kwargs: Any,
) -> TensorWrappedGammaTensorPointer:
    return tensor.zeros_like(*args, **kwargs)


@implements(TensorWrappedGammaTensorPointer, np.ones_like)
def ones_like(
    tensor: TensorWrappedGammaTensorPointer,
    *args: Any,
    **kwargs: Any,
) -> TensorWrappedGammaTensorPointer:
    return tensor.ones_like(*args, **kwargs)


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
@serializable(capnp_bytes=True)
class GammaTensor:
    """
    A differential privacy tensor that contains data belonging to atleast 2 or more unique data subjects.

    Attributes:
        child: jnp.array
            The private data itself.
        data_subjects: DataSubjectArray
            (DP Metadata) A custom NumPy class that keeps track of which data subjects contribute which datapoints in
            this tensor.
        min_vals: lazyrepeatarray
            (DP Metadata) A custom class that keeps track of (data-independent) minimum values for this tensor.
        max_vals: lazyrepeatarray
            (DP Metadata) A custom class that keeps track of (data-independent) maximum values for this tensor.
        func_str: str
            A string that will determine which function was used to build the current tensor.
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

    def _object2bytes(self) -> bytes:
        # TODO Tudor: fix this
        schema = get_capnp_schema(schema_file="gamma_tensor.capnp")

        gamma_tensor_struct: CapnpModule = schema.GammaTensor  # type: ignore
        gamma_msg = gamma_tensor_struct.new_message()
        # this is how we dispatch correct deserialization of bytes
        gamma_msg.magicHeader = serde_magic_header(type(self))

        # do we need to serde func? if so how?
        # what about the state dict?

        if isinstance(self.child, np.ndarray) or np.isscalar(self.child):
            chunk_bytes(capnp_serialize(np.array(self.child), to_bytes=True), "child", gamma_msg)  # type: ignore
            gamma_msg.isNumpy = True
        elif isinstance(self.child, jnp.ndarray):
            chunk_bytes(
                capnp_serialize(jax2numpy(self.child, self.child.dtype), to_bytes=True),
                "child",
                gamma_msg,
            )
            gamma_msg.isNumpy = True
        else:
            chunk_bytes(serialize(self.child, to_bytes=True), "child", gamma_msg)  # type: ignore
            gamma_msg.isNumpy = False

        gamma_msg.sources = serialize(self.sources, to_bytes=True)
        gamma_msg.isLinear = self.is_linear
        gamma_msg.id = self.id.to_string()
        gamma_msg.jaxOp = serialize(self.jax_op, to_bytes=True)

        # return gamma_msg.to_bytes_packed()
        return gamma_msg.to_bytes()

    @staticmethod
    def _bytes2object(buf: bytes) -> GammaTensor:
        # TODO Tudor: fix this
        schema = get_capnp_schema(schema_file="gamma_tensor.capnp")
        gamma_struct: CapnpModule = schema.GammaTensor  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        # capnp from_bytes is now a context
        with gamma_struct.from_bytes(
            buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        ) as gamma_msg:

            if gamma_msg.isNumpy:
                child = capnp_deserialize(
                    combine_bytes(gamma_msg.child), from_bytes=True
                )
            else:
                child = deserialize(combine_bytes(gamma_msg.child), from_bytes=True)

            state = deserialize(gamma_msg.sources, from_bytes=True)
            is_linear = gamma_msg.isLinear
            id_str = UID.from_string(gamma_msg.id)
            jax_op = deserialize(gamma_msg.jaxOp, from_bytes=True)

            return GammaTensor(
                child=child,
                is_linear=is_linear,
                sources=state,
                id=id_str,
                jax_op=jax_op,
            )
