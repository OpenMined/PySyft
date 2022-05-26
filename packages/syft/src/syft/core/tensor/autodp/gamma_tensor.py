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
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union

# third party
import flax
import jax
from jax import numpy as jnp
import numpy as np
from numpy.random import randint
from scipy.optimize import shgo

# relative
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs
from ....core.node.common.action.get_or_set_property_action import (
    GetOrSetPropertyAction,
)
from ....core.node.common.action.get_or_set_property_action import PropertyActions
from ....lib.numpy.array import capnp_deserialize
from ....lib.numpy.array import capnp_serialize
from ....lib.python.util import upcast
from ....util import inherit_tags
from ...adp.data_subject_ledger import DataSubjectLedger
from ...adp.data_subject_list import DataSubjectList
from ...adp.data_subject_list import liststrtonumpyutf8
from ...adp.data_subject_list import numpyutf8tolist
from ...adp.vectorized_publish import vectorized_publish
from ...common.serde.capnp import CapnpModule
from ...common.serde.capnp import get_capnp_schema
from ...common.serde.capnp import serde_magic_header
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize as serialize
from ...common.uid import UID
from ...node.abstract.node import AbstractNodeClient
from ...node.common.action.run_class_method_action import RunClassMethodAction
from ...pointer.pointer import Pointer
from ..config import DEFAULT_INT_NUMPY_TYPE
from ..fixed_precision_tensor import FixedPrecisionTensor
from ..lazy_repeat_array import compute_min_max
from ..lazy_repeat_array import lazyrepeatarray
from ..smpc import utils
from ..smpc.mpc_tensor import MPCTensor
from ..smpc.utils import TYPE_TO_RING_SIZE

if TYPE_CHECKING:
    # stdlib
    from dataclasses import dataclass
else:
    # third party
    from flax.struct import dataclass


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
        "data_subjects",
        "min_vals",
        "max_vals",
        "public_dtype",
        "public_shape",
    ]

    __serde_overrides__ = {
        "client": [lambda x: x.address, lambda y: y],
        "public_shape": [lambda x: x, lambda y: upcast(y)],
    }
    _exhausted = False
    is_enum = False

    def __init__(
        self,
        data_subjects: DataSubjectList,
        min_vals: np.typing.ArrayLike,
        max_vals: np.typing.ArrayLike,
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

        self.min_vals = min_vals
        self.max_vals = max_vals
        self.data_subjects = data_subjects
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
        return (
            self.synthetic.__repr__()
            + "\n\n (The data printed above is synthetic - it's an imitation of the real data.)"
        )

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

    def _apply_tensor_op(self, other: Any, op_str: str) -> Any:
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass
        # We always maintain a Tensor hierarchy Tensor ---> PT--> Actual Data
        attr_path_and_name = f"syft.core.tensor.tensor.Tensor.{op_str}"

        min_vals, max_vals = compute_min_max(
            self.min_vals, self.max_vals, other, op_str
        )
        result = TensorWrappedGammaTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
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
        result_public_dtype = self.public_dtype

        result.public_shape = result_public_shape
        result.public_dtype = result_public_dtype

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

    def __add__(
        self,
        other: Union[
            TensorWrappedGammaTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__add__")

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
        """Apply the "matmul" operation between "self" and "other"

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
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__lt__")

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
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__gt__")

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
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__le__")

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
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__eq__")

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
        return TensorWrappedGammaTensorPointer._apply_op(self, other, "__ne__")

    def concatenate(
        self,
        other: TensorWrappedGammaTensorPointer,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> MPCTensor:
        """Apply the "add" operation between "self" and "other"

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

    def sum(
        self,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.sum"
        min_vals, max_vals = compute_min_max(self.min_vals, self.max_vals, None, "sum")

        result = TensorWrappedGammaTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
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

        result.public_shape = np.array([1]).shape
        result.public_dtype = self.public_dtype

        return result

    def exp(
        self,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.exp"

        # TODO: should modify to log reduction.
        def exp_reduction(val: np.ndarray) -> np.ndarray:
            pos_index = val >= 0
            neg_index = val < 0
            exp = np.exp((pos_index * val * -1) + (neg_index * val))
            pos_values = (pos_index) * exp
            neg_values = (neg_index) * exp * -1
            return pos_values + neg_values

        min_vals = self.min_vals.copy()
        min_vals.data = np.array(exp_reduction(min_vals.data))
        max_vals = self.max_vals.copy()
        max_vals.data = np.array(exp_reduction(max_vals.data))

        result = TensorWrappedGammaTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
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

        result.public_shape = self.public_shape
        result.public_dtype = self.public_dtype

        return result

    def reciprocal(
        self,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "reciprocal" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.reciprocal"

        min_vals = self.min_vals.copy()
        min_vals.data = np.array(1 / min_vals.data)
        max_vals = self.max_vals.copy()
        max_vals.data = np.array(1 / max_vals.data)

        result = TensorWrappedGammaTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
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

        result.public_shape = self.public_shape
        result.public_dtype = self.public_dtype

        return result

    @property
    def T(self) -> TensorWrappedGammaTensorPointer:
        # We always maintain a Tensor hierarchy Tensor ---> PT--> Actual Data
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.T"

        result = TensorWrappedGammaTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=self.min_vals.transpose(),
            max_vals=self.max_vals.transpose(),
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

    def one_hot(self: TensorWrappedGammaTensorPointer) -> np.array:
        tensor_size = np.empty(self.public_shape).size
        one_hot_Y = np.zeros((tensor_size, self.max_vals.data[0] + 1))
        one_hot_Y = one_hot_Y.T

        attr_path_and_name = "syft.core.tensor.tensor.Tensor.one_hot"

        result = TensorWrappedGammaTensorPointer(
            data_subjects=self.data_subjects,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
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

        result.public_shape = one_hot_Y.shape
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
                child=FixedPrecisionTensor(value=None),
                data_subjects=self.data_subjects,
                min_val=self.min_vals,  # type: ignore
                max_val=self.max_vals,  # type: ignore
            ),
            public_shape=public_shape,
            public_dtype=public_dtype,
        )


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


def no_op(x: Dict[str, GammaTensor]) -> Dict[str, GammaTensor]:
    """A Private input will be initialized with this function.
    Whenever you manipulate a private input (i.e. add it to another private tensor),
    the result will have a different function. Thus we can check to see if the f
    """
    return x


def jax2numpy(value: jnp.array, dtype: np.dtype) -> np.array:
    # are we incurring copying here?
    return np.asarray(value, dtype=dtype)


def numpy2jax(value: np.array, dtype: np.dtype) -> jnp.array:
    return jnp.asarray(value, dtype=dtype)


@dataclass
@serializable(capnp_bytes=True)
class GammaTensor:
    PointerClassOverride = TensorWrappedGammaTensorPointer

    child: jnp.array
    data_subjects: DataSubjectList
    min_val: lazyrepeatarray = flax.struct.field(pytree_node=False)
    max_val: lazyrepeatarray = flax.struct.field(pytree_node=False)
    is_linear: bool = True
    func: Callable = flax.struct.field(pytree_node=False, default_factory=lambda: no_op)
    id: str = flax.struct.field(
        pytree_node=False, default_factory=lambda: str(randint(0, 2**31 - 1))
    )  # TODO: Need to check if there are any scenarios where this is not secure
    state: dict = flax.struct.field(pytree_node=False, default_factory=dict)
    fpt_values: Optional[FixedPrecisionTensor] = None

    def __post_init__(
        self,
    ) -> None:  # Might not serve any purpose anymore, since state trees are updated during ops
        if len(self.state) == 0 and self.func is not no_op:
            self.state[self.id] = self

        if not isinstance(self.child, FixedPrecisionTensor):
            # child = the actual private data
            self.child = FixedPrecisionTensor(self.child)

    def decode(self) -> np.ndarray:
        return self.child.decode()

    def run(self, state: dict) -> Callable:
        """This method traverses the computational tree and returns all the private inputs"""
        # TODO: Can we eliminate "state" and use self.state below?
        # we hit a private input
        if self.func is no_op:
            return self.decode()
        return self.func(state)

    def __add__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):

            def _add(state: dict) -> jax.numpy.DeviceArray:
                return jnp.add(self.run(state), other.run(state))

            # print("this is the other.state", other.state)
            output_state[other.id] = other
            # state.update(other.state)
            # print("this is the output_state", output_state)

            child = self.child + other.child
            min_val = self.min_val + other.min_val
            max_val = self.max_val + other.max_val
        else:

            def _add(state: dict) -> jax.numpy.DeviceArray:
                return jnp.add(self.run(state), other)

            child = self.child + other
            min_val = self.min_val + other
            max_val = self.max_val + other
        # print("the state we returned is: ", output_state)
        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_add,
            state=output_state,
        )

    def __sub__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):

            def _sub(state: dict) -> jax.numpy.DeviceArray:
                return jnp.subtract(self.run(state), other.run(state))

            # print("this is the other.state", other.state)
            output_state[other.id] = other
            # state.update(other.state)
            # print("this is the output_state", output_state)

            child = self.child - other.child
            min_min = self.min_val.data - other.min_val.data
            min_max = self.min_val.data - other.max_val.data
            max_min = self.max_val.data - other.min_val.data
            max_max = self.max_val.data - other.max_val.data
            _min_val = np.minimum.reduce([min_min, min_max, max_min, max_max])
            _max_val = np.maximum.reduce([min_min, min_max, max_min, max_max])
            min_val = self.min_val.copy()
            min_val.data = _min_val
            max_val = self.max_val.copy()
            max_val.data = _max_val

        else:

            def _sub(state: dict) -> jax.numpy.DeviceArray:
                return jnp.subtract(self.run(state), other)

            child = self.child - other
            min_val = self.min_val - other
            max_val = self.max_val - other
        # print("the state we returned is: ", output_state)
        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_sub,
            state=output_state,
        )

    def __mul__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):

            def _mul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.multiply(self.run(state), other.run(state))

            output_state[other.id] = other
            child = self.child * other.child
            min_min = self.min_val.data * other.min_val.data
            min_max = self.min_val.data * other.max_val.data
            max_min = self.max_val.data * other.min_val.data
            max_max = self.max_val.data * other.max_val.data
            _min_val = np.array(np.min([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            _max_val = np.array(np.max([min_min, min_max, max_min, max_max], axis=0))  # type: ignore

        else:

            def _mul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.multiply(self.run(state), other)

            child = self.child * other
            min_min = self.min_val.data * other
            min_max = self.min_val.data * other
            max_min = self.max_val.data * other
            max_max = self.max_val.data * other
            _min_val = np.array(np.min([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            _max_val = np.array(np.max([min_min, min_max, max_min, max_max], axis=0))  # type: ignore

        min_val = self.min_val.copy()
        min_val.data = _min_val
        max_val = self.max_val.copy()
        max_val.data = _max_val

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_mul,
            state=output_state,
        )

    def __matmul__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):

            def _matmul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.matmul(self.run(state), other.run(state))

            output_state[other.id] = other
            child = self.child @ other.child
            min_val = self.min_val.__matmul__(other.min_val)
            max_val = self.max_val.__matmul__(other.max_val)

        else:

            def _matmul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.matmul(self.run(state), other)

            child = self.child @ other
            min_val = self.min_val.__matmul__(other)
            max_val = self.max_val.__matmul__(other)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_matmul,
            state=output_state,
        )

    def __rmatmul__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):

            def _rmatmul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.matmul(
                    other.run(state),
                    self.run(state),
                )

            output_state[other.id] = other
            child = self.child.__rmatmul__(other.child)
            min_val = self.min_val.__rmatmul__(other.min_val)
            max_val = self.max_val.__rmatmul__(other.max_val)

        else:

            def _rmatmul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.matmul(other, self.run(state))

            child = self.child.__rmatmul__(other)
            min_val = self.min_val.__rmatmul__(other)
            max_val = self.max_val.__rmatmul__(other)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_rmatmul,
            state=output_state,
        )

    def __gt__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):

            def _gt(state: dict) -> jax.numpy.DeviceArray:
                return jnp.greater(self.run(state), other.run(state))

            output_state[other.id] = other
            child = self.child.__gt__(other.child)

        else:

            def _gt(state: dict) -> jax.numpy.DeviceArray:
                return jnp.greater(self.run(state), other)

            child = self.child.__gt__(other)

        min_val = self.min_val * 0
        max_val = (self.max_val * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_gt,
            state=output_state,
        )

    def exp(self) -> GammaTensor:
        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        # relative
        from ...smpc.approximations import exp

        def exp_reduction(val: np.ndarray) -> np.ndarray:
            pos_index = val >= 0
            neg_index = val < 0
            exp = np.exp((pos_index * val * -1) + (neg_index * val))
            pos_values = (pos_index) * exp
            neg_values = (neg_index) * exp * -1
            return pos_values + neg_values

        min_val = self.min_val.copy()
        min_val.data = np.array(exp_reduction(min_val.data))
        max_val = self.max_val.copy()
        max_val.data = np.array(exp_reduction(max_val.data))

        def _exp(state: dict) -> jax.numpy.DeviceArray:
            return jnp.log(self.run(state))

        return GammaTensor(
            child=exp(self.child),
            min_val=min_val,
            max_val=max_val,
            data_subjects=self.data_subjects,
            func=_exp,
            state=output_state,
        )

    def reciprocal(self) -> GammaTensor:
        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        # relative
        from ...smpc.approximations import reciprocal

        min_val = self.min_val.copy()
        min_val.data = np.array(1 / (min_val.data))
        max_val = self.max_val.copy()
        max_val.data = np.array(1 / (max_val.data))

        def _reciprocal(state: dict) -> jax.numpy.DeviceArray:
            return jnp.divide(1, self.run(state))

        # TODO: Explore why overflow does not occur for arrays
        fpt = self.child.copy()
        if hasattr(fpt.child, "shape") and fpt.child.shape == ():
            fpt.child = np.expand_dims(fpt.child, 0)

        child_inv = reciprocal(fpt)

        if hasattr(self.child.child, "shape") and self.child.child.shape == ():
            child_inv.child = np.squeeze(child_inv.child)

        return GammaTensor(
            child=child_inv,
            min_val=min_val,
            max_val=max_val,
            data_subjects=self.data_subjects,
            func=_reciprocal,
            state=output_state,
        )

    def transpose(self, *args: Any, **kwargs: Any) -> GammaTensor:
        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        def _transpose(state: dict) -> jax.numpy.DeviceArray:
            return jnp.transpose(self.run(state))

        return GammaTensor(
            child=self.child.transpose(),
            data_subjects=self.data_subjects,
            min_val=self.min_val.transpose(),
            max_val=self.max_val.transpose(),
            func=_transpose,
            state=output_state,
        )

    def sum(self, *args: Tuple[Any, ...], **kwargs: Any) -> GammaTensor:
        def _sum(state: dict) -> jax.numpy.DeviceArray:
            return jnp.sum(self.run(state))

        output_state = dict()
        output_state[self.id] = self
        # output_state.update(self.state)

        child = self.child.sum()
        # Change sum before Merge
        min_val, max_val = compute_min_max(self.min_val, self.max_val, None, "sum")

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_sum,
            state=output_state,
        )

    def sqrt(self) -> GammaTensor:
        def _sqrt(state: dict) -> jax.numpy.DeviceArray:
            return jnp.sqrt(self.run(state))

        state = dict()
        state.update(self.state)

        child = jnp.sqrt(self.child)
        min_val = jnp.sqrt(self.min_val)
        max_val = jnp.sqrt(self.max_val)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_sqrt,
            state=state,
        )

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: Optional[float] = None,
    ) -> jax.numpy.DeviceArray:
        # TODO: Add data scientist privacy budget as an input argument, and pass it
        # into vectorized_publish
        if sigma is None:
            sigma = self.child.mean() / 4  # TODO @Ishan: replace this with calibration

        if self.child.dtype != np.int64:
            raise Exception(
                "Data type of private values is not np.int64: ", self.child.dtype
            )
        fpt_values = self.fpt_values
        fpt_encode_func = None  # Function for encoding noise
        if fpt_values is not None:
            fpt_encode_func = fpt_values.encode

        if (
            not self.state
        ):  # if state tree is empty (e.g. publishing a PhiTensor w/ public vals directly)
            self.state[self.id] = self

        return vectorized_publish(
            min_vals=self.min_val,
            max_vals=self.max_val,
            state_tree=self.state,
            data_subjects=self.data_subjects,
            is_linear=self.is_linear,
            sigma=sigma,
            output_func=self.func,
            ledger=ledger,
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            fpt_encode_func=fpt_encode_func,
        )

    def expand_dims(self, axis: int) -> GammaTensor:
        def _expand_dims(state: dict) -> jax.numpy.DeviceArray:
            return jnp.expand_dims(self.run(state), axis)

        state = dict()
        state.update(self.state)

        return GammaTensor(
            child=jnp.expand_dims(self.child, axis),
            data_subjects=self.data_subjects,
            min_val=self.min_val,
            max_val=self.max_val,
            func=_expand_dims,
            state=state,
        )

    def squeeze(self, axis: Optional[int] = None) -> GammaTensor:
        def _squeeze(state: dict) -> jax.numpy.DeviceArray:
            return jnp.squeeze(self.run(state), axis)

        state = dict()
        state.update(self.state)
        return GammaTensor(
            child=jnp.squeeze(self.child, axis),
            data_subjects=self.data_subjects,
            min_val=self.min_val,
            max_val=self.max_val,
            func=_squeeze,
            state=state,
        )

    def __len__(self) -> int:
        return len(self.child)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.child.shape

    @property
    def lipschitz_bound(self) -> float:
        # TODO: Check if there are any functions for which lipschitz bounds shouldn't be computed
        # if dis(self.func) == dis(no_op):
        #     raise Exception

        print("Starting JAX JIT")
        fn = jax.jit(self.func)
        print("Traced self.func with jax's jit, now calculating gradient")
        grad_fn = jax.grad(fn)
        print("Obtained gradient, creating lookup tables")
        i2k, k2i, i2v, i2s = create_new_lookup_tables(self.state)

        print("created lookup tables, now getting bounds")
        i2minval = jnp.concatenate([x for x in i2v]).reshape(-1, 1)
        i2maxval = jnp.concatenate([x for x in i2v]).reshape(-1, 1)
        bounds = jnp.concatenate([i2minval, i2maxval], axis=1)
        print("Obtained bounds")
        # sample_input = i2minval.reshape(-1)
        _ = i2minval.reshape(-1)
        print("Obtained all inputs")

        def max_grad_fn(input_values: np.ndarray) -> float:
            vectors = {}
            n = 0
            for i, size_param in enumerate(i2s):
                vectors[i2k[i]] = input_values[n : n + size_param]  # noqa: E203
                n += size_param

            grad_pred = grad_fn(vectors)

            m = 0
            for value in grad_pred.values():
                m = max(m, jnp.max(value))

            # return negative because we want to maximize instead of minimize
            return -m

        print("starting SHGO")
        res = shgo(max_grad_fn, bounds, iters=1, constraints=tuple())
        print("Ran SHGO")
        # return negative because we flipped earlier
        return -float(res.fun)

    @property
    def dtype(self) -> np.dtype:
        return self.child.dtype

    @staticmethod
    def get_input_tensors(state_tree: dict[int, GammaTensor]) -> List:
        # TODO: See if we can call np.stack on the output and create a vectorized tensor instead of a list of tensors
        input_tensors = []
        for tensor in state_tree.values():
            if tensor.func is no_op:
                input_tensors.append(tensor)
            else:
                input_tensors += GammaTensor.get_input_tensors(tensor.state)
        return input_tensors

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="gamma_tensor.capnp")

        gamma_tensor_struct: CapnpModule = schema.GammaTensor  # type: ignore
        gamma_msg = gamma_tensor_struct.new_message()
        # this is how we dispatch correct deserialization of bytes
        gamma_msg.magicHeader = serde_magic_header(type(self))

        # what is the difference between inputs and value which do we serde
        # do we need to serde func? if so how?
        # what about the state dict?
        gamma_msg.child = serialize(self.child, to_bytes=True)
        gamma_msg.state = serialize(self.state, to_bytes=True)
        gamma_msg.dataSubjectsIndexed = capnp_serialize(
            self.data_subjects.data_subjects_indexed
        )
        gamma_msg.oneHotLookup = capnp_serialize(
            liststrtonumpyutf8(self.data_subjects.one_hot_lookup)
        )
        gamma_msg.minVal = serialize(self.min_val, to_bytes=True)
        gamma_msg.maxVal = serialize(self.max_val, to_bytes=True)
        gamma_msg.isLinear = self.is_linear
        gamma_msg.id = self.id

        # return gamma_msg.to_bytes_packed()
        return gamma_msg.to_bytes()

    @staticmethod
    def _bytes2object(buf: bytes) -> GammaTensor:
        schema = get_capnp_schema(schema_file="gamma_tensor.capnp")
        gamma_struct: CapnpModule = schema.GammaTensor  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        # capnp from_bytes is now a context
        with gamma_struct.from_bytes(
            buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        ) as gamma_msg:
            child = deserialize(gamma_msg.child, from_bytes=True)
            state = deserialize(gamma_msg.state, from_bytes=True)
            data_subjects_indexed = capnp_deserialize(gamma_msg.dataSubjectsIndexed)
            one_hot_lookup = numpyutf8tolist(capnp_deserialize(gamma_msg.oneHotLookup))
            data_subjects = DataSubjectList(one_hot_lookup, data_subjects_indexed)
            min_val = deserialize(gamma_msg.minVal, from_bytes=True)
            max_val = deserialize(gamma_msg.maxVal, from_bytes=True)
            is_linear = gamma_msg.isLinear
            id_str = gamma_msg.id

            return GammaTensor(
                child=child,
                data_subjects=data_subjects,
                min_val=min_val,
                max_val=max_val,
                is_linear=is_linear,
                state=state,
                id=id_str,
            )
