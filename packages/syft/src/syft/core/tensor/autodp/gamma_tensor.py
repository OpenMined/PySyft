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
from numpy.typing import NDArray
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
from ...adp.data_subject_list import dslarraytonumpyutf8
from ...adp.data_subject_list import numpyutf8todslarray
from ...adp.vectorized_publish import vectorized_publish
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
from ...pointer.pointer import Pointer
from ..config import DEFAULT_INT_NUMPY_TYPE
from ..fixed_precision_tensor import FixedPrecisionTensor
from ..lazy_repeat_array import compute_min_max
from ..lazy_repeat_array import lazyrepeatarray
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from ..smpc import utils
from ..smpc.mpc_tensor import MPCTensor
from ..smpc.utils import TYPE_TO_RING_SIZE
from ..util import implements

if TYPE_CHECKING:
    # stdlib
    from dataclasses import dataclass
else:
    # third party
    from flax.struct import dataclass


@serializable(recursive_serde=True)
class TensorWrappedGammaTensorPointer(Pointer, PassthroughTensor):
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
        "data_subjects": [dslarraytonumpyutf8, numpyutf8todslarray],
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
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "truediv" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.sum"
        min_vals = self.min_vals.sum(*args, **kwargs)
        max_vals = self.max_vals.sum(*args, **kwargs)

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
        dummy_res = np.empty(self.public_shape).sum(*args, **kwargs)
        result.public_shape = dummy_res.shape
        result.public_dtype = self.public_dtype

        return result

    def __getitem__(
        self, key: Union[int, bool, slice]
    ) -> TensorWrappedGammaTensorPointer:
        """Apply the slice  operation on "self"
        Args:
            y (Union[int,bool,slice]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.__getitem__"
        result: TensorWrappedGammaTensorPointer
        min_vals = self.min_vals.__getitem__(key)
        max_vals = self.max_vals.__getitem__(key)

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
            ) = lib.python.util.downcast_args_and_kwargs(args=[key], kwargs={})

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
            args=[key],
            kwargs={},
        )
        dummy_res = np.empty(self.public_shape).__getitem__(key)
        result.public_shape = dummy_res.shape
        result.public_dtype = self.public_dtype

        return result

    def ones_like(
        self,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the "ones like" operation on self"

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.ones_like"
        min_vals = self.min_vals.ones_like(*args, **kwargs)
        max_vals = self.max_vals.ones_like(*args, **kwargs)

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
        dummy_res = np.ones_like(np.empty(self.public_shape), *args, **kwargs)
        result.public_shape = dummy_res.shape
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

    def softmax(
        self,
    ) -> Union[TensorWrappedGammaTensorPointer, MPCTensor]:
        """Apply the softmax operation on self

        Args:
            y (Union[TensorWrappedGammaTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedGammaTensorPointer,MPCTensor] : Result of the operation.
        """
        attr_path_and_name = "syft.core.tensor.tensor.Tensor.softmax"

        # TODO: should modify to log reduction.
        def softmax(val: np.ndarray) -> np.ndarray:
            logits = val - val.max()
            numerator = np.exp(logits)
            inv = 1 / numerator.sum()
            return numerator * inv

        min_vals = self.min_vals.copy()
        min_vals.data = np.array(softmax(min_vals.data))
        max_vals = self.max_vals.copy()
        max_vals.data = np.array(softmax(max_vals.data))

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


@implements(TensorWrappedGammaTensorPointer, np.ones_like)
def ones_like(
    tensor: TensorWrappedGammaTensorPointer,
    *args: Tuple[Any, ...],
    **kwargs: Dict[Any, Any],
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


def no_op(x: GammaTensor) -> GammaTensor:
    """A Private input will be initialized with this function.
    Whenever you manipulate a private input (i.e. add it to another private tensor),
    the result will have a different function. Thus we can check to see if the f
    """
    res = x
    if isinstance(x, GammaTensor) and isinstance(x.data_subjects, np.ndarray):
        res = GammaTensor(
            child=x.child,
            data_subjects=np.zeros_like(x.data_subjects, np.int64),
            min_vals=x.min_vals,
            max_vals=x.max_vals,
            func=x.func,
            state=GammaTensor.convert_dsl(x.state),
        )
    return res


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
    data_subjects: np.ndarray
    min_vals: Union[lazyrepeatarray, np.ndarray] = flax.struct.field(pytree_node=False)
    max_vals: Union[lazyrepeatarray, np.ndarray] = flax.struct.field(pytree_node=False)
    is_linear: bool = True
    func: Callable = flax.struct.field(pytree_node=False, default_factory=lambda: no_op)
    id: str = flax.struct.field(
        pytree_node=False, default_factory=lambda: str(randint(0, 2**31 - 1))
    )  # TODO: Need to check if there are any scenarios where this is not secure
    state: dict = flax.struct.field(pytree_node=False, default_factory=dict)

    def __post_init__(
        self,
    ) -> None:  # Might not serve any purpose anymore, since state trees are updated during ops
        if self.state and len(self.state) == 0 and self.func is not no_op:
            self.state[self.id] = self

        # if not isinstance(self.min_vals, lazyrepeatarray):
        #     self.min_vals = lazyrepeatarray(data=self.min_vals, shape=self.shape)
        #     self.max_vals = lazyrepeatarray(data=self.max_vals, shape=self.shape)

        if isinstance(self.min_vals, lazyrepeatarray):
            if self.min_vals.data.size != 1:
                self.min_vals.data = self.min_vals.data.min()
                self.max_vals.data = self.max_vals.data.max()

    def decode(self) -> np.ndarray:
        if isinstance(self.child, FixedPrecisionTensor):
            return self.child.decode()
        else:
            return self.child

    def run(self, state: dict) -> Union[Callable, GammaTensor]:
        """This method traverses the computational tree and returns all the private inputs"""
        # TODO: Can we eliminate "state" and use self.state below?
        # we hit a private input
        if self.func is no_op:
            return self  # .decode()
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

            output_state[other.id] = other
            # state.update(other.state)

            child = self.child + other.child
            min_val = self.min_vals + other.min_vals
            max_val = self.max_vals + other.max_vals
            output_ds = self.data_subjects + other.data_subjects

        else:

            def _add(state: dict) -> jax.numpy.DeviceArray:
                return jnp.add(self.run(state), other)

            child = self.child + other
            min_val = self.min_vals + other
            max_val = self.max_vals + other
            output_ds = self.data_subjects + other

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func=_add,
            state=output_state,
        )

    def __rtruediv__(self, other: SupportedChainType) -> GammaTensor:

        if is_acceptable_simple_type(other):
            return GammaTensor(
                child=(1 / self.child) * other,
                min_vals=(1 / self.min_vals) * other,
                max_vals=(1 / self.max_vals) * other,
                data_subjects=(1 / self.data_subjects) * other,
                # scalar_manager=self.scalar_manager,
            )
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

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

            output_state[other.id] = other

            child = self.child - other.child
            min_min = self.min_vals.data - other.min_vals.data
            min_max = self.min_vals.data - other.max_vals.data
            max_min = self.max_vals.data - other.min_vals.data
            max_max = self.max_vals.data - other.max_vals.data
            _min_val = np.minimum.reduce([min_min, min_max, max_min, max_max])
            _max_val = np.maximum.reduce([min_min, min_max, max_min, max_max])
            min_val = self.min_vals.copy()
            min_val.data = _min_val
            max_val = self.max_vals.copy()
            max_val.data = _max_val

            output_ds = self.data_subjects - other.data_subjects

        else:

            def _sub(state: dict) -> jax.numpy.DeviceArray:
                return jnp.subtract(self.run(state), other)

            child = self.child - other
            min_val = self.min_vals - other
            max_val = self.max_vals - other
            output_ds = self.data_subjects - other

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
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
            min_min = self.min_vals.data * other.min_vals.data
            min_max = self.min_vals.data * other.max_vals.data
            max_min = self.max_vals.data * other.min_vals.data
            max_max = self.max_vals.data * other.max_vals.data
            _min_val = np.array(np.min([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            _max_val = np.array(np.max([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            output_ds = self.data_subjects * other.data_subjects

        else:

            def _mul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.multiply(self.run(state), other)

            child = self.child * other
            min_min = self.min_vals.data * other
            min_max = self.min_vals.data * other
            max_min = self.max_vals.data * other
            max_max = self.max_vals.data * other
            _min_val = np.array(np.min([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            _max_val = np.array(np.max([min_min, min_max, max_min, max_max], axis=0))  # type: ignore
            output_ds = self.data_subjects * other

        min_val = self.min_vals.copy()
        min_val.data = _min_val
        max_val = self.max_vals.copy()
        max_val.data = _max_val

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
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
            min_val = self.min_vals.__matmul__(other.min_vals)
            max_val = self.max_vals.__matmul__(other.max_vals)
            output_ds = self.data_subjects @ other.data_subjects

        else:

            def _matmul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.matmul(self.run(state), other)

            child = self.child @ other
            min_val = self.min_vals.__matmul__(other)
            max_val = self.max_vals.__matmul__(other)

            output_ds = self.data_subjects @ other

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
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
            min_val = self.min_vals.__rmatmul__(other.min_vals)
            max_val = self.max_vals.__rmatmul__(other.max_vals)
            output_ds = self.data_subjects.__rmatmul__(other.data_subjects)

        else:

            def _rmatmul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.matmul(other, self.run(state))

            child = self.child.__rmatmul__(other)
            min_val = self.min_vals.__rmatmul__(other)
            max_val = self.max_vals.__rmatmul__(other)
            output_ds = self.data_subjects.__rmatmul__(other)

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
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
            output_ds = self.data_subjects > other.data_subjects

        else:

            def _gt(state: dict) -> jax.numpy.DeviceArray:
                return jnp.greater(self.run(state), other)

            child = self.child.__gt__(other)
            output_ds = self.data_subjects.__gt__(other)

        min_val = self.min_vals * 0
        max_val = (self.max_vals * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func=_gt,
            state=output_state,
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GammaTensor):
            return (self.child == other.child).all() and (
                self.data_subjects == other.data_subjects
            ).all()

        return False

    def __lt__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):

            def _lt(state: dict) -> jax.numpy.DeviceArray:
                return jnp.less(self.run(state), other.run(state))

            output_state[other.id] = other
            child = self.child.__lt__(other.child)
            output_ds = self.data_subjects > other.data_subjects

        else:

            def _lt(state: dict) -> jax.numpy.DeviceArray:
                return jnp.greater(self.run(state), other)

            child = self.child.__lt__(other)
            output_ds = self.data_subjects.__lt__(other)

        min_val = self.min_vals * 0
        max_val = (self.max_vals * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func=_lt,
            state=output_state,
        )

    def __le__(self, other: Any) -> GammaTensor:
        # relative
        from .phi_tensor import PhiTensor

        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        if isinstance(other, PhiTensor):
            other = other.gamma

        if isinstance(other, GammaTensor):

            def _le(state: dict) -> jax.numpy.DeviceArray:
                return jnp.less_equal(self.run(state), other.run(state))

            output_state[other.id] = other
            child = self.child.__le__(other.child)
            output_ds = self.data_subjects <= other.data_subjects

        else:

            def _le(state: dict) -> jax.numpy.DeviceArray:
                return jnp.less_equal(self.run(state), other)

            child = self.child.__le__(other)
            output_ds = self.data_subjects

        min_val = self.min_vals * 0
        max_val = (self.max_vals * 0) + 1

        return GammaTensor(
            child=child,
            data_subjects=output_ds,
            min_vals=min_val,
            max_vals=max_val,
            func=_le,
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

        min_val = self.min_vals.copy()
        min_val.data = np.array(exp_reduction(min_val.data))
        max_val = self.max_vals.copy()
        max_val.data = np.array(exp_reduction(max_val.data))

        def _exp(state: dict) -> jax.numpy.DeviceArray:
            return jnp.log(self.run(state))

        return GammaTensor(
            child=exp(self.child),
            min_vals=min_val,
            max_vals=max_val,
            data_subjects=np.exp(self.data_subjects),
            func=_exp,
            state=output_state,
        )

    def log(self) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        if isinstance(self.min_vals, lazyrepeatarray):
            min_val = lazyrepeatarray(
                data=np.log(self.min_vals.data.min()), shape=self.shape
            )
            max_val = lazyrepeatarray(
                data=np.log(self.max_vals.data.max()), shape=self.shape
            )
        elif isinstance(self.min_vals, np.ndarray):
            min_val = lazyrepeatarray(data=np.log(self.min_vals), shape=self.shape)
            max_val = lazyrepeatarray(data=np.log(self.max_vals), shape=self.shape)
        elif isinstance(self.min_vals, (int, float)):
            min_val = lazyrepeatarray(data=np.log(self.min_vals), shape=self.shape)
            max_val = lazyrepeatarray(data=np.log(self.max_vals), shape=self.shape)
            # min_vals = np.log(self.min_vals),
            # max_vals = np.log(self.max_vals)
        else:
            raise NotImplementedError(
                f"Undefined behaviour for type: {type(self.min_vals)}"
            )

        def _log(state: dict) -> jax.numpy.DeviceArray:
            return jnp.log(self.run(state))

        return GammaTensor(
            child=np.log(self.child),
            min_vals=min_val,
            max_vals=max_val,
            data_subjects=np.log(self.data_subjects),
            func=_log,
            state=output_state,
        )

    def reciprocal(self) -> GammaTensor:
        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        # relative
        from ...smpc.approximations import reciprocal

        min_val = self.min_vals.copy()
        min_val.data = np.array(1 / (min_val.data))
        max_val = self.max_vals.copy()
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
            min_vals=min_val,
            max_vals=max_val,
            data_subjects=self.data_subjects,
            func=_reciprocal,
            state=output_state,
        )

    def softmax(self) -> GammaTensor:
        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        # relative
        from ...smpc.approximations import exp
        from ...smpc.approximations import reciprocal

        def softmax(val: np.ndarray) -> np.ndarray:
            logits = val - val.max()
            numerator = np.exp(logits)
            inv = 1 / numerator.sum()
            return numerator * inv

        min_val = self.min_vals.copy()
        min_val.data = np.array(softmax(min_val.data))
        max_val = self.max_vals.copy()
        max_val.data = np.array(softmax(max_val.data))
        fpt = self.child.copy()
        if not isinstance(fpt.child, np.ndarray):
            raise ValueError("Softmax currently works only for numpy child")

        fpt.child = fpt.child - fpt.child.max()
        numerator = exp(fpt)
        inv = reciprocal(numerator.sum())  # type: ignore

        def _softmax(state: dict) -> jax.numpy.DeviceArray:
            return jnp.exp(self.run(state)) / jnp.exp(self.run(state)).sum()

        return GammaTensor(
            child=numerator * inv,
            min_vals=min_val,
            max_vals=max_val,
            data_subjects=self.data_subjects,
            func=_softmax,
            state=output_state,
        )

    def transpose(self, *args: Any, **kwargs: Any) -> GammaTensor:
        output_state = dict()
        # Add this tensor to the chain
        output_state[self.id] = self

        def _transpose(state: dict) -> jax.numpy.DeviceArray:
            return jnp.transpose(self.run(state))

        output_ds = self.data_subjects.transpose(*args)
        output_data = self.child.transpose(*args)

        min_vals = lazyrepeatarray(data=output_data.min(), shape=output_data.shape)
        max_vals = lazyrepeatarray(data=output_data.max(), shape=output_data.shape)

        return GammaTensor(
            child=output_data,
            data_subjects=output_ds,
            min_vals=min_vals,
            max_vals=max_vals,
            func=_transpose,
            state=output_state,
        )

    @property
    def T(self) -> GammaTensor:
        return self.transpose()

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, **kwargs: Any
    ) -> GammaTensor:
        def _sum(state: dict) -> jax.numpy.DeviceArray:
            return jnp.sum(self.run(state).child)  # type: ignore

        output_state = dict()
        output_state[self.id] = self

        child = self.child.sum(axis=axis, **kwargs)

        min_v = child.min()
        max_v = child.max()

        return GammaTensor(
            child=child,
            data_subjects=np.array(self.data_subjects.sum(axis=axis, **kwargs)),
            min_vals=lazyrepeatarray(data=min_v, shape=child.shape),
            max_vals=lazyrepeatarray(data=max_v, shape=child.shape),
            func=_sum,
            state=output_state,
        )

    def ones_like(self, *args: Tuple[Any, ...], **kwargs: Any) -> GammaTensor:
        def _ones_like(state: dict) -> jax.numpy.DeviceArray:
            return jnp.ones_like(self.run(state))

        output_state = dict()
        output_state[self.id] = self
        # output_state.update(self.state)

        child = (
            np.ones_like(self.child, *args, **kwargs)
            if isinstance(self.child, np.ndarray)
            else self.child.ones_like(*args, **kwargs)
        )

        min_val = self.min_vals.ones_like(*args, **kwargs)
        max_val = self.max_vals.ones_like(*args, **kwargs)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func=_ones_like,
            state=output_state,
        )

    def zeros_like(self, *args: Tuple[Any, ...], **kwargs: Any) -> GammaTensor:
        def _zeros_like(state: dict) -> jax.numpy.DeviceArray:
            return jnp.zeros_like(self.run(state))

        output_state = dict()
        output_state[self.id] = self
        # output_state.update(self.state)

        child = (
            np.zeros_like(self.child, *args, **kwargs)
            if isinstance(self.child, np.ndarray)
            else self.child.zeros_like(*args, **kwargs)
        )

        min_val = self.min_vals.zeros_like(*args, **kwargs)
        max_val = self.max_vals.zeros_like(*args, **kwargs)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func=_zeros_like,
            state=output_state,
        )

    def ravel(self) -> GammaTensor:
        def _ravel(state: dict) -> jax.numpy.DeviceArray:
            return jnp.ravel(self.run(state))

        output_state = dict()
        output_state[self.id] = self
        # output_state.update(self.state)

        data = self.child
        output_data = data.ravel()

        output_data_subjects = self.data_subjects.ravel()

        min_val = lazyrepeatarray(data=self.min_vals.data, shape=output_data.shape)
        max_val = lazyrepeatarray(data=self.max_vals.data, shape=output_data.shape)

        return GammaTensor(
            child=output_data,
            data_subjects=output_data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func=_ravel,
            state=output_state,
        )

    def reshape(self, shape: Tuple[int, ...]) -> GammaTensor:
        child = self.child.reshape(shape)
        output_shape = child.shape

        if isinstance(self.min_vals, lazyrepeatarray):
            if self.min_vals.data.shape == 1:
                minv = self.min_vals.reshape(output_shape)
                maxv = self.max_vals.reshape(output_shape)
            elif self.min_vals.data.shape == self.min_vals.shape:
                minv = self.min_vals.reshape(output_shape)
                minv.data = minv.data.min()

                maxv = self.max_vals.reshape(output_shape)
                maxv.data = maxv.data.max()
            else:
                minv = self.min_vals.reshape(output_shape)
                minv.data = minv.data.min()

                maxv = self.max_vals.reshape(output_shape)
                maxv.data = maxv.data.max()

        elif isinstance(self.min_vals, (int, float)):
            minv = self.min_vals  # type: ignore
            maxv = self.max_vals
        else:
            minv = self.min_vals
            maxv = self.max_vals

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects.reshape(shape),
            min_vals=minv,
            max_vals=maxv,
        )

    def _argmax(self, axis: Optional[int]) -> np.ndarray:
        return self.child.argmax(axis)

    def mean(self, axis: Union[int, Tuple[int, ...]], **kwargs: Any) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        def _mean(
            state: dict, axis: Union[int, Tuple[int, ...]] = axis
        ) -> jax.numpy.DeviceArray:
            return jnp.mean(self.run(state), axis)

        result = self.child.mean(axis, **kwargs)
        minv = (
            self.min_vals.data
            if isinstance(self.min_vals, lazyrepeatarray)
            else self.min_vals
        )
        maxv = (
            self.max_vals.data
            if isinstance(self.max_vals, lazyrepeatarray)
            else self.max_vals
        )
        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.mean(axis, **kwargs),
            min_vals=lazyrepeatarray(data=minv, shape=result.shape),
            max_vals=lazyrepeatarray(data=(maxv + minv) / 2, shape=result.shape),
            state=output_state,
            func=_mean,
        )

    def expand_dims(self, axis: Optional[int] = None) -> GammaTensor:

        result = np.expand_dims(self.child, axis)

        target_shape_dsl = list(self.data_subjects.shape)
        if axis:
            target_shape_dsl.insert(axis + 1, 1)

        return GammaTensor(
            child=result,
            data_subjects=np.expand_dims(self.data_subjects, axis),
            min_vals=lazyrepeatarray(data=self.min_vals.data, shape=result.shape),
            max_vals=lazyrepeatarray(data=self.max_vals.data, shape=result.shape),
        )

    def std(self, axis: Union[int, Tuple[int, ...]], **kwargs: Any) -> GammaTensor:
        output_state = dict()
        output_state[self.id] = self

        def _std(
            state: dict, axis: Union[int, Tuple[int, ...]] = axis
        ) -> jax.numpy.DeviceArray:
            return jnp.std(self.run(state), axis)

        result = self.child.std(axis, **kwargs)
        minv = (
            self.min_vals.data
            if isinstance(self.min_vals, lazyrepeatarray)
            else self.min_vals
        )
        maxv = (
            self.max_vals.data
            if isinstance(self.max_vals, lazyrepeatarray)
            else self.max_vals
        )
        return GammaTensor(
            child=result,
            data_subjects=self.data_subjects.std(axis, **kwargs),
            min_vals=lazyrepeatarray(data=0, shape=result.shape),
            max_vals=lazyrepeatarray(
                data=0.25 * (maxv + minv) ** 2, shape=result.shape
            ),
            state=output_state,
            func=_std,
        )

    def dot(self, other: Union[np.ndarray, GammaTensor]) -> GammaTensor:
        # TODO: These bounds might not be super tight- if min,max = [-1, 1], there might be a dot product
        # such that the minimum value should be 0
        if isinstance(other, np.ndarray):
            result = jnp.dot(self.child, other)

            output_ds = self.data_subjects.dot(other)

            if isinstance(self.min_vals, lazyrepeatarray):
                minv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.min_vals.data, other
                    ).min(),
                    shape=result.shape,
                )
                maxv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.max_vals.data, other
                    ).max(),
                    shape=result.shape,
                )

            elif isinstance(self.min_vals, (int, float)):
                minv = lazyrepeatarray(
                    data=jnp.dot(np.ones_like(self.child) * self.min_vals, other).min(),
                    shape=result.shape,
                )
                maxv = lazyrepeatarray(
                    data=jnp.dot(np.ones_like(self.child) * self.max_vals, other).max(),
                    shape=result.shape,
                )
            else:
                raise NotImplementedError

            return GammaTensor(
                child=result,
                data_subjects=output_ds,
                min_vals=minv,
                max_vals=maxv,
            )
        elif isinstance(other, GammaTensor):
            output_state = dict()
            output_state[self.id] = self
            output_state[other.id] = other

            output_ds = self.data_subjects.dot(other.data_subjects)

            def _dot(state: dict) -> jax.numpy.DeviceArray:
                return jnp.dot(self.run(state))

            result = jnp.dot(self.child, other.child)

            if isinstance(self.min_vals, lazyrepeatarray):

                minv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.min_vals.data,
                        np.ones_like(other.child) * other.min_vals.data,
                    ).min(),
                    shape=result.shape,
                )
                maxv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.max_vals.data,
                        np.ones_like(other.child) * other.max_vals.data,
                    ).max(),
                    shape=result.shape,
                )
            elif isinstance(self.min_vals, (int, float)):
                minv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.min_vals,
                        np.ones_like(other.child) * other.min_vals,
                    ).min(),
                    shape=result.shape,
                )
                maxv = lazyrepeatarray(
                    data=jnp.dot(
                        np.ones_like(self.child) * self.max_vals,
                        np.ones_like(other.child) * other.max_vals,
                    ).max(),
                    shape=result.shape,
                )
            else:
                raise NotImplementedError

            return GammaTensor(
                child=result,
                data_subjects=output_ds,
                min_vals=minv,
                max_vals=maxv,
                func=_dot,
                state=output_state,
            )
        else:
            raise NotImplementedError(
                f"Undefined behaviour for GT.dot with {type(other)}"
            )

    def sqrt(self) -> GammaTensor:
        def _sqrt(state: dict) -> jax.numpy.DeviceArray:
            return jnp.sqrt(self.run(state))

        state = dict()
        state.update(self.state)

        min_v = jnp.sqrt(self.min_vals.data)
        max_v = jnp.sqrt(self.min_vals.data)

        child = jnp.sqrt(self.child)
        min_val = lazyrepeatarray(min_v, shape=child.shape)
        max_val = lazyrepeatarray(max_v, shape=child.shape)

        return GammaTensor(
            child=child,
            data_subjects=self.data_subjects,
            min_vals=min_val,
            max_vals=max_val,
            func=_sqrt,
            state=state,
        )

    def abs(self) -> GammaTensor:
        def _abs(state: dict) -> jax.numpy.DeviceArray:
            return jnp.abs(self.run(state))

        state = dict()
        state.update(self.state)

        data = self.child
        output = np.abs(data)

        min_v = np.abs(self.min_vals.data)
        max_v = np.abs(self.min_vals.data)

        return GammaTensor(
            child=output,
            data_subjects=self.data_subjects,
            min_vals=lazyrepeatarray(min_v, shape=output.shape),
            max_vals=lazyrepeatarray(max_v, shape=output.shape),
            func=_abs,
            state=state,
        )

    def clip(self, a_min: float, a_max: float) -> GammaTensor:
        def _clip(state: dict) -> jax.numpy.DeviceArray:
            return jnp.clip(self.run(state))

        state = dict()
        state.update(self.state)

        output_data = self.child.clip(a_min, a_max)

        min_v = np.clip(self.min_vals.data, a_min, a_max)
        max_v = np.clip(self.max_vals.data, a_min, a_max)

        min_vals = lazyrepeatarray(data=min_v, shape=output_data.shape)
        max_vals = lazyrepeatarray(data=max_v, shape=output_data.shape)

        return GammaTensor(
            child=output_data,
            data_subjects=self.data_subjects,
            min_vals=min_vals,
            max_vals=max_vals,
            func=_clip,
            state=state,
        )

    @staticmethod
    def convert_dsl(state: dict, new_state: Optional[dict] = None) -> Dict:
        if new_state is None:
            new_state = dict()
        if state:
            for tensor in list(state.values()):
                if isinstance(tensor.data_subjects, np.ndarray):
                    new_tensor = GammaTensor(
                        child=tensor.child,
                        data_subjects=np.zeros_like(
                            tensor.data_subjects, dtype=np.int64
                        ),
                        min_vals=tensor.min_vals,
                        max_vals=tensor.max_vals,
                        func=tensor.func,
                        state=GammaTensor.convert_dsl(tensor.state, {}),
                    )
                    # for idx, row in enumerate(tensor.data_subjects):
                    #     tensor.data_subjects[idx] = jnp.zeros_like(np.zeros_like(row), jnp.int64)
                else:

                    new_tensor = tensor
                new_state[new_tensor.id] = new_tensor
            return new_state
        else:
            return {}

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: Optional[float] = None,
        dsl_hack: bool = False,
    ) -> jax.numpy.DeviceArray:

        if dsl_hack is False:
            if sigma is None:
                sigma = (
                    self.child.mean() / 4
                )  # TODO @Ishan: replace this with calibration

            # if self.child.dtype != np.int64:
            #     raise Exception(
            #         "Data type of private values is not np.int64: ", self.child.dtype
            #     )

            if (
                not self.state
            ):  # if state tree is empty (e.g. publishing a PhiTensor w/ public vals directly)
                self.state[self.id] = self

            return vectorized_publish(
                min_vals=self.min_vals,
                max_vals=self.max_vals,
                state_tree=self.state,
                data_subjects=self.data_subjects,
                is_linear=self.is_linear,
                sigma=sigma,
                output_func=self.func,
                ledger=ledger,
                get_budget_for_user=get_budget_for_user,
                deduct_epsilon_for_user=deduct_epsilon_for_user,
            )
        else:
            tensor = GammaTensor(
                child=self.child,
                data_subjects=np.zeros_like(self.data_subjects),
                min_vals=self.min_vals,
                max_vals=self.max_vals,
                state=GammaTensor.convert_dsl(self.state),
            )

            return tensor.publish(
                get_budget_for_user, deduct_epsilon_for_user, ledger, sigma
            )

    # def expand_dims(self, axis: int) -> GammaTensor:
    #     def _expand_dims(state: dict) -> jax.numpy.DeviceArray:
    #         return jnp.expand_dims(self.run(state), axis)
    #
    #     state = dict()
    #     state.update(self.state)
    #
    #     return GammaTensor(
    #         child=jnp.expand_dims(self.child, axis),
    #         data_subjects=self.data_subjects,
    #         min_vals=self.min_vals,
    #         max_vals=self.max_vals,
    #         func=_expand_dims,
    #         state=state,
    #     )

    def squeeze(self, axis: Optional[int] = None) -> GammaTensor:
        def _squeeze(state: dict) -> jax.numpy.DeviceArray:
            return jnp.squeeze(self.run(state), axis)

        state = dict()
        state.update(self.state)
        return GammaTensor(
            child=jnp.squeeze(self.child, axis),
            data_subjects=jnp.squeeze(self.data_subjects, None),
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            func=_squeeze,
            state=state,
        )

    def __len__(self) -> int:
        return len(self.child)

    def __getitem__(self, item: Union[int, slice, PassthroughTensor]) -> GammaTensor:
        # TODO: Technically we could reduce ds.one_hot_lookup to remove any DS that won't be there
        # There technically isn't any penalty for keeping it as is, but maybe there's a sidechannel attack
        # where you index into one value in a GammaTensor and get all the data subjects of that Tensor?

        if isinstance(self.min_vals, (int, float)):
            minv = self.min_vals
            maxv = self.max_vals
        elif isinstance(self.min_vals, lazyrepeatarray):
            minv = self.min_vals[item]  # type: ignore
            maxv = self.max_vals[item]
        else:
            raise NotImplementedError

        if isinstance(item, PassthroughTensor):
            data = self.child[item.child]
            if self.shape == self.data_subjects.shape:
                return GammaTensor(
                    child=data,
                    min_vals=minv,
                    max_vals=maxv,
                    data_subjects=self.data_subjects[item.child],
                )
            elif len(self.shape) < len(self.data_subjects.shape):
                return GammaTensor(
                    child=data,
                    min_vals=minv,
                    max_vals=maxv,
                    data_subjects=self.data_subjects[item.child],
                    # self.data_subjects.data_subjects_indexed[:, item.child],
                )
            else:
                raise Exception(
                    f"Incompatible shapes: {self.shape}, {self.data_subjects.shape}"
                )
        else:
            data = self.child[item]

            return GammaTensor(
                child=data,
                min_vals=minv,
                max_vals=maxv,
                data_subjects=self.data_subjects[item],
            )

    def __setitem__(
        self, key: Union[int, slice, NDArray], value: Union[GammaTensor, np.ndarray]
    ) -> None:
        # relative
        from .phi_tensor import PhiTensor

        if isinstance(value, (PhiTensor, GammaTensor)):
            self.child[key] = value.child
            minv = value.child.min()
            maxv = value.child.max()

            if minv < self.min_vals.data.min():
                self.min_vals.data = minv

            if maxv > self.max_vals.data.max():
                self.max_vals.data = maxv

            self.data_subjects[key] = value.data_subjects

            # output_dsl = DataSubjectList.insert(
            #     dsl1=self.data_subjects, dsl2=value.data_subjects, index=key
            # )
            # self.data_subjects.one_hot_lookup = output_dsl.one_hot_lookup
            # self.data_subjects.data_subjects_indexed = output_dsl.data_subjects_indexed

        elif isinstance(value, np.ndarray):
            self.child[key] = value
            minv = value.min()
            maxv = value.max()

            if minv < self.min_vals.data.min():
                self.min_vals.data = minv

            if maxv > self.max_vals.data.max():
                self.max_vals.data = maxv

        else:
            raise NotImplementedError

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

        gamma_msg.state = serialize(self.state, to_bytes=True)
        chunk_bytes(
            capnp_serialize(dslarraytonumpyutf8(self.data_subjects), to_bytes=True),
            "dataSubjects",
            gamma_msg,
        )

        # Explicity convert lazyrepeatarray data to ndarray
        self.min_vals.data = np.array(self.min_vals.data)
        self.max_vals.data = np.array(self.max_vals.data)

        gamma_msg.minVal = serialize(self.min_vals, to_bytes=True)
        gamma_msg.maxVal = serialize(self.max_vals, to_bytes=True)
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

            if gamma_msg.isNumpy:
                child = capnp_deserialize(
                    combine_bytes(gamma_msg.child), from_bytes=True
                )
            else:
                child = deserialize(combine_bytes(gamma_msg.child), from_bytes=True)

            state = deserialize(gamma_msg.state, from_bytes=True)

            data_subjects = numpyutf8todslarray(
                capnp_deserialize(
                    combine_bytes(gamma_msg.dataSubjects), from_bytes=True
                )
            )

            min_val = deserialize(gamma_msg.minVal, from_bytes=True)
            max_val = deserialize(gamma_msg.maxVal, from_bytes=True)
            is_linear = gamma_msg.isLinear
            id_str = gamma_msg.id

            return GammaTensor(
                child=child,
                data_subjects=data_subjects,
                min_vals=min_val,
                max_vals=max_val,
                is_linear=is_linear,
                state=state,
                id=id_str,
            )
