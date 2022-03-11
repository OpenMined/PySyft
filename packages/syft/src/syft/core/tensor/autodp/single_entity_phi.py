# future
from __future__ import annotations

# stdlib
import operator
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple as TypeTuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
import numpy as np
import numpy.typing as npt
import torch

# relative
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs
from ....proto.core.adp.phi_tensor_pb2 import (
    SingleEntityPhiTensor as SingleEntityPhiTensor_PB,
)
from ....proto.core.tensor.single_entity_phi_tensor_pb2 import (
    TensorWrappedSingleEntityPhiTensorPointer as TensorWrappedSingleEntityPhiTensorPointer_PB,
)
from ....util import inherit_tags
from ...adp.entity import Entity
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize as serialize
from ...common.uid import UID
from ...node.abstract.node import AbstractNodeClient
from ...node.common.action.run_class_method_action import RunClassMethodAction
from ...pointer.pointer import Pointer
from ..ancestors import AutogradTensorAncestor
from ..broadcastable import is_broadcastable
from ..passthrough import AcceptableSimpleType  # type: ignore
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..passthrough import implements  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from ..smpc import utils
from ..smpc.mpc_tensor import MPCTensor
from ..smpc.share_tensor import ShareTensor
from ..smpc.utils import TYPE_TO_RING_SIZE
from ..tensor import Tensor
from ..util import inputs2child  # type: ignore
from .adp_tensor import ADPTensor
from .dp_tensor_converter import convert_to_gamma_tensor
from .initial_gamma import InitialGammaTensor
from .initial_gamma import IntermediateGammaTensor


@serializable()
class TensorWrappedSingleEntityPhiTensorPointer(Pointer):
    """
    This tensor represents a pointer to a very specific tensor chain. Eventually we'll have some sort
    of more intelligent/general representation for pointers to chains of objects, but for now this is
    what we're going with. This pointer represents all the arguments of the objects in the chain as its
    attributes.

    Thus, this class has two groups of attributes: one set are the attributes for SingeEntityPhiTensor:
        child: SupportedChainType,
        entity: Entity,
        min_vals: np.ndarray,
        max_vals: np.ndarray,
        scalar_manager: Optional[VirtualMachinePrivateScalarManager] = None,

    And the others are for initializing a Pointer object:
        client=self.client,
        id_at_location=self.id_at_location,
        object_type=self.object_type,
        tags=self.tags,
        description=self.description,
    """

    __name__ = "TensorWrappedSingleEntityPhiTensorPointer"
    __module__ = "syft.core.tensor.autodp.single_entity_phi"

    def __init__(
        self,
        entity: Entity,
        min_vals: np.typing.ArrayLike,
        max_vals: np.typing.ArrayLike,
        client: Any,
        scalar_manager: Optional[VirtualMachinePrivateScalarManager] = None,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
        public_shape: Optional[TypeTuple[int, ...]] = None,
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
        self.entity = entity
        self.scalar_manager = scalar_manager
        self.public_shape = public_shape
        self.public_dtype = public_dtype

    @property
    def synthetic(self) -> np.ndarray:
        return (
            np.random.rand(*list(self.public_shape)) * (self.max_vals - self.min_vals)  # type: ignore
            + self.min_vals
        ).astype(self.public_dtype)

    def __repr__(self) -> str:
        return (
            self.synthetic.__repr__()
            + "\n\n (The data printed above is synthetic - it's an imitation of the real data.)"
        )

    def share(self, *parties: TypeTuple[AbstractNodeClient, ...]) -> MPCTensor:
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

        attr_path_and_name = f"syft.core.tensor.tensor.Tensor.__{op_str}__"

        result = TensorWrappedSingleEntityPhiTensorPointer(
            entity=self.entity,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            client=self.client,
            scalar_manager=self.scalar_manager,
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

        if isinstance(other, TensorWrappedSingleEntityPhiTensorPointer):
            other_shape = other.public_shape
            other_dtype = other.public_dtype
        elif isinstance(other, (int, float)):
            other_shape = (1,)
            other_dtype = np.int32
        elif isinstance(other, bool):
            other_shape = (1,)
            other_dtype = np.dtype("bool")
        elif isinstance(other, np.ndarray):
            other_shape = other.shape
            other_dtype = other.dtype
        else:
            raise ValueError(
                f"Invalid Type for TensorWrappedSingleEntityPhiTensorPointer:{type(other)}"
            )

        if self.public_shape is not None and other_shape is not None:
            result_public_shape = utils.get_shape(
                op_str, self.public_shape, other_shape
            )

        if self.public_dtype is not None and other_dtype is not None:
            if self.public_dtype != other_dtype:
                raise ValueError(
                    f"Type for self and other do not match ({self.public_dtype} vs {other_dtype})"
                )
            result_public_dtype = self.public_dtype

        result.public_shape = result_public_shape
        result.public_dtype = result_public_dtype

        return result

    @staticmethod
    def _apply_op(
        self: TensorWrappedSingleEntityPhiTensorPointer,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
        op_str: str,
    ) -> Union[MPCTensor, TensorWrappedSingleEntityPhiTensorPointer]:
        """Performs the operation based on op_str

        Args:
            other (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]): second operand.

        Returns:
            Tuple[MPCTensor,Union[MPCTensor,int,float,np.ndarray]] : Result of the operation
        """
        op = getattr(operator, op_str)

        if (
            isinstance(other, TensorWrappedSingleEntityPhiTensorPointer)
            and self.client != other.client
        ):

            parties = [self.client, other.client]

            self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
            other_mpc = MPCTensor(
                secret=other, shape=other.public_shape, parties=parties
            )

            return op(self_mpc, other_mpc)

        elif isinstance(other, MPCTensor):

            return op(other, self)

        return self._apply_tensor_op(other=other, op_str=op_str)

    def __add__(
        self,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedSingleEntityPhiTensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedSingleEntityPhiTensorPointer._apply_op(self, other, "add")

    def __sub__(
        self,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedSingleEntityPhiTensorPointer, MPCTensor]:
        """Apply the "sub" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedSingleEntityPhiTensorPointer._apply_op(self, other, "sub")

    def __mul__(
        self,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedSingleEntityPhiTensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedSingleEntityPhiTensorPointer._apply_op(self, other, "mul")

    def __lt__(
        self,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedSingleEntityPhiTensorPointer, MPCTensor]:
        """Apply the "lt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedSingleEntityPhiTensorPointer._apply_op(self, other, "lt")

    def __gt__(
        self,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedSingleEntityPhiTensorPointer, MPCTensor]:
        """Apply the "gt" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedSingleEntityPhiTensorPointer._apply_op(self, other, "gt")

    def __ge__(
        self,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedSingleEntityPhiTensorPointer, MPCTensor]:
        """Apply the "ge" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedSingleEntityPhiTensorPointer._apply_op(self, other, "ge")

    def __le__(
        self,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedSingleEntityPhiTensorPointer, MPCTensor]:
        """Apply the "le" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedSingleEntityPhiTensorPointer._apply_op(self, other, "le")

    def __eq__(  # type: ignore
        self,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedSingleEntityPhiTensorPointer, MPCTensor]:
        """Apply the "eq" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedSingleEntityPhiTensorPointer._apply_op(self, other, "eq")

    def __ne__(  # type: ignore
        self,
        other: Union[
            TensorWrappedSingleEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedSingleEntityPhiTensorPointer, MPCTensor]:
        """Apply the "ne" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedSingleEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedSingleEntityPhiTensorPointer._apply_op(self, other, "ne")

    def to_local_object_without_private_data_child(self) -> SingleEntityPhiTensor:
        """Convert this pointer into a partial version of the SingleEntityPhiTensor but without
        any of the private data therein."""

        public_shape = getattr(self, "public_shape", None)
        public_dtype = getattr(self, "public_dtype", None)
        return Tensor(
            child=SingleEntityPhiTensor(
                child=None,
                entity=self.entity,
                min_vals=self.min_vals,  # type: ignore
                max_vals=self.max_vals,  # type: ignore
                scalar_manager=self.scalar_manager,
            ),
            public_shape=public_shape,
            public_dtype=public_dtype,
        )

    def _object2proto(self) -> "TensorWrappedSingleEntityPhiTensorPointer_PB":

        _entity = serialize(self.entity)
        _min_vals = serialize(self.min_vals)
        _max_vals = serialize(self.max_vals)
        _location = serialize(self.client.address)
        _scalar_manager = serialize(self.scalar_manager, to_bytes=True)
        _id_at_location = serialize(self.id_at_location)
        _object_type = self.object_type
        _tags = self.tags
        _description = self.description
        _public_shape = serialize(getattr(self, "public_shape", None), to_bytes=True)
        _public_dtype = serialize(getattr(self, "public_dtype", None), to_bytes=True)

        return TensorWrappedSingleEntityPhiTensorPointer_PB(
            entity=_entity,
            min_vals=_min_vals,
            max_vals=_max_vals,
            location=_location,
            scalar_manager=_scalar_manager,
            id_at_location=_id_at_location,
            object_type=_object_type,
            tags=_tags,
            description=_description,
            public_shape=_public_shape,
            public_dtype=_public_dtype,
        )

    @staticmethod
    def _proto2object(
        proto: TensorWrappedSingleEntityPhiTensorPointer_PB,
    ) -> "TensorWrappedSingleEntityPhiTensorPointer":

        entity = deserialize(blob=proto.entity)
        min_vals = deserialize(blob=proto.min_vals)
        max_vals = deserialize(blob=proto.max_vals)
        client = deserialize(blob=proto.location)
        scalar_manager = deserialize(blob=proto.scalar_manager, from_bytes=True)
        id_at_location = deserialize(blob=proto.id_at_location)
        object_type = proto.object_type
        tags = proto.tags
        public_shape = deserialize(blob=proto.public_shape, from_bytes=True)
        public_dtype = deserialize(blob=proto.public_dtype, from_bytes=True)
        description = proto.description

        return TensorWrappedSingleEntityPhiTensorPointer(
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            client=client,
            scalar_manager=scalar_manager,
            id_at_location=id_at_location,
            object_type=object_type,
            tags=tags,
            description=description,
            public_shape=public_shape,
            public_dtype=public_dtype,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type

        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return TensorWrappedSingleEntityPhiTensorPointer_PB


@serializable()
class SingleEntityPhiTensor(PassthroughTensor, AutogradTensorAncestor, ADPTensor):

    PointerClassOverride = TensorWrappedSingleEntityPhiTensorPointer

    # Number of entities in a SEPT is by definition 1
    n_entities = 1
    __slots__ = (
        "child",
        "_min_vals",
        "_max_vals",
        "entity",
        "scalar_manager",
    )

    def __init__(
        self,
        child: SupportedChainType,
        entity: Entity,
        min_vals: np.ndarray,
        max_vals: np.ndarray,
        scalar_manager: Optional[VirtualMachinePrivateScalarManager] = None,
    ) -> None:

        # child = the actual private data
        super().__init__(child)

        # identically shaped tensor to "child" but making the LOWEST possible value of this private value
        self._min_vals = min_vals

        # identically shaped tensor to "child" but making the HIGHEST possible value of this private value
        self._max_vals = max_vals

        # the identity of the data subject
        self.entity = entity

        if scalar_manager is None:
            self.scalar_manager = VirtualMachinePrivateScalarManager()
        else:
            self.scalar_manager = scalar_manager

    def init_pointer(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> TensorWrappedSingleEntityPhiTensorPointer:
        return TensorWrappedSingleEntityPhiTensorPointer(
            # Arguments specifically for SEPhiTensor
            entity=self.entity,
            min_vals=self._min_vals,
            max_vals=self._max_vals,
            scalar_manager=self.scalar_manager,
            # Arguments required for a Pointer to work
            client=client,
            id_at_location=id_at_location,
            object_type=object_type,
            tags=tags,
            description=description,
        )

    @property
    def gamma(self) -> InitialGammaTensor:
        """Property to cast this tensor into a GammaTensor"""
        return self.create_gamma()

    def create_gamma(
        self, scalar_manager: Optional[VirtualMachinePrivateScalarManager] = None
    ) -> InitialGammaTensor:
        """Return a new Gamma tensor based on this phi tensor"""

        if scalar_manager is None:
            scalar_manager = self.scalar_manager

        # Gamma expects an entity for each scalar
        entities = np.array([self.entity] * np.array(self.child.shape).prod()).reshape(
            self.shape
        )

        return InitialGammaTensor(
            values=self.child,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            entities=entities,
            scalar_manager=scalar_manager,
        )

    def publish(
        self, acc: Any, sigma: float, user_key: VerifyKey
    ) -> AcceptableSimpleType:
        print("PUBLISHING TO GAMMA:")
        print(self.child)
        return self.gamma.publish(acc=acc, sigma=sigma, user_key=user_key)

    @property
    def min_vals(self) -> np.ndarray:
        return self._min_vals

    @property
    def max_vals(self) -> np.ndarray:
        return self._max_vals

    def astype(self, np_type: np.dtype) -> SingleEntityPhiTensor:
        return self.__class__(
            child=self.child.astype(np_type),
            entity=self.entity,
            min_vals=self.min_vals.astype(np_type),
            max_vals=self.max_vals.astype(np_type),
            scalar_manager=self.scalar_manager,
        )

    def __repr__(self) -> str:

        """Pretty print some information, optimized for Jupyter notebook viewing."""
        return (
            f"{self.__class__.__name__}(entity={self.entity.name}, child={self.child})"
        )

    def __and__(self, other: Any) -> SingleEntityPhiTensor:
        """Note: this is bitwise and, not logical and"""
        return SingleEntityPhiTensor(
            child=self.child & other,
            min_vals=np.zeros_like(self.child),
            max_vals=np.ones_like(self.child),
            entity=self.entity,
        )

    def __or__(self, other: Any) -> SingleEntityPhiTensor:
        return SingleEntityPhiTensor(
            child=self.child | other,
            min_vals=np.zeros_like(self.child),
            max_vals=np.ones_like(self.child),
            entity=self.entity,
        )

    # Check for shape1 = (1,s), and shape2 = (,s) --> as an example
    def __eq__(
        self, other: SupportedChainType
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                # If other is a Numpy Array, we need to check if shapes can be broadcast
                if is_broadcastable(other.shape, self.child.shape):  # type: ignore
                    data = self.child == other
                else:
                    raise Exception(
                        f"Tensor shapes do not match "  # type: ignore
                        f"for __eq__: {self.child.shape} != {other.child.shape}"  # type: ignore
                    )
            else:
                data = self.child == other
        elif isinstance(other, SingleEntityPhiTensor):
            if self.entity != other.entity:
                return convert_to_gamma_tensor(self) == convert_to_gamma_tensor(other)
            else:
                if is_broadcastable(self.child.shape, other.child.shape):  # type: ignore
                    data = self.child == other.child
                else:
                    raise Exception(
                        f"Tensor shapes do not match for __eq__: {self.child.shape} != {other.child.shape}"
                    )
        elif isinstance(other, PassthroughTensor):
            if is_broadcastable(self.child.shape, other.child.shape):  # type: ignore
                data = self.child == other.child
        else:
            raise Exception(
                f"Tensor shapes do not match for __eq__: {self.child} != len{other}"
            )
        return SingleEntityPhiTensor(
            child=data,
            entity=self.entity,
            min_vals=self.min_vals * 0.0,
            max_vals=self.max_vals * 0.0 + 1.0,
            scalar_manager=self.scalar_manager,
        )

    def __ne__(
        self, other: SupportedChainType
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:
        # Make use of the equal operator we just implemented, and invert the result
        opposite_result = self.__eq__(other)

        if isinstance(opposite_result, SingleEntityPhiTensor):
            return SingleEntityPhiTensor(
                child=opposite_result.child ^ 1,
                entity=opposite_result.entity,
                min_vals=opposite_result.min_vals,
                max_vals=opposite_result.max_vals,
                scalar_manager=opposite_result.scalar_manager,
            )
        elif isinstance(opposite_result, InitialGammaTensor):
            return InitialGammaTensor(
                values=opposite_result.values ^ 1,
                entities=opposite_result.entities,
                min_vals=opposite_result.min_vals,
                max_vals=opposite_result.max_vals,
                scalar_manager=opposite_result.scalar_manager,
            )
        else:
            raise Exception

    def __abs__(self) -> SingleEntityPhiTensor:

        data = self.child.abs()

        # create true/false gate inputs
        minvals_is_gt0 = self.min_vals > 0
        minvals_is_le0 = -minvals_is_gt0 + 1
        maxvals_is_gt0 = self.max_vals >= 0
        maxvals_is_le0 = -maxvals_is_gt0 + 1

        # create true/false gates
        is_strict_gt0 = minvals_is_gt0
        is_gtlt0 = minvals_is_le0 * maxvals_is_gt0
        is_strict_lt0 = minvals_is_le0 * maxvals_is_le0

        # if min_vals > 0, then new min_vals doesn't change
        min_vals_strict_gt0 = self.min_vals

        # if min_vals < 0 and max_vals > 0, then new min_vals = 0
        min_vals_gtlt0 = self.min_vals * 0

        # if min_vals < 0 and max_vals < 0, then new min_vals = -max_vals
        min_vals_strict_lt0 = -self.max_vals

        # sum of masked options
        min_vals = is_strict_gt0 * min_vals_strict_gt0
        min_vals = min_vals + (is_gtlt0 * min_vals_gtlt0)
        min_vals = min_vals + (is_strict_lt0 * min_vals_strict_lt0)

        #  if min_vals > 0, then new min_vals doesn't change
        max_vals_strict_gt0 = self.max_vals

        # if min_vals < 0 and max_vals > 0, then new min_vals = 0
        max_vals_gtlt0 = np.max([self.max_vals, -self.min_vals])  # type: ignore

        #  if min_vals < 0 and max_vals < 0, then new min_vals = -max_vals
        max_vals_strict_lt0 = -self.min_vals

        # sum of masked options
        max_vals = is_strict_gt0 * max_vals_strict_gt0
        max_vals = max_vals + (is_gtlt0 * max_vals_gtlt0)
        max_vals = max_vals + (is_strict_lt0 * max_vals_strict_lt0)

        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def __add__(
        self, other: SupportedChainType
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:

        # if the tensor being added is also private
        if isinstance(other, SingleEntityPhiTensor):
            if self.entity != other.entity:
                left = convert_to_gamma_tensor(self)
                right = convert_to_gamma_tensor(other)
                return left + right

            data = self.child + other.child
            min_vals = self.min_vals + other.min_vals
            max_vals = self.max_vals + other.max_vals
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        # if the tensor being added is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            data = self.child + other
            min_vals = self.min_vals + other
            max_vals = self.max_vals + other
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )
        elif isinstance(other, IntermediateGammaTensor):
            return convert_to_gamma_tensor(self) + other
        else:
            print("Type is unsupported:" + str(type(other)))
            raise NotImplementedError

    def __neg__(self) -> SingleEntityPhiTensor:

        data = self.child * -1
        min_vals = self.max_vals * -1
        max_vals = self.min_vals * -1
        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def __getitem__(self, key: Any) -> SingleEntityPhiTensor:

        data = self.child.__getitem__(key)
        min_vals = self.min_vals.__getitem__(key)
        max_vals = self.max_vals.__getitem__(key)

        if isinstance(data, (np.number, bool, int, float)):
            data = np.array([data])  # 1 dimensional np.array
        if isinstance(min_vals, (np.number, bool, int, float)):
            min_vals = np.array(min_vals)  # 1 dimensional np.array
        if isinstance(max_vals, (np.number, bool, int, float)):
            max_vals = np.array(max_vals)  # 1 dimensional np.array

        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def __mul__(
        self, other: SupportedChainType
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:

        if isinstance(other, SingleEntityPhiTensor):
            print(f"SELF ENTITY:{self.entity.name}")
            print(f"OTHER ENTITY:{other.entity.name}")
            if self.entity != other.entity:
                print("Entities are not the same?!?!?!")
                return convert_to_gamma_tensor(self) * convert_to_gamma_tensor(other)

            data = self.child * other.child

            min_min = self.min_vals * other.min_vals
            min_max = self.min_vals * other.max_vals
            max_min = self.max_vals * other.min_vals
            max_max = self.max_vals * other.max_vals

            min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )
        elif is_acceptable_simple_type(other):

            data = self.child * other

            min_min = self.min_vals * other
            min_max = self.min_vals * other
            max_min = self.max_vals * other
            max_max = self.max_vals * other

            min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        else:
            return NotImplemented

    def __pos__(self) -> SingleEntityPhiTensor:
        """Identity operator, returns itself"""
        return self

    def __sub__(
        self, other: SupportedChainType
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:

        if isinstance(other, SingleEntityPhiTensor):
            if self.entity != other.entity:
                return convert_to_gamma_tensor(self) - convert_to_gamma_tensor(other)

            data = self.child - other.child
            min_min = self.min_vals - other.min_vals
            min_max = self.min_vals - other.max_vals
            max_min = self.max_vals - other.min_vals
            max_max = self.max_vals - other.max_vals
            min_vals = np.minimum.reduce([min_min, min_max, max_min, max_max])
            max_vals = np.maximum.reduce([min_min, min_max, max_min, max_max])

            entity = self.entity

        elif is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                if not is_broadcastable(other.shape, self.child.shape):  # type: ignore
                    raise Exception(
                        f"Shapes do not match for subtraction: {self.child.shape} and {other.shape}"
                    )
            data = self.child - other
            min_vals = self.min_vals - other
            max_vals = self.max_vals - other
            entity = self.entity
        else:
            raise NotImplementedError
        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def __truediv__(self, other: SupportedChainType) -> SingleEntityPhiTensor:

        if isinstance(other, SingleEntityPhiTensor):

            if self.entity != other.entity:
                # this should return a GammaTensor
                return NotImplemented

            data = self.child / other.child

            if (other.min_vals == 0).any() or (other.max_vals == 0).any():

                raise Exception(
                    "Infinite sensitivity - we can support this in the future but not yet"
                )

            else:

                min_min = self.min_vals / other.min_vals
                min_max = self.min_vals / other.max_vals
                max_min = self.max_vals / other.min_vals
                max_max = self.max_vals / other.max_vals

                min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
                max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore

            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )
        else:
            # Ignoring unsupported operand error b/c other logic is taken care of
            return self * (1 / other)  # type: ignore

    def compress(
        self,
        condition: np.typing.ArrayLike,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
    ) -> SingleEntityPhiTensor:
        """Return selected slices of this array along a given axis"""
        if (
            isinstance(self.child, int)
            or isinstance(self.child, float)
            or isinstance(self.child, bool)
        ):
            # For these data types, the compress operation is meaningless, so don't change them.
            data = self.child
            print(
                f"Warning: Tensor data was of type {type(data)}, compress operation had no effect."
            )
        else:
            data = self.child.compress(condition, axis, out)

            # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these data types, the compress operation is meaningless, so don't change them.
            min_vals = self.min_vals
            # print(f'Warning: Tensor data was of type {type(data)}, compress operation had no effect.')
        else:
            min_vals = self.min_vals.compress(condition, axis, out)

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these data types, the compress operation is meaningless, so don't change them.
            max_vals = self.max_vals
            # print(f'Warning: Tensor data was of type {type(data)}, compress operation had no effect.')
        else:
            max_vals = self.max_vals.compress(condition, axis, out)

        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def dot(self, other: SupportedChainType) -> SingleEntityPhiTensor:
        return self.manual_dot(other)

    # ndarray.flatten(order='C')
    def flatten(self, order: str = "C") -> SingleEntityPhiTensor:
        if (
            isinstance(self.child, int)
            or isinstance(self.child, float)
            or isinstance(self.child, bool)
        ):
            # For these data types, the flatten operation is meaningless, so don't change them.
            data = self.child
            print(
                f"Warning: Tensor data was of type {type(data)}, flatten operation had no effect."
            )
        else:
            data = self.child.flatten(order)

            # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these data types, the flatten operation is meaningless, so don't change them.
            min_vals = self.min_vals
            # print(f'Warning: Tensor data was of type {type(data)}, flatten operation had no effect.')
        else:
            min_vals = self.min_vals.flatten(order)

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these data types, the flatten operation is meaningless, so don't change them.
            max_vals = self.max_vals
            # print(f'Warning: Tensor data was of type {type(data)}, flatten operation had no effect.')
        else:
            max_vals = self.max_vals.flatten(order)

        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    # def partition(
    #     self,
    #     kth: Union[int, List[int], np.ndarray],
    #     axis: Optional[int] = -1,
    #     kind: Optional[str] = "introselect",
    #     order: Optional[Union[str, List[str]]] = None,
    # ) -> SingleEntityPhiTensor:
    #     # this method mutates self
    #     """Interchange two axes of the Tensor"""
    #     if (
    #         isinstance(self.child, int)
    #         or isinstance(self.child, float)
    #         or isinstance(self.child, bool)
    #     ):
    #         # For these singleton data types, the partition operation is meaningless, so don't change them.
    #         print(
    #             f"Warning: Tensor data was of type {type(self.child)}, partition operation had no effect."
    #         )
    #     else:
    #         self.child.partition(kth, axis, kind, order)
    #
    #         # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
    #     if (
    #         isinstance(self.min_vals, int)
    #         or isinstance(self.min_vals, float)
    #         or isinstance(self.min_vals, bool)
    #     ):
    #         # For these singleton data types, the partition operation is meaningless, so don't change them.
    #         print(
    #             f"Warning: Min_vals metadata was of type {type(self.min_vals)}, partition operation had no effect."
    #         )
    #     else:
    #         self.min_vals.partition(kth, axis, kind, order)
    #
    #     if (
    #         isinstance(self.max_vals, int)
    #         or isinstance(self.max_vals, float)
    #         or isinstance(self.max_vals, bool)
    #     ):
    #         # For these singleton data types, the partition operation is meaningless, so don't change them.
    #         print(
    #             f"Warning: Max_vals metadata was of type {type(self.max_vals)}, partition operation had no effect."
    #         )
    #     else:
    #         self.max_vals.partition(kth, axis, kind, order)
    #
    #     return self

    # ndarray.ravel(order='C')
    def ravel(self, order: str = "C") -> SingleEntityPhiTensor:
        if (
            isinstance(self.child, int)
            or isinstance(self.child, float)
            or isinstance(self.child, bool)
        ):
            # For these data types, the ravel operation is meaningless, so don't change them.
            data = self.child
            print(
                f"Warning: Tensor data was of type {type(data)}, ravel operation had no effect."
            )
        else:
            data = self.child.ravel(order)

            # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these data types, the ravel operation is meaningless, so don't change them.
            min_vals = self.min_vals
            # print(f'Warning: Tensor data was of type {type(data)}, ravel operation had no effect.')
        else:
            min_vals = self.min_vals.ravel(order)

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these data types, the ravel operation is meaningless, so don't change them.
            max_vals = self.max_vals
            # print(f'Warning: Tensor data was of type {type(data)}, ravel operation had no effect.')
        else:
            max_vals = self.max_vals.ravel(order)
        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def repeat(
        self, repeats: Union[int, TypeTuple[int, ...]], axis: Optional[int] = None
    ) -> SingleEntityPhiTensor:

        data = self.child.repeat(repeats, axis=axis)
        min_vals = self.min_vals.repeat(repeats, axis=axis)
        max_vals = self.max_vals.repeat(repeats, axis=axis)
        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def reshape(self, *args: Any) -> SingleEntityPhiTensor:
        if (
            isinstance(self.child, int)
            or isinstance(self.child, float)
            or isinstance(self.child, bool)
        ):
            # For these data types, the reshape operation is meaningless, so don't change them.
            data = self.child
            print(
                f"Warning: Tensor data was of type {type(data)}, reshape operation had no effect."
            )
        else:
            data = self.child.reshape(*args)

            # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these data types, the reshape operation is meaningless, so don't change them.
            min_vals = self.min_vals
            # print(f'Warning: Tensor data was of type {type(data)}, reshape operation had no effect.')
        else:
            min_vals = self.min_vals.reshape(*args)

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these data types, the reshape operation is meaningless, so don't change them.
            max_vals = self.max_vals
            # print(f'Warning: Tensor data was of type {type(data)}, reshape operation had no effect.')
        else:
            max_vals = self.max_vals.reshape(*args)

        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def resize(
        self,
        new_shape: Union[TypeTuple[int], int, typing.Iterable],
        refcheck: bool = True,
    ) -> None:
        """Change shape and size of array, in-place."""
        if (
            isinstance(self.child, int)
            or isinstance(self.child, float)
            or isinstance(self.child, bool)
        ):
            # For these data types, the resize operation is meaningless, so don't change them.
            pass
            print(
                f"Warning: Tensor data was of type {type(self.child)}, resize operation had no effect."
            )
        else:
            # self.child = self.child.resize(new_shape, refcheck)
            self.child = np.resize(self.child, new_shape)

        # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these data types, the resize operation is meaningless, so don't change them.
            pass
            # print(f'Warning: min_vals data was of type {type(self.min_vals)}, resize operation had no effect.')
        else:
            self._min_vals = np.reshape(self.min_vals, new_shape)

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these data types, the resize operation is meaningless, so don't change them.
            pass
            # print(f'Warning: max_vals data was of type {type(data)}, resize operation had no effect.')
        else:
            self._max_vals = np.reshape(self.max_vals, new_shape)

        return None

    def squeeze(self, axis: Optional[int] = None) -> SingleEntityPhiTensor:
        if (
            isinstance(self.child, int)
            or isinstance(self.child, float)
            or isinstance(self.child, bool)
        ):
            # For these data types, the squeeze operation is meaningless, so don't change them.
            data = self.child
            print(
                f"Warning: Tensor data was of type {type(data)}, squeeze operation had no effect."
            )
        else:
            data = self.child.squeeze(axis)

            # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these data types, the squeeze operation is meaningless, so don't change them.
            min_vals = self.min_vals
            # print(f'Warning: Tensor data was of type {type(data)}, squeeze operation had no effect.')
        else:
            min_vals = self.min_vals.squeeze(axis)

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these data types, the squeeze operation is meaningless, so don't change them.
            max_vals = self.max_vals
            # print(f'Warning: Tensor data was of type {type(data)}, squeeze operation had no effect.')
        else:
            max_vals = self.max_vals.squeeze(axis)

        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def sum(self, *args: Any, **kwargs: Any) -> SingleEntityPhiTensor:

        data = self.child.sum(*args, **kwargs)
        min_vals = self.min_vals.sum(*args, **kwargs)
        max_vals = self.max_vals.sum(*args, **kwargs)
        entity = self.entity

        data = np.array(data)
        min_vals = np.array(min_vals)
        max_vals = np.array(max_vals)

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def swapaxes(self, axis1: int, axis2: int) -> SingleEntityPhiTensor:
        """Interchange two axes of the Tensor"""
        if (
            isinstance(self.child, int)
            or isinstance(self.child, float)
            or isinstance(self.child, bool)
        ):
            # For these singleton data types, the swapaxes operation is meaningless, so don't change them.
            data = self.child
            print(
                f"Warning: Tensor data was of type {type(data)}, swapaxes operation had no effect."
            )
        else:
            data = self.child.swapaxes(axis1, axis2)

            # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these singleton data types, the swapaxes operation is meaningless, so don't change them.
            min_vals = self.min_vals
            # print(f'Warning: Tensor data was of type {type(data)}, swapaxes operation had no effect.')
        else:
            min_vals = self.min_vals.swapaxes(axis1, axis2)

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these singleton data types, the swapaxes operation is meaningless, so don't change them.
            max_vals = self.max_vals
            # print(f'Warning: Tensor data was of type {type(data)}, swapaxes operation had no effect.')
        else:
            max_vals = self.max_vals.swapaxes(axis1, axis2)

        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def transpose(self, *args: Any, **kwargs: Any) -> SingleEntityPhiTensor:
        """Transposes self.child, min_vals, and max_vals if these can be transposed, otherwise doesn't change them."""
        if (
            isinstance(self.child, int)
            or isinstance(self.child, float)
            or isinstance(self.child, bool)
        ):
            # For these data types, the transpose operation is meaningless, so don't change them.
            data = self.child
            print(
                f"Warning: Tensor data was of type {type(data)}, transpose operation had no effect."
            )
        else:
            data = self.child.transpose(*args)

        # TODO: Should we give warnings for min_val and max_val being single floats/integers/booleans too?
        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            # For these data types, the transpose operation is meaningless, so don't change them.
            min_vals = self.min_vals
            # print(f'Warning: Tensor data was of type {type(data)}, transpose operation had no effect.')
        else:
            min_vals = self.min_vals.transpose(*args)

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            # For these data types, the transpose operation is meaningless, so don't change them.
            max_vals = self.max_vals
            # print(f'Warning: Tensor data was of type {type(data)}, transpose operation had no effect.')
        else:
            max_vals = self.max_vals.transpose(*args)

        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def __le__(
        self, other: SupportedChainType
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:

        # if the tensor being compared is also private
        if isinstance(other, SingleEntityPhiTensor):

            if self.entity != other.entity:
                # this should return a GammaTensor
                return convert_to_gamma_tensor(self) <= convert_to_gamma_tensor(other)

            if len(self.child) != len(other.child):
                raise Exception(
                    f"Tensor dims do not match for __le__: {len(self.child)} != {len(other.child)}"  # type: ignore
                )

            data = (
                self.child <= other.child
            ) * 1  # the * 1 just makes sure it returns integers instead of True/False
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            data = (self.child <= other) * 1
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        else:
            return NotImplemented

    def __ge__(
        self, other: SupportedChainType
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:

        # if the tensor being compared is also private
        if isinstance(other, SingleEntityPhiTensor):

            if self.entity != other.entity:
                # this should return a GammaTensor
                return convert_to_gamma_tensor(self) >= convert_to_gamma_tensor(other)

            if len(self.child) != len(other.child):
                raise Exception(
                    f"Tensor dims do not match for __ge__: {len(self.child)} != {len(other.child)}"  # type: ignore
                )

            data = (
                self.child >= other.child
            ) * 1  # the * 1 just makes sure it returns integers instead of True/False
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            data = (self.child >= other) * 1
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        else:
            return NotImplemented

    def __lt__(
        self, other: SupportedChainType
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:

        # if the tensor being compared is also private
        if isinstance(other, SingleEntityPhiTensor):

            if self.entity != other.entity:
                # this should return a GammaTensor
                return convert_to_gamma_tensor(self) < convert_to_gamma_tensor(other)

            if len(self.child) != len(other.child):
                raise Exception(
                    f"Tensor dims do not match for __lt__: {len(self.child)} != {len(other.child)}"  # type: ignore
                )

            data = (
                self.child < other.child
            ) * 1  # the * 1 just makes sure it returns integers instead of True/False
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            data = (self.child < other) * 1
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        else:
            return NotImplemented

    def __gt__(self, other: SupportedChainType) -> SingleEntityPhiTensor:

        # if the tensor being compared is also private
        if isinstance(other, SingleEntityPhiTensor):

            if self.entity != other.entity:
                # this should return a GammaTensor
                return convert_to_gamma_tensor(self) > convert_to_gamma_tensor(other)

            if len(self.child) != len(other.child):
                raise Exception(
                    f"Tensor dims do not match for __gt__: {len(self.child)} != {len(other.child)}"  # type: ignore
                )

            data = (
                self.child > other.child
            ) * 1  # the * 1 just makes sure it returns integers instead of True/False
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        # if the tensor being compared is a public tensor / int / float / etc.
        elif is_acceptable_simple_type(other):

            data = (self.child > other) * 1
            min_vals = self.min_vals * 0
            max_vals = (self.max_vals * 0) + 1
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        else:
            return NotImplemented

    def clip(
        self, a_min: npt.ArrayLike, a_max: npt.ArrayLike, *args: Any
    ) -> SingleEntityPhiTensor:

        if a_min is None and a_max is None:
            raise Exception("ValueError: clip: must set either max or min")

        if is_acceptable_simple_type(self.child):
            if isinstance(self.child, np.ndarray):
                data = self.child.clip(a_min, a_max)
            else:
                # self.child is a singleton
                data = max(a_min, min(self.child, a_max)) if a_min <= a_max else a_max
        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().clip(a_min, a_max)

        if isinstance(self.min_vals, np.ndarray):
            min_vals = np.clip(self.min_vals, a_min=a_min, a_max=a_max, *args)
        else:
            min_vals = (
                max(a_min, min(self.min_vals, a_max)) if a_min <= a_max else a_max
            )

        if isinstance(self.max_vals, np.ndarray):
            max_vals = np.clip(self.max_vals, a_min=a_min, a_max=a_max, *args)
        else:
            max_vals = (
                max(a_min, min(self.max_vals, a_max)) if a_min <= a_max else a_max
            )

        return SingleEntityPhiTensor(
            child=data,
            entity=self.entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def any(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        where: Optional[bool] = True,
    ) -> SingleEntityPhiTensor:
        """Test whether any element along a given axis evaluates to True"""

        if is_acceptable_simple_type(self.child):
            if isinstance(self.child, np.ndarray):
                data = self.child.any(
                    axis=axis, out=np.array(True), keepdims=keepdims, where=where
                )
            else:
                # self.child is a singleton
                data = self.child != 0

        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().any(
                axis=axis, out=np.array(True), keepdims=keepdims, where=where
            )

        if isinstance(self.min_vals, np.ndarray):
            # test whether any min val evaluates to True
            min_vals = self.min_vals.any(
                axis=axis, out=np.array(True), keepdims=keepdims, where=where
            )
        else:
            min_vals = self.min_vals != 0

        if isinstance(self.max_vals, np.ndarray):
            # test whether any max val evaluates to True
            max_vals = self.max_vals.any(
                axis=axis, out=np.array(True), keepdims=keepdims, where=where
            )
        else:
            max_vals = self.max_vals != 0

        # * 1 just makes sure it returns integers instead of True/False
        return SingleEntityPhiTensor(
            child=data * 1,
            entity=self.entity,
            min_vals=min_vals * 1,
            max_vals=max_vals * 1,
            scalar_manager=self.scalar_manager,
        )

    def all(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        where: Optional[bool] = True,
    ) -> SingleEntityPhiTensor:
        """Test whether all elements along a given axis evaluates to True"""

        if is_acceptable_simple_type(self.child):
            if isinstance(self.child, np.ndarray):
                data = self.child.all(
                    axis=axis, out=np.array(True), keepdims=keepdims, where=where
                )
            else:
                # self.child is a singleton
                data = self.child != 0

        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().all(
                axis=axis, out=np.array(True), keepdims=keepdims, where=where
            )

        if isinstance(self.min_vals, np.ndarray):
            # test whether all min vals evaluate to True
            min_vals = self.min_vals.all(
                axis=axis, out=np.array(True), keepdims=keepdims, where=where
            )
        else:
            min_vals = self.min_vals != 0

        if isinstance(self.max_vals, np.ndarray):
            # test whether all max vals evaluate to True
            max_vals = self.max_vals.all(
                axis=axis, out=np.array(True), keepdims=keepdims, where=where
            )
        else:
            max_vals = self.max_vals != 0

        # * 1 just makes sure it returns integers instead of True/False
        return SingleEntityPhiTensor(
            child=data * 1,
            entity=self.entity,
            min_vals=min_vals * 1,
            max_vals=max_vals * 1,
            scalar_manager=self.scalar_manager,
        )

    def abs(
        self,
        out: Optional[np.ndarray] = None,
    ) -> SingleEntityPhiTensor:
        """Calculate the absolute value element-wise"""
        if is_acceptable_simple_type(self.child):
            data = self.child.__abs__()

        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().__abs__()

        if isinstance(self.min_vals, np.ndarray):
            min_vals = np.abs(self.min_vals, out)
        else:
            min_vals = abs(self.min_vals)

        if isinstance(self.max_vals, np.ndarray):
            max_vals = np.abs(self.max_vals, out)
        else:
            max_vals = abs(self.max_vals)

        return SingleEntityPhiTensor(
            child=data,
            entity=self.entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def pow(self, value: SupportedChainType) -> SingleEntityPhiTensor:
        """Return elements raised to powers from value, element-wise"""

        if isinstance(value, SingleEntityPhiTensor):

            if self.entity != value.entity:
                return NotImplemented

            data = self.child**value.child

            min_min = self.min_vals**value.min_vals
            min_max = self.min_vals**value.max_vals
            max_min = self.max_vals**value.min_vals
            max_max = self.max_vals**value.max_vals

            min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        elif is_acceptable_simple_type(value):

            data = self.child**value

            min_min = self.min_vals**value
            min_max = self.min_vals**value
            max_min = self.max_vals**value
            max_max = self.max_vals**value

            min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)  # type: ignore
            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )

        else:
            return NotImplemented

    def copy(
        self, order: Optional[str] = "K", subok: Optional[bool] = True
    ) -> SingleEntityPhiTensor:
        """Return copy of the given object"""
        if is_acceptable_simple_type(self.child):
            data = self.child.copy()

        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().copy()

        elif isinstance(self.child, ShareTensor):
            data = self.child.copy()

        if isinstance(self.min_vals, np.ndarray):
            min_vals = np.array(self.min_vals, order=order, subok=subok, copy=True)
        else:
            min_vals = self.min_vals

        if isinstance(self.max_vals, np.ndarray):
            max_vals = np.array(self.max_vals, order=order, subok=subok, copy=True)
        else:
            max_vals = self.max_vals

        return SingleEntityPhiTensor(
            child=data,
            entity=self.entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    @staticmethod
    def sept_like(tensor: SingleEntityPhiTensor, ent: Entity) -> SingleEntityPhiTensor:
        """Create a SEPT with identical data, but belonging to a different entity"""
        return SingleEntityPhiTensor(
            child=tensor.child,
            entity=ent,
            min_vals=tensor.min_vals,
            max_vals=tensor.max_vals,
            scalar_manager=tensor.scalar_manager,
        )

    @staticmethod
    def zeros_like(
        tensor: SingleEntityPhiTensor, ent: Optional[Entity] = None
    ) -> SingleEntityPhiTensor:
        """Create a SEPT of the same shape, but with zeros as its data.
        By default, the tensor will have the same entity, but a different entity can be passed instead.
        """
        if ent:
            return SingleEntityPhiTensor(
                child=np.zeros_like(tensor.child),
                entity=ent,
                min_vals=np.zeros_like(tensor.min_vals),
                max_vals=np.ones_like(tensor.max_vals),
                scalar_manager=tensor.scalar_manager,
            )
        else:
            return SingleEntityPhiTensor(
                child=np.zeros_like(tensor.child),
                entity=tensor.entity,
                min_vals=np.zeros_like(tensor.min_vals),
                max_vals=np.ones_like(tensor.max_vals),
                scalar_manager=tensor.scalar_manager,
            )

    @staticmethod
    def ones_like(
        tensor: SingleEntityPhiTensor, ent: Optional[Entity] = None
    ) -> SingleEntityPhiTensor:
        """Create a SEPT of the same shape, but with ones as its data
        By default, the tensor will have the same entity, but a different entity can be passed instead.
        """
        if ent:
            return SingleEntityPhiTensor(
                child=np.ones_like(tensor.child),
                entity=ent,
                min_vals=np.zeros_like(tensor.min_vals),
                max_vals=np.ones_like(tensor.max_vals),
                scalar_manager=tensor.scalar_manager,
            )
        else:
            return SingleEntityPhiTensor(
                child=np.ones_like(tensor.child),
                entity=tensor.entity,
                min_vals=np.zeros_like(tensor.min_vals),
                max_vals=np.ones_like(tensor.max_vals),
                scalar_manager=tensor.scalar_manager,
            )

    def take(
        self,
        indices: np.ArrayLike,
        axis: Optional[int] = None,
        mode: Optional[str] = "raise",
    ) -> SingleEntityPhiTensor:
        """Take elements from an array along an axis"""
        if is_acceptable_simple_type(self.child):
            data = self.child.take(indices=indices, axis=axis, mode=mode)

        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().take(indices=indices, axis=axis, mode=mode)

        min_vals = self.min_vals.take(indices=indices, axis=axis, mode=mode)
        max_vals = self.max_vals.take(indices=indices, axis=axis, mode=mode)

        return SingleEntityPhiTensor(
            child=data,
            entity=self.entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def diagonal(
        self,
        offset: Optional[int] = 0,
        axis1: Optional[int] = 0,
        axis2: Optional[int] = 1,
    ) -> SingleEntityPhiTensor:
        """Return specified diagonals"""
        if is_acceptable_simple_type(self.child):
            if (
                isinstance(self.child, int)
                or isinstance(self.child, float)
                or isinstance(self.child, bool)
            ):
                raise Exception(
                    "ValueError: diag requires an array of at least two dimensions"
                )

            elif isinstance(self.child, np.matrix):
                # Make diagonal of matrix 1-D to preserve backward compatibility.
                data = np.asarray(self.child).diagonal(
                    offset=offset, axis1=axis1, axis2=axis2
                )
            else:
                data = np.asanyarray(self.child).diagonal(
                    offset=offset, axis1=axis1, axis2=axis2
                )
        elif isinstance(self.child, torch.Tensor):
            if isinstance(self.child.numpy(), np.matrix):
                # Make diagonal of matrix 1-D to preserve backward compatibility.
                data = np.asarray(self.child.numpy()).diagonal(
                    offset=offset, axis1=axis1, axis2=axis2
                )
            else:
                data = np.asanyarray(self.child.numpy()).diagonal(
                    offset=offset, axis1=axis1, axis2=axis2
                )

        if (
            isinstance(self.min_vals, int)
            or isinstance(self.min_vals, float)
            or isinstance(self.min_vals, bool)
        ):
            raise Exception(
                "ValueError: diag requires an array of at least two dimensions"
            )

        elif isinstance(self.min_vals, np.matrix):
            # Make diagonal of matrix 1-D to preserve backward compatibility.
            min_vals = np.asarray(self.min_vals).diagonal(
                offset=offset, axis1=axis1, axis2=axis2
            )
        else:
            min_vals = np.asanyarray(self.min_vals).diagonal(
                offset=offset, axis1=axis1, axis2=axis2
            )

        if (
            isinstance(self.max_vals, int)
            or isinstance(self.max_vals, float)
            or isinstance(self.max_vals, bool)
        ):
            raise Exception(
                "ValueError: diag requires an array of at least two dimensions"
            )

        elif isinstance(self.max_vals, np.matrix):
            # Make diagonal of matrix 1-D to preserve backward compatibility.
            max_vals = np.asarray(self.max_vals).diagonal(
                offset=offset, axis1=axis1, axis2=axis2
            )
        else:
            max_vals = np.asanyarray(self.max_vals).diagonal(
                offset=offset, axis1=axis1, axis2=axis2
            )

        return SingleEntityPhiTensor(
            child=data,
            entity=self.entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def max(
        self,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[int] = None,
        where: bool = True,
    ) -> SingleEntityPhiTensor:
        # Note: Who knew this method had SO MANY ARGUMENTS?!?!?
        if is_acceptable_simple_type(self.child):
            if isinstance(self.child, np.ndarray):
                data = self.child.max(axis, out, keepdims, initial, where)
            else:
                # This implies self.child is a singleton (int, float, bool, etc)
                data = self.child
        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().max(axis, out, keepdims, initial, where)
        else:
            raise NotImplementedError

        if isinstance(self.min_vals, np.ndarray):
            min_vals = self.min_vals.max(axis, out, keepdims, initial, where)
        else:
            min_vals = self.min_vals

        if isinstance(self.max_vals, np.ndarray):
            max_vals = self.max_vals.max(axis, out, keepdims, initial, where)
        else:
            max_vals = self.max_vals

        return SingleEntityPhiTensor(
            child=data,
            max_vals=max_vals,
            min_vals=min_vals,
            entity=self.entity,
        )

    def min(
        self,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        keepdims: Optional[bool] = False,
        initial: Optional[int] = None,
        where: bool = True,
    ) -> SingleEntityPhiTensor:
        # Note: Who knew this method had SO MANY ARGUMENTS?!?!?
        if is_acceptable_simple_type(self.child):
            if isinstance(self.child, np.ndarray):
                data = self.child.min(axis, out, keepdims, initial, where)
            else:
                # This implies self.child is a singleton (int, float, bool, etc)
                data = self.child
        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().min(axis, out, keepdims, initial, where)
        else:
            raise NotImplementedError

        if isinstance(self.min_vals, np.ndarray):
            min_vals = self.min_vals.min(axis, out, keepdims, initial, where)
        else:
            min_vals = self.min_vals

        if isinstance(self.max_vals, np.ndarray):
            max_vals = self.max_vals.min(axis, out, keepdims, initial, where)
        else:
            max_vals = self.max_vals

        return SingleEntityPhiTensor(
            child=data,
            max_vals=max_vals,
            min_vals=min_vals,
            entity=self.entity,
        )

    # TODO: Figure out how to do type annotation for dtype
    def trace(
        self,
        offset: int = 0,
        axis1: Optional[int] = 0,
        axis2: Optional[int] = 1,
        dtype: Optional[Any] = None,
        out: np.ndarray = None,
    ) -> SingleEntityPhiTensor:
        if isinstance(self.child, np.ndarray):
            data = self.child.trace(offset, axis1, axis2, dtype, out)
        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().trace(offset, axis1, axis2, dtype, out)
        else:
            data = self.child * len(self.child)

        if isinstance(self.min_vals, np.ndarray):
            mins = self.min_vals.trace(offset, axis1, axis2, dtype, out)
        else:
            mins = self.min_vals * len(self.child)

        if isinstance(self.max_vals, np.ndarray):
            maxes = self.max_vals.trace(offset, axis1, axis2, dtype, out)
        else:
            maxes = self.max_vals * len(self.child)

        return SingleEntityPhiTensor(
            child=data,
            min_vals=mins,
            max_vals=maxes,
            entity=self.entity,
        )

    def prod(
        self,
        axis: Optional[int] = None,
        dtype: Optional[Any] = None,
        out: Optional[np.ndarray] = None,
        keepdims: Optional[bool] = False,
        initial: int = 1,
        where: Optional[bool] = True,
    ) -> SingleEntityPhiTensor:
        return SingleEntityPhiTensor(
            child=self.child.prod(axis, dtype, out, keepdims, initial, where),
            min_vals=self.min_vals.prod(axis, dtype, out, keepdims, initial, where),
            max_vals=self.max_vals.prod(axis, dtype, out, keepdims, initial, where),
            entity=self.entity,
        )

    def round(self, decimals: int = 0) -> SingleEntityPhiTensor:
        if decimals != 0:
            raise Exception(
                "We currently only support np.int32. Sorry about the inconvenience-"
                "we plan to support more types soon!"
            )
        return SingleEntityPhiTensor(
            child=self.child.astype(dtype=np.int32),
            min_vals=self.min_vals.astype(dtype=np.int32),
            max_vals=self.max_vals.astype(dtype=np.int32),
            scalar_manager=self.scalar_manager,
            entity=self.entity,
        )

    def __floordiv__(
        self, other: Union[AcceptableSimpleType, SingleEntityPhiTensor]
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray) and not is_broadcastable(
                self.shape, other.shape
            ):
                raise Exception(
                    f"Shapes not broadcastable: {self.shape} and {other.shape}"
                )
            else:
                data = self.child // other
                mins = self.min_vals // other
                maxes = self.max_vals // other
        elif isinstance(other, SingleEntityPhiTensor):
            if is_broadcastable(self.shape, other.shape):
                if self.entity == other.entity:
                    data = self.child // other.child
                    mins = self.min_vals // other.min_vals
                    maxes = self.max_vals // other.max_vals
                else:
                    # return convert_to_gamma_tensor(self) // convert_to_gamma_tensor(other)
                    raise NotImplementedError
            else:
                raise Exception(
                    f"Shapes not broadcastable: {self.shape} and {other.shape}"
                )
        else:
            raise NotImplementedError
        return SingleEntityPhiTensor(
            child=data,
            max_vals=maxes,
            min_vals=mins,
            entity=self.entity,
            scalar_manager=self.scalar_manager,
        )

    #
    # # TODO: Check to see if non-integers are ever introduced
    # def __mod__(
    #     self, other: Union[AcceptableSimpleType, SingleEntityPhiTensor]
    # ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:
    #     if is_acceptable_simple_type(other):
    #         if isinstance(other, np.ndarray) and not is_broadcastable(
    #             self.shape, other.shape
    #         ):
    #             raise Exception(
    #                 f"Shapes not broadcastable: {self.shape} and {other.shape}"
    #             )
    #         else:
    #             data = self.child % other
    #             mins = self.min_vals % other
    #             maxes = self.max_vals % other
    #     elif isinstance(other, SingleEntityPhiTensor):
    #         if is_broadcastable(self.shape, other.shape):
    #             if self.entity == other.entity:
    #                 data = self.child % other.child
    #                 mins = self.min_vals % other.min_vals
    #                 maxes = self.max_vals % other.max_vals
    #             else:
    #                 # return convert_to_gamma_tensor(self) % convert_to_gamma_tensor(other)
    #                 raise NotImplementedError
    #         else:
    #             raise Exception(
    #                 f"Shapes not broadcastable: {self.shape} and {other.shape}"
    #             )
    #     else:
    #         raise NotImplementedError
    #     return SingleEntityPhiTensor(
    #         child=data,
    #         max_vals=maxes,
    #         min_vals=mins,
    #         entity=self.entity,
    #         scalar_manager=self.scalar_manager,
    #     )
    # #
    # # def __divmod__(
    # #     self, other: Union[AcceptableSimpleType, SingleEntityPhiTensor]
    # # ) -> TypeTuple:
    # #     return self // other, self % other

    def __matmul__(
        self, other: Union[np.ndarray, SingleEntityPhiTensor]
    ) -> Union[SingleEntityPhiTensor, IntermediateGammaTensor]:
        if not isinstance(other, (np.ndarray, SingleEntityPhiTensor)):
            raise Exception(
                f"Matrix multiplication not yet implemented for type {type(other)}"
            )
        else:
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Shapes not broadcastable: {self.shape} and {other.shape}"
                )
            else:
                if isinstance(other, np.ndarray):
                    data = self.child.__matmul__(other)
                    mins = self.min_vals.__matmul__(other)
                    maxes = self.max_vals.__matmul__(other)
                elif isinstance(other, SingleEntityPhiTensor):
                    if self.entity != other.entity:
                        # return convert_to_gamma_tensor(self).__matmul__(convert_to_gamma_tensor(other))
                        raise NotImplementedError
                    else:
                        data = self.child.__matmul__(other.child)
                        mins = self.min_vals.__matmul__(other.min_vals)
                        maxes = self.max_vals.__matmul__(other.max_vals)
                else:
                    raise NotImplementedError
                return SingleEntityPhiTensor(
                    child=data,
                    max_vals=maxes,
                    min_vals=mins,
                    entity=self.entity,
                    scalar_manager=self.scalar_manager,
                )

    def cumsum(
        self,
        axis: Optional[int] = None,
        dtype: Optional[Any] = None,
        out: np.ndarray = None,
    ) -> SingleEntityPhiTensor:
        if dtype and dtype != np.int32:
            raise Exception(
                "We currently only support np.int32 dtypes. "
                "We have plans to support more in the future though!"
            )
        if isinstance(self.child, np.ndarray):
            data = self.child.cumsum(axis, dtype, out)
        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().cumsum(axis, dtype, out)
        else:
            data = self.child * len(self.child)

        if isinstance(self.min_vals, np.ndarray):
            mins = self.min_vals.cumsum(axis, dtype, out)
        else:
            mins = self.min_vals * len(self.child)

        if isinstance(self.max_vals, np.ndarray):
            maxes = self.max_vals.cumsum(axis, dtype, out)
        else:
            maxes = self.max_vals * len(self.child)

        return SingleEntityPhiTensor(
            child=data, min_vals=mins, max_vals=maxes, entity=self.entity
        )

    def cumprod(
        self,
        axis: Optional[int] = None,
        dtype: Optional[Any] = None,
        out: np.ndarray = None,
    ) -> SingleEntityPhiTensor:
        if dtype and dtype != np.int32:
            raise Exception(
                "We currently only support np.int32 dtypes. "
                "We have plans to support more in the future though!"
            )
        if isinstance(self.child, np.ndarray):
            data = self.child.cumprod(axis, dtype, out)
        elif isinstance(self.child, torch.Tensor):
            data = self.child.numpy().cumprod(axis, dtype, out)
        else:
            data = self.child * len(self.child)

        if isinstance(self.min_vals, np.ndarray):
            mins = self.min_vals.cumprod(axis, dtype, out)
        else:
            mins = self.min_vals * len(self.child)

        if isinstance(self.max_vals, np.ndarray):
            maxes = self.max_vals.cumprod(axis, dtype, out)
        else:
            maxes = self.max_vals * len(self.child)

        return SingleEntityPhiTensor(
            child=data, min_vals=mins, max_vals=maxes, entity=self.entity
        )

    def _object2proto(self) -> SingleEntityPhiTensor_PB:
        proto_init_kwargs = {
            "min_vals": serialize(self._min_vals),
            "max_vals": serialize(self._max_vals),
        }

        # to de-duplicate entity and scalar manager at the RowEntityPhiTensor level
        # we need to allow this to sometimes be None and not store it
        if not getattr(self, "_remove_entity_scalar_manager", False):
            proto_init_kwargs["entity"] = serialize(self.entity)
            proto_init_kwargs["scalar_manager"] = serialize(self.scalar_manager)

        # either numpy array or ShareTensor
        if self.child is None:
            proto_init_kwargs["none"] = serialize(self.child)
        elif isinstance(self.child, np.ndarray):
            proto_init_kwargs["array"] = serialize(self.child)
        elif isinstance(self.child, torch.Tensor):
            proto_init_kwargs["array"] = serialize(np.array(self.child))
        else:
            proto_init_kwargs["tensor"] = serialize(self.child)

        return SingleEntityPhiTensor_PB(**proto_init_kwargs)

    @staticmethod
    def _proto2object(proto: SingleEntityPhiTensor_PB) -> SingleEntityPhiTensor:
        # either numpy array or ShareTensor
        if proto.HasField("tensor"):
            child = deserialize(proto.tensor)
        elif proto.HasField("array"):
            child = deserialize(proto.array)
        else:
            child = deserialize(proto.none)

        return SingleEntityPhiTensor(
            child=child,
            entity=deserialize(proto.entity) if proto.HasField("entity") else None,
            min_vals=deserialize(proto.min_vals),
            max_vals=deserialize(proto.max_vals),
            scalar_manager=deserialize(proto.scalar_manager)
            if proto.HasField("scalar_manager")
            else None,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SingleEntityPhiTensor_PB


@implements(SingleEntityPhiTensor, np.expand_dims)
def expand_dims(a: npt.ArrayLike, axis: Optional[int] = None) -> SingleEntityPhiTensor:

    entity = a.entity  # type: ignore

    min_vals = np.expand_dims(a=a.min_vals, axis=axis)  # type: ignore
    max_vals = np.expand_dims(a=a.max_vals, axis=axis)  # type: ignore

    data = np.expand_dims(a.child, axis=axis)  # type: ignore

    return SingleEntityPhiTensor(
        child=data,
        entity=entity,
        min_vals=min_vals,
        max_vals=max_vals,
        scalar_manager=a.scalar_manager,  # type: ignore
    )


@implements(SingleEntityPhiTensor, np.mean)
def mean(*args: Any, **kwargs: Any) -> SingleEntityPhiTensor:
    entity = args[0].entity
    scalar_manager = args[0].scalar_manager

    for arg in args[1:]:
        if not isinstance(arg, SingleEntityPhiTensor):
            raise Exception("Can only call np.mean on objects of the same type.")

        if arg.entity != entity:
            return NotImplemented

    min_vals = np.mean([x.min_vals for x in args], **kwargs)
    max_vals = np.mean([x.max_vals for x in args], **kwargs)

    args, kwargs = inputs2child(*args, **kwargs)  # type: ignore

    data = np.mean(args, **kwargs)

    return SingleEntityPhiTensor(
        child=data,
        entity=entity,
        min_vals=min_vals,
        max_vals=max_vals,
        scalar_manager=scalar_manager,
    )
