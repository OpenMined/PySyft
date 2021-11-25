# future
from __future__ import annotations

# stdlib
from collections.abc import Sequence
import operator
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Tuple as TypeTuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np
import numpy.typing as npt

# relative
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs
from ....core.adp.entity import DataSubjectGroup as DSG
from ....core.adp.entity import Entity
from ....proto.core.tensor.row_entity_phi_tensor_pb2 import (
    TensorWrappedRowEntityPhiTensorPointer as TensorWrappedRowEntityPhiTensorPointer_PB,
)
from ....util import inherit_tags
from ...adp.vm_private_scalar_manager import (
    VirtualMachinePrivateScalarManager as TypeScalarManager,
)
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize as serialize
from ...common.uid import UID
from ...node.abstract.node import AbstractNodeClient
from ...node.common.action.run_class_method_action import RunClassMethodAction
from ...pointer.pointer import Pointer
from ..broadcastable import is_broadcastable
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import implements  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from ..smpc import utils
from ..smpc.mpc_tensor import MPCTensor
from ..smpc.utils import TYPE_TO_RING_SIZE
from ..tensor import Tensor
from ..types import AcceptableSimpleType
from .adp_tensor import ADPTensor
from .initial_gamma import InitialGammaTensor
from .intermediate_gamma import IntermediateGammaTensor as IGT
from .single_entity_phi import SingleEntityPhiTensor
from .single_entity_phi import TensorWrappedSingleEntityPhiTensorPointer


@serializable()
class TensorWrappedRowEntityPhiTensorPointer(Pointer):
    """
    This tensor represents a pointer to a very specific tensor chain.

    Thus, this class has two groups of attributes: one set are the attributes for RowEntityPhiTensor:
        n_entities: int,
        unique_entities: set[Entity],

    And the others are for initializing a Pointer object:
        client=self.client,
        id_at_location=self.id_at_location,
        object_type=self.object_type,
        tags=self.tags,
        description=self.description,
    """

    __name__ = "TensorWrappedRowEntityPhiTensorPointer"
    __module__ = "syft.core.tensor.autodp.row_entity_phi"

    def __init__(
        self,
        unique_entities: set[Entity],
        n_entities: int,
        client: Any,  # union of all clients?
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

        self.unique_entities = unique_entities
        self.n_entities = n_entities
        self.public_shape = public_shape
        self.public_dtype = public_dtype

    # what to print in this case?
    @property
    def synthetic(self) -> np.ndarray:
        return (np.random.rand(*list(self.public_shape)) * 1).astype(  # type: ignore
            self.public_dtype
        )

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

        result = TensorWrappedRowEntityPhiTensorPointer(
            unique_entities=self.unique_entities,
            n_entities=self.n_entities,
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

        if isinstance(other, TensorWrappedRowEntityPhiTensorPointer):
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
                f"Invalid Type for TensorWrappedRowEntityPhiTensorPointer:{type(other)}"
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
        self: TensorWrappedRowEntityPhiTensorPointer,
        other: Union[
            TensorWrappedRowEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
        op_str: str,
    ) -> Union[MPCTensor, TensorWrappedRowEntityPhiTensorPointer]:
        """Performs the operation based on op_str

        Args:
            other (Union[TensorWrappedRowEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]): second operand.

        Returns:
            Tuple[MPCTensor,Union[MPCTensor,int,float,np.ndarray]] : Result of the operation
        """
        op = getattr(operator, op_str)

        if (
            isinstance(other, TensorWrappedRowEntityPhiTensorPointer)
            and self.client != other.client
        ):

            parties = [self.client, other.client]

            self_mpc = MPCTensor(secret=self, shape=self.public_shape, parties=parties)
            other_mpc = MPCTensor(
                secret=other, shape=other.public_shape, parties=parties
            )

            return op(self_mpc, other_mpc)

        elif isinstance(other, TensorWrappedSingleEntityPhiTensorPointer):
            # should we check if other.client is present in self.client?
            pass

        elif isinstance(other, MPCTensor):

            return op(other, self)

        return self._apply_tensor_op(other=other, op_str=op_str)

    def __add__(
        self,
        other: Union[
            TensorWrappedRowEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedRowEntityPhiTensorPointer, MPCTensor]:
        """Apply the "add" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedRowEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedRowEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedRowEntityPhiTensorPointer._apply_op(self, other, "add")

    def __sub__(
        self,
        other: Union[
            TensorWrappedRowEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedRowEntityPhiTensorPointer, MPCTensor]:
        """Apply the "sub" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedRowEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedRowEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedRowEntityPhiTensorPointer._apply_op(self, other, "sub")

    def __mul__(
        self,
        other: Union[
            TensorWrappedRowEntityPhiTensorPointer, MPCTensor, int, float, np.ndarray
        ],
    ) -> Union[TensorWrappedRowEntityPhiTensorPointer, MPCTensor]:
        """Apply the "mul" operation between "self" and "other"

        Args:
            y (Union[TensorWrappedRowEntityPhiTensorPointer,MPCTensor,int,float,np.ndarray]) : second operand.

        Returns:
            Union[TensorWrappedRowEntityPhiTensorPointer,MPCTensor] : Result of the operation.
        """
        return TensorWrappedRowEntityPhiTensorPointer._apply_op(self, other, "mul")

    def to_local_object_without_private_data_child(self) -> RowEntityPhiTensor:
        """Convert this pointer into a partial version of the RowEntityPhiTensor but without
        any of the private data therein."""

        public_shape = getattr(self, "public_shape", None)
        public_dtype = getattr(self, "public_dtype", None)
        return Tensor(
            child=RowEntityPhiTensor(
                rows=None,
            ),
            public_shape=public_shape,
            public_dtype=public_dtype,
        )

    def _object2proto(self) -> "TensorWrappedRowEntityPhiTensorPointer_PB":

        _unique_entities = list(map(lambda x: serialize(x), self.unique_entities))
        _n_entities = self.n_entities
        _location = serialize(self.client.address)
        _id_at_location = serialize(self.id_at_location)
        _object_type = self.object_type
        _tags = self.tags
        _description = self.description
        _public_shape = serialize(getattr(self, "public_shape", None), to_bytes=True)
        _public_dtype = serialize(getattr(self, "public_dtype", None), to_bytes=True)

        return TensorWrappedRowEntityPhiTensorPointer_PB(
            unique_entities=_unique_entities,
            n_entities=_n_entities,
            location=_location,
            id_at_location=_id_at_location,
            object_type=_object_type,
            tags=_tags,
            description=_description,
            public_shape=_public_shape,
            public_dtype=_public_dtype,
        )

    @staticmethod
    def _proto2object(
        proto: TensorWrappedRowEntityPhiTensorPointer_PB,
    ) -> "TensorWrappedRowEntityPhiTensorPointer":

        unique_entities = set(map(lambda x: deserialize(blob=x), proto.unique_entities))
        n_entities = proto.n_entities
        client = deserialize(blob=proto.location)
        id_at_location = deserialize(blob=proto.id_at_location)
        object_type = proto.object_type
        tags = proto.tags
        public_shape = deserialize(blob=proto.public_shape, from_bytes=True)
        public_dtype = deserialize(blob=proto.public_dtype, from_bytes=True)
        description = proto.description

        return TensorWrappedRowEntityPhiTensorPointer(
            unique_entities=unique_entities,
            n_entities=n_entities,
            client=client,
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

        return TensorWrappedRowEntityPhiTensorPointer_PB


@serializable(recursive_serde=True)
class RowEntityPhiTensor(PassthroughTensor, ADPTensor):
    """This tensor is one of several tensors whose purpose is to carry metadata
    relevant to automatically tracking the privacy budgets of tensor operations. This
    tensor is called 'Phi' tensor because it assumes that each number in the tensor
    originates from a single entity (no numbers originate from multiple entities). This
    tensor is called 'RowEntity' because it additionally assumes that all entries in a row
    come from the same entity (note: multiple rows may also be from the same or different
    entities). The reason we have a tensor specifically for tracking row-organized entities
    is that data entity-linked by row is very common and specifically accommodating it offers
    significant performance benefits over other DP tracking tensors. Note that when
    we refer to the number of 'rows' we simply refer to the length of the first dimension. This
    tensor can have an arbitrary number of dimensions."""

    PointerClassOverride = TensorWrappedRowEntityPhiTensorPointer

    # a list of attributes needed for serialization using RecursiveSerde
    __attr_allowlist__ = ["child", "n_entities", "unique_entities"]

    def __init__(self, rows: Optional[Sequence], check_shape: bool = True):
        """Initialize a RowEntityPhiTensor

        rows: the actual data organized as an iterable (can be any type of iterable)
        check_shape: whether or not we are already confident that the objects in iterable
            'rows' all have the same dimension (check if we're not sure).

        """
        # Container type heirachy: https://docs.python.org/3/library/collections.abc.html
        self.child: Sequence
        super().__init__(rows)

        # include this check because it's expensive to check and sometimes we can skip it when
        # we already know the rows are identically shaped.
        if check_shape and rows is not None:

            # shape of the first row we use for reference
            shape = rows[0].shape

            # check each row to make sure it's the same shape as the first
            for row in rows[1:]:
                if shape != row.shape:
                    raise Exception(
                        f"All rows in RowEntityPhiTensor must match: {shape} != {row.shape}"
                    )

        """Calculate the number of unique entities behind the REPT"""
        self.unique_entities: set[Entity] = set()
        self.n_entities = 0
        for entity in self.entities.flatten():
            if isinstance(entity, Entity):
                if entity not in self.unique_entities:
                    self.unique_entities.add(entity)
                    self.n_entities += 1
                else:
                    continue
            elif isinstance(entity, DSG):
                for e in entity.entity_set:
                    if e not in self.unique_entities:
                        self.unique_entities.add(e)
                        self.n_entities += 1
                    else:
                        continue
            else:
                raise Exception(f"{type(entity)}")

    def init_pointer(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> TensorWrappedRowEntityPhiTensorPointer:
        return TensorWrappedRowEntityPhiTensorPointer(
            # Arguments specifically for REPhiTensor
            unique_entities=self.unique_entities,
            n_entities=self.n_entities,
            # Arguments required for a Pointer to work
            client=client,
            id_at_location=id_at_location,
            object_type=object_type,
            tags=tags,
            description=description,
        )

    @property
    def scalar_manager(self) -> TypeScalarManager:
        return self.child[0].scalar_manager

    @property
    def min_vals(self) -> np.ndarray:
        return np.concatenate([x.min_vals for x in self.child]).reshape(self.shape)

    @property
    def max_vals(self) -> np.ndarray:
        return np.concatenate([x.max_vals for x in self.child]).reshape(self.shape)

    @property
    def value(self) -> np.ndarray:
        return np.concatenate([x.child for x in self.child]).reshape(self.shape)

    @property
    def entities(self) -> np.ndarray:
        # we must cast the result of x.shape prod to an int because sometimes
        # it can be a float depending on the original type which fails to
        # multiply the [x.entity] array
        return np.array(
            [[x.entity] * int(np.array(x.shape).prod()) for x in self.child]
        ).reshape(self.shape)

    @property
    def gamma(self) -> InitialGammaTensor:
        return self.create_gamma()

    @staticmethod
    def convert_to_gamma(input_list: List) -> IGT:
        """This converts a REPT's data into a GammaTensor without having to initialize it. Used in comparison ops"""
        values = []
        entities = []
        mins = []
        maxes = []

        if len(input_list) > 1:
            target_shape = [len(input_list)] + list(input_list[0].shape)
        else:
            print(type(input_list), input_list)
            target_shape = input_list[0].shape

        for tensor in input_list:
            if isinstance(tensor, SingleEntityPhiTensor):
                values.append(tensor.child)
                entity_array = np.array(tensor.entity, dtype=object).repeat(
                    len(tensor.child.flatten())
                )
                entities.append(entity_array.reshape(tensor.shape))
                mins.append(tensor.min_vals)
                maxes.append(tensor.max_vals)
            elif isinstance(tensor, IGT):
                values.append(tensor._values())
                entities.append(tensor._entities())
                mins.append(tensor._min_values())
                maxes.append(tensor._max_values())
            else:
                raise Exception(f"Unknown type in REPT: {type(tensor)}")
        return InitialGammaTensor(
            values=np.concatenate(values).reshape(target_shape),
            entities=np.concatenate(entities).reshape(target_shape),
            min_vals=np.concatenate(mins).reshape(target_shape),
            max_vals=np.concatenate(maxes).reshape(target_shape),
            scalar_manager=input_list[0].scalar_manager,
        )

    def create_gamma(
        self, scalar_manager: Optional[TypeScalarManager] = None
    ) -> InitialGammaTensor:

        if scalar_manager is None:
            scalar_manager = self.scalar_manager

        return InitialGammaTensor(
            values=self.value,  # 5 x 10 data
            min_vals=self.min_vals,  # 5 x 10 minimum values
            max_vals=self.max_vals,  # 5 x 10 maximum values
            entities=self.entities,  # list of 5 entities
            scalar_manager=scalar_manager,
        )

    @property
    def shape(self) -> Tuple[Any, ...]:
        return tuple([len(self.child)] + list(self.child[0].shape))

    def __eq__(self, other: Any) -> Union[RowEntityPhiTensor, IGT]:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            gamma_output = False
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] == other)
                else:
                    result = self.child[i] == other.child[i]
                    if isinstance(result, InitialGammaTensor):
                        gamma_output = True
                    new_list.append(result)
            if not gamma_output:
                return RowEntityPhiTensor(rows=new_list, check_shape=False)
            else:
                return RowEntityPhiTensor.convert_to_gamma(new_list)
        else:
            raise Exception(
                f"Tensor dims do not match for __eq__: {len(self.child)} != {len(other.child)}"
            )

    def __ne__(self, other: Any) -> Union[RowEntityPhiTensor, IGT]:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            gamma_output = False
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] != other)
                else:
                    result = self.child[i] != other.child[i]
                    if isinstance(result, InitialGammaTensor):
                        gamma_output = True
                    new_list.append(result)
            if not gamma_output:
                return RowEntityPhiTensor(rows=new_list, check_shape=False)
            else:
                return RowEntityPhiTensor.convert_to_gamma(new_list)
        else:
            raise Exception(
                f"Tensor dims do not match for __ne__: {len(self.child)} != {len(other.child)}"
            )

    def __add__(
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:
        # TODO: Catch unacceptable types (str, dict, etc) to avoid errors for other.child below
        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] + other)
                else:
                    # Private/Public and Private/Private are handled by the underlying SEPT self.child objects.
                    new_list.append(self.child[i] + other.child[i])
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            # Broadcasting is possible, but we're skipping that for now.
            raise Exception(
                f"Tensor dims do not match for __add__: {len(self.child)} != {len(other.child)}"
            )

    def __sub__(
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:
        # TODO: Catch unacceptable types (str, dict, etc) to avoid errors for other.child below
        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] - other)
                else:
                    new_list.append(self.child[i] - other.child[i])
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __sub__: {len(self.child)} != {len(other.child)}"
            )

    def __mul__(
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:
        new_list = list()
        if is_acceptable_simple_type(other):
            if isinstance(other, np.ndarray):
                if is_broadcastable(self.shape, other.shape):
                    new_list.append(
                        [self.child[i] * other[i] for i in range(len(self.child))]
                    )
                else:
                    raise Exception(
                        f"Tensor dims do not match for __sub__: {getattr(self.child, 'shape', None)} != {other.shape}"
                    )
            else:  # int, float, bool, etc
                new_list = [child * other for child in self.child]
        elif isinstance(other, RowEntityPhiTensor):
            if is_broadcastable(self.shape, other.shape):
                for self_child, other_child in zip(self.child, other.child):
                    if len(self.child) != len(other.child):
                        raise ValueError(
                            "Zipping two different lengths will drop data. "
                            + f"{len(self.child)} vs {len(other.child)}"
                        )
                    new_list.append(self_child * other_child)
            else:
                raise Exception(
                    f"Tensor dims do not match for __sub__: {self.shape} != {other.shape}"
                )
        elif isinstance(other, SingleEntityPhiTensor):
            for child in self.child:
                # If even a single SEPT in the REPT isn't broadcastable, the multiplication operation doesn't work
                if not is_broadcastable(child.shape, other.shape):
                    raise Exception(
                        f"Tensor dims do not match for __sub__: {self.shape} != {other.shape}"
                    )
            new_list = [i * other for i in self.child]
        else:
            raise NotImplementedError
        return RowEntityPhiTensor(rows=new_list)

        # if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
        #     new_list = list()
        #     for i in range(len(self.child)):
        #         if is_acceptable_simple_type(other):
        #             if isinstance(other, (int, bool, float)):
        #                 new_list.append(self.child[i] * other)
        #             else:
        #                 new_list.append(self.child[i] * other[i])
        #         else:
        #             if isinstance(other, RowEntityPhiTensor):
        #                 new_list.append(self.child[i] * other.child[i])
        #             elif isinstance(other, SingleEntityPhiTensor):
        #
        #                 new_list.append(self.child[i] * other)
        #     return RowEntityPhiTensor(rows=new_list, check_shape=False)
        # else:
        #     raise Exception(
        #         f"Tensor dims do not match for __mul__: {len(self.child)} != {len(other.child)}"
        #     )

    def __pos__(self) -> RowEntityPhiTensor:
        return RowEntityPhiTensor(rows=[+x for x in self.child], check_shape=False)

    def __neg__(self) -> RowEntityPhiTensor:
        return RowEntityPhiTensor(rows=[-x for x in self.child], check_shape=False)

    def __or__(self, other: Any) -> RowEntityPhiTensor:
        return RowEntityPhiTensor(
            rows=[x | other for x in self.child], check_shape=False
        )

    def __and__(self, other: Any) -> RowEntityPhiTensor:
        return RowEntityPhiTensor(
            rows=[x & other for x in self.child], check_shape=False
        )

    def __truediv__(
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] / other)
                else:
                    new_list.append(self.child[i] / other.child[i])
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __truediv__: {len(self.child)} != {len(other.child)}"
            )

    def repeat(
        self, repeats: Union[int, List[int]], axis: Optional[int] = None
    ) -> RowEntityPhiTensor:

        if not isinstance(repeats, int):
            raise Exception(
                f"{type(self)}.repeat for repeats: List {repeats} not implemented yet"
            )

        if axis is None:
            raise Exception(
                "Conservatively, RowEntityPhiTensor doesn't yet support repeat(axis=None)"
            )

        if axis == 0 or axis == -len(self.shape):
            new_list = list()
            for _ in range(repeats):
                for row in self.child:
                    new_list.append(row)
            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        elif axis > 0:
            new_list = list()
            for row in self.child:
                new_list.append(row.repeat(repeats, axis=axis - 1))
            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        # axis is negative
        elif abs(axis) < len(self.shape):
            new_list = list()
            for row in self.child:
                new_list.append(row.repeat(repeats, axis=axis))
            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                "'axis' arg is negative and strangely large... not sure what to do."
            )

    def flatten(self, order: Optional[str] = "C") -> RowEntityPhiTensor:
        new_list = list()
        for tensor in self.child:
            new_list.append(tensor.flatten(order))
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def ravel(self, order: Optional[str] = "C") -> RowEntityPhiTensor:
        new_list = list()
        for tensor in self.child:
            new_list.append(tensor.ravel(order))
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def swapaxes(self, axis1: int, axis2: int) -> RowEntityPhiTensor:
        if axis1 == 0 or axis2 == 0:
            raise Exception(
                "For now, you can't swap the first axis b/c that would "
                "probably create a Gamma Tensor. Sorry about that!"
            )
        new_list = list()
        for tensor in self.child:
            # Axis=0 for REPT.child is Axis=1 for REPT, so subtract 1
            new_list.append(tensor.swapaxes(axis1 - 1, axis2 - 1))
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def squeeze(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> RowEntityPhiTensor:
        if axis == 0:
            # If the first axis can be squeezed then there is only one
            # tensor in the REPT, as such it might be a SEPT
            # TODO: Check if the output type is still a REPT
            # if isinstance(self.child[0], SEPT): return self.child[0]
            return RowEntityPhiTensor(rows=self.child[0])
        else:
            new_list = list()
            for tensor in self.child:
                new_list.append(tensor.squeeze(axis))
            self.child = new_list

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def reshape(
        self,
        shape: Union[
            int,
            Union[Sequence[int], Sequence[Sequence[int]]],
        ],
    ) -> RowEntityPhiTensor:
        if isinstance(shape, int):
            raise Exception(
                f"{type(self)}.reshape for shape: int {shape} is not implemented"
            )
        # This is to fix the bug where shape = ([a, b, c], )
        if isinstance(shape[0], Sequence):
            shape = shape[0]

        if shape[0] != self.shape[0]:
            raise Exception(
                "For now, you can't reshape the first dimension because that would"
                + "probably require creating a gamma tensor."
                + str(shape)
                + " and "
                + str(self.shape)
            )

        new_list = list()
        for row in self.child:
            new_list.append(row.reshape(shape[1:]))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def resize(
        self,
        new_shape: Union[int, Tuple[int, ...]],
        refcheck: Optional[bool] = True,
    ) -> None:
        """This method is identical to reshape, but it modifies the Tensor in-place instead of returning a new one"""
        if isinstance(new_shape, int):
            raise Exception(f"new_shape: {new_shape} must be a Tuple for {type(self)}")
        if new_shape[0] != self.shape[0]:
            raise Exception(
                "For now, you can't reshape the first dimension because that would"
                + "probably require creating a gamma tensor."
            )

        new_list = list()
        for row in self.child:
            new_list.append(row.reshape(new_shape[1:]))

        # Modify the tensor data in-place instead of returning a new one.
        self.child = new_list

    def compress(
        self,
        condition: List[bool],
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
    ) -> RowEntityPhiTensor:
        # TODO: Could any conditions result in GammaTensors being formed?
        # TODO: Will min/max vals change upon filtering? I don't think so, since they're data independent
        new_list = list()
        for tensor in self.child:
            new_list.append(tensor.compress(condition, axis, out))
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def partition(
        self,
        kth: Union[int, Tuple[int, ...]],
        axis: Optional[int] = -1,
        kind: Optional[str] = "introselect",
        order: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> RowEntityPhiTensor:
        if axis == 0:  # Unclear how to sort the SEPTs in a REPT
            raise NotImplementedError
        new_list = list()
        for tensor in self.child:
            new_list.append(tensor.partition(kth, axis, kind, order))
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    # Since this is being used differently compared to supertype, ignoring type annotation errors
    def sum(self, *args: Any, **kwargs: Any) -> RowEntityPhiTensor:

        new_list = list()
        for row in self.child:
            new_list.append(row.sum(*args, **kwargs))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    # Since this is being used differently compared to supertype, ignoring type annotation errors
    def transpose(self, *dims: Optional[Any]) -> RowEntityPhiTensor:
        if dims:
            if dims[0] != 0:
                raise Exception("Can't move dim 0 in RowEntityPhiTensor at this time")

            new_dims = list(np.array(dims[1:]))

            new_list = list()
            for row in self.child:
                new_list.append(row.transpose(*new_dims))
        else:
            new_list = list()
            for row in self.child:
                new_list.append(row.transpose())
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def __le__(self, other: Any) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            gamma_output = False
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] <= other)
                else:
                    result = self.child[i] <= other.child[i]
                    if isinstance(result, InitialGammaTensor):
                        gamma_output = True
                    new_list.append(result)
            if not gamma_output:
                return RowEntityPhiTensor(rows=new_list, check_shape=False)
            else:
                return RowEntityPhiTensor.convert_to_gamma(new_list)
        else:
            raise Exception(
                f"Tensor dims do not match for __le__: {len(self.child)} != {len(other.child)}"
            )

    def __lt__(self, other: Any) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            gamma_output = False
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] < other)
                else:
                    result = self.child[i] < other.child[i]
                    if isinstance(result, InitialGammaTensor):
                        gamma_output = True
                    new_list.append(result)
            if not gamma_output:
                return RowEntityPhiTensor(rows=new_list, check_shape=False)
            else:
                return RowEntityPhiTensor.convert_to_gamma(new_list)
        else:
            raise Exception(
                f"Tensor dims do not match for __lt__: {len(self.child)} != {len(other.child)}"
            )

    def __gt__(self, other: Any) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            gamma_output = False
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] > other)
                else:
                    result = self.child[i] > other.child[i]
                    if isinstance(result, InitialGammaTensor):
                        gamma_output = True
                    new_list.append(result)
            if not gamma_output:
                return RowEntityPhiTensor(rows=new_list, check_shape=False)
            else:
                return RowEntityPhiTensor.convert_to_gamma(new_list)
        else:
            raise Exception(
                f"Tensor dims do not match for __gt__: {len(self.child)} != {len(other.child)}"
            )

    def __ge__(self, other: Any) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            gamma_output = False
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] >= other)
                else:
                    result = self.child[i] >= other.child[i]
                    if isinstance(result, InitialGammaTensor):
                        gamma_output = True
                    new_list.append(result)
            if not gamma_output:
                return RowEntityPhiTensor(rows=new_list, check_shape=False)
            else:
                return RowEntityPhiTensor.convert_to_gamma(new_list)
        else:
            raise Exception(
                f"Tensor dims do not match for __ge__: {len(self.child)} != {len(other.child)}"
            )

    def cumprod(self, axis: Optional[int] = None) -> RowEntityPhiTensor:
        new_list = list()
        for tensor in self.child:
            new_list.append(tensor.cumprod(axis))
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def cumsum(self, axis: Optional[int] = None) -> RowEntityPhiTensor:
        new_list = list()
        for tensor in self.child:
            new_list.append(tensor.cumsum(axis))
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def clip(
        self, a_min: npt.ArrayLike, a_max: npt.ArrayLike, *args: Any
    ) -> RowEntityPhiTensor:

        if a_min is None and a_max is None:
            raise Exception("ValueError: clip: must set either max or min")

        new_list = list()
        for row in self.child:
            new_list.append(row.clip(a_min=a_min, a_max=a_max, *args))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def __floordiv__(
        self,
        other: Union[AcceptableSimpleType, SingleEntityPhiTensor, RowEntityPhiTensor],
    ) -> RowEntityPhiTensor:

        # We will let the underlying SingleEntityPhiTensor logic handle most of the errors/exceptions
        new_list = list()
        if is_acceptable_simple_type(other):
            for tensor in self.child:
                new_list.append(tensor // other)
        elif isinstance(other, SingleEntityPhiTensor):
            for tensor in self.child:
                new_list.append(tensor // other)
        elif isinstance(other, RowEntityPhiTensor):
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Shapes not broadcastable: {self.shape}, {other.shape}"
                )
            else:
                if other.shape[0] == 1:
                    for tensor in self.child:
                        new_list.append(tensor // other.child[0])
                else:
                    if len(self.child) != len(other.child):
                        raise ValueError(
                            "Zipping two different lengths will drop data. "
                            + f"{len(self.child)} vs {len(other.child)}"
                        )
                    for self_tensors, other_tensors in zip(self.child, other.child):
                        new_list.append(self_tensors // other_tensors)
        else:
            raise NotImplementedError
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def __mod__(
        self,
        other: Union[AcceptableSimpleType, SingleEntityPhiTensor, RowEntityPhiTensor],
    ) -> RowEntityPhiTensor:

        # We will let the underlying SingleEntityPhiTensor logic handle most of the errors/exceptions
        new_list = list()
        if is_acceptable_simple_type(other):
            for tensor in self.child:
                new_list.append(tensor % other)
        elif isinstance(other, SingleEntityPhiTensor):
            for tensor in self.child:
                new_list.append(tensor % other)
        elif isinstance(other, RowEntityPhiTensor):
            if not is_broadcastable(self.shape, other.shape):
                raise Exception(
                    f"Shapes not broadcastable: {self.shape}, {other.shape}"
                )
            else:
                if other.shape[0] == 1:
                    for tensor in self.child:
                        new_list.append(tensor % other.child[0])
                else:
                    if len(self.child) != len(other.child):
                        raise ValueError(
                            "Zipping two different lengths will drop data. "
                            + f"{len(self.child)} vs {len(other.child)}"
                        )
                    for self_tensors, other_tensors in zip(self.child, other.child):
                        new_list.append(self_tensors % other_tensors)
        else:
            raise NotImplementedError
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def __divmod__(
        self,
        other: Union[AcceptableSimpleType, SingleEntityPhiTensor, RowEntityPhiTensor],
    ) -> Tuple:
        # Let logic written out in mod, floordiv and SEPT handle all exceptions
        return self // other, self % other

    def __matmul__(
        self,
        other: Union[AcceptableSimpleType, SingleEntityPhiTensor, RowEntityPhiTensor],
    ) -> RowEntityPhiTensor:
        new_list = list()
        if isinstance(other, np.ndarray):
            for tensor in self.child:
                new_list.append(tensor.__matmul__(other))
        elif isinstance(other, SingleEntityPhiTensor):
            # Whether the output of the matmul is SEPT or IGT, we let SEPT code determine that
            for tensor in self.child:
                new_list.append(tensor.__matmul__(other))
        elif isinstance(other, RowEntityPhiTensor):
            if len(self.child) != len(other.child):
                raise ValueError(
                    "Zipping two different lengths will drop data. "
                    + f"{len(self.child)} vs {len(other.child)}"
                )
            for self_tensor, other_tensor in zip(self.child, other.child):
                new_list.append(self_tensor.__matmul__(other_tensor))
        else:
            raise NotImplementedError
        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def trace(
        self,
        offset: int = 0,
        axis1: int = 1,
        axis2: int = 2,
        dtype: Optional[Any] = np.int32,
        out: Optional[np.ndarray] = None,
    ) -> RowEntityPhiTensor:
        if axis1 == 0 or axis2 == 0:
            raise NotImplementedError  # This would create a GammaTensor
        if dtype != np.int32:
            raise Exception(
                "We currently only support np.int32 dtypes for our tensors. "
                "We will be adding support for more dtypes soon! Sorry for the inconvenience."
            )

        # Axis #1 for REPT = Axis #0 for its SEPT child, etc
        axis1 -= 1
        axis2 -= 1
        new_list = list()
        for tensor in self.child:
            new_list.append(tensor.trace(offset, axis1, axis2, dtype, out))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def prod(
        self,
        axis: int = 1,  # might be bad that the default behaviour currently is not implemented...
        dtype: Optional[Any] = None,
        out: Optional[np.ndarray] = None,
        keepdims: Optional[bool] = False,
        initial: int = 1,
        where: Optional[bool] = True,
    ) -> RowEntityPhiTensor:
        if dtype and dtype != np.int32:
            raise Exception(
                "We currently only support np.int32 dtypes for our tensors. "
                "We will be adding support for more dtypes soon! Sorry for the inconvenience."
            )
        if axis == 0:
            raise NotImplementedError  # GammaTensor
        new_list = list()
        for tensor in self.child:
            new_list.append(tensor.prod(axis - 1, dtype, out, keepdims, initial, where))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def any(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        where: Optional[bool] = True,
    ) -> RowEntityPhiTensor:
        """Test whether any element along a given axis evaluates to True"""

        new_list = list()
        for row in self.child:
            new_list.append(row.any(axis, keepdims, where))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def all(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        where: Optional[bool] = True,
    ) -> RowEntityPhiTensor:
        """Test whether all elements along a given axis evaluates to True"""

        new_list = list()
        for row in self.child:
            new_list.append(row.all(axis, keepdims, where))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def abs(
        self,
        out: Optional[np.ndarray] = None,
    ) -> RowEntityPhiTensor:
        """Calculate the absolute value element-wise"""

        new_list = list()
        for row in self.child:
            new_list.append(row.abs(out))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def pow(
        self, value: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:
        """Return elements raised to powers from value, element-wise"""
        new_list = list()
        if is_acceptable_simple_type(value):
            if isinstance(value, np.ndarray):
                new_list.append(
                    [self.child[i].pow(value[i]) for i in range(len(self.child))]
                )
            else:  # int, float, bool, etc
                new_list = [child.pow(value) for child in self.child]
        elif isinstance(value, RowEntityPhiTensor):
            new_list = [
                self.child[i].pow(value.child[i]) for i in range(len(self.child))
            ]
        elif isinstance(value, SingleEntityPhiTensor):
            new_list = [i.pow(value) for i in self.child]
        else:
            raise NotImplementedError

        return RowEntityPhiTensor(rows=new_list)

    def copy(
        self, order: Optional[str] = "K", subok: Optional[bool] = True
    ) -> RowEntityPhiTensor:
        """Return copy of the given object"""
        new_list = list()
        for row in self.child:
            new_list.append(row.copy(order=order, subok=subok))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def take(
        self,
        indices: np.ArrayLike,
        axis: Optional[int] = None,
        mode: Optional[str] = "raise",
    ) -> RowEntityPhiTensor:
        """Take elements from an array along an axis"""
        new_list = list()
        for row in self.child:
            new_list.append(row.take(indices=indices, axis=axis, mode=mode))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def diagonal(
        self,
        offset: Optional[int] = 0,
        axis1: Optional[int] = 0,
        axis2: Optional[int] = 1,
    ) -> RowEntityPhiTensor:
        """Return specified diagonals"""
        new_list = list()
        for row in self.child:
            new_list.append(row.diagonal(offset=offset, axis1=axis1, axis2=axis2))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def round(self, decimals: int = 0) -> RowEntityPhiTensor:
        new_list = list()
        for tensor in self.child:
            new_list.append(tensor.round(decimals))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)


@implements(RowEntityPhiTensor, np.expand_dims)
def expand_dims(a: np.typing.ArrayLike, axis: int) -> RowEntityPhiTensor:

    if axis == 0:
        raise Exception(
            "Currently, we don't have functionality for axis=0 but we could with a bit more work."
        )

    new_rows = list()
    for row in a.child:
        new_rows.append(np.expand_dims(row, axis - 1))

    return RowEntityPhiTensor(rows=new_rows, check_shape=False)
