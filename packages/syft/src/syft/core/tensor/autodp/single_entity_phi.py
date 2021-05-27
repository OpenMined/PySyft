# future
from __future__ import annotations

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np

# syft relative
from ....core.common.serde.serializable import Serializable
from ....lib.util import full_name_with_name
from ....proto.core.tensor.tensor_pb2 import Tensor as Tensor_PB
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import bind_protobuf
from ...common.serde.serialize import _serialize as serialize
from ..ancestors import AutogradTensorAncestor
from ..passthrough import PassthroughTensor
from ..passthrough import implements
from ..passthrough import inputs2child
from ..passthrough import is_acceptable_simple_type


@bind_protobuf
class SingleEntityPhiTensor(PassthroughTensor, AutogradTensorAncestor, Serializable):
    def __init__(
        self,
        child,
        entity,
        min_vals,
        max_vals,
        scalar_manager=VirtualMachinePrivateScalarManager(),
    ):
        super().__init__(child)

        self.entity = entity
        self._min_vals = min_vals
        self._max_vals = max_vals
        self.scalar_manager = scalar_manager

    @property
    def min_vals(self):
        return self._min_vals

    @property
    def max_vals(self):
        return self._max_vals

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(entity={self.entity.name}, child={self.child})"
        )

    def __add__(self, other):

        if isinstance(other, SingleEntityPhiTensor):

            if self.entity != other.entity:
                # this should return a GammaTensor
                return NotImplemented

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

        else:
            return NotImplemented

    def __sub__(self, other):

        if isinstance(other, SingleEntityPhiTensor):

            if self.entity != other.entity:
                # this should return a GammaTensor
                return NotImplemented

            data = self.child - other.child
            min_vals = self.min_vals - other.min_vals
            max_vals = self.max_vals - other.max_vals
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

    def __mul__(self, other):

        if other.__class__ == SingleEntityPhiTensor:

            if self.entity != other.entity:
                # this should return a GammaTensor
                return NotImplemented

            data = self.child * other.child

            min_min = self.min_vals * other.min_vals
            min_max = self.min_vals * other.max_vals
            max_min = self.max_vals * other.min_vals
            max_max = self.max_vals * other.max_vals

            min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)
            max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)
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

            min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)
            max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)
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

    def __truediv__(self, other):

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

                min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)
                max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)

            entity = self.entity

            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )
        else:
            return self * (1 / other)

    def repeat(self, repeats, axis=None):

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

    def reshape(self, *args):

        data = self.child.reshape(*args)
        min_vals = self.min_vals.reshape(*args)
        max_vals = self.max_vals.reshape(*args)
        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    # ndarray.flatten(order='C')
    def flatten(self, order: str = "C") -> SingleEntityPhiTensor:
        data = self.child.flatten(order=order)
        min_vals = self.min_vals.flatten(order=order)
        max_vals = self.max_vals.flatten(order=order)
        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def __getitem__(self, key) -> SingleEntityPhiTensor:
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

    def sum(self, *args, **kwargs):

        data = self.child.sum(*args, **kwargs)
        min_vals = self.min_vals.sum(*args, **kwargs)
        max_vals = self.max_vals.sum(*args, **kwargs)
        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def dot(self, other):
        return self.manual_dot(other)

    def transpose(self, *args, **kwargs):

        data = self.child.transpose(*args)
        min_vals = self.min_vals.transpose(*args)
        max_vals = self.max_vals.transpose(*args)
        entity = self.entity

        return SingleEntityPhiTensor(
            child=data,
            entity=entity,
            min_vals=min_vals,
            max_vals=max_vals,
            scalar_manager=self.scalar_manager,
        )

    def _object2proto(self) -> Tensor_PB:
        arrays = []
        tensors = []
        if isinstance(self.child, np.ndarray):
            use_tensors = False
            arrays = [
                serialize(self.child),
                serialize(self.min_vals),
                serialize(self.max_vals),
            ]
        else:
            use_tensors = True
            tensors = [
                serialize(self.child),
                serialize(self.min_vals),
                serialize(self.max_vals),
            ]

        return Tensor_PB(
            obj_type=full_name_with_name(klass=type(self)),
            use_tensors=use_tensors,
            arrays=arrays,
            tensors=tensors,
            entity=serialize(self.entity),
        )

    @staticmethod
    def _proto2object(proto: Tensor_PB) -> SingleEntityPhiTensor:
        use_tensors = proto.use_tensors
        children = []
        if use_tensors:
            children = [deserialize(tensor) for tensor in proto.tensors]
        else:
            children = [deserialize(array) for array in proto.arrays]

        child = children.pop(0)
        min_vals = children.pop(0)
        max_vals = children.pop(0)

        entity = deserialize(blob=proto.entity)

        return SingleEntityPhiTensor(
            child=child, entity=entity, min_vals=min_vals, max_vals=max_vals
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tensor_PB


# @implements(SingleEntityPhiTensor, np.min)
# def npmax(*args, **kwargs):
#     print("mining1")
#     args, kwargs = inputs2child(*args, **kwargs)
#     return np.min(*args, **kwargs)


@implements(SingleEntityPhiTensor, np.mean)
def mean(*args, **kwargs):
    entity = args[0].entity
    scalar_manager = args[0].scalar_manager

    for arg in args[1:]:
        if not isinstance(arg, SingleEntityPhiTensor):
            raise Exception("Can only call np.mean on objects of the same type.")

        if arg.entity != entity:
            return NotImplemented

    min_vals = np.mean([x.min_vals for x in args], **kwargs)
    max_vals = np.mean([x.max_vals for x in args], **kwargs)

    args, kwargs = inputs2child(*args, **kwargs)

    data = np.mean(args, **kwargs)

    return SingleEntityPhiTensor(
        child=data,
        entity=entity,
        min_vals=min_vals,
        max_vals=max_vals,
        scalar_manager=scalar_manager,
    )


@implements(SingleEntityPhiTensor, np.expand_dims)
def expand_dims(a, axis):

    entity = a.entity

    min_vals = np.expand_dims(a=a.min_vals, axis=axis)
    max_vals = np.expand_dims(a=a.max_vals, axis=axis)

    data = np.expand_dims(a.child, axis=axis)

    return SingleEntityPhiTensor(
        child=data,
        entity=entity,
        min_vals=min_vals,
        max_vals=max_vals,
        scalar_manager=a.scalar_manager,
    )
