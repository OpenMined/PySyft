# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional
from typing import Tuple as TypeTuple
from typing import Union

# third party
from nacl.signing import VerifyKey
import numpy as np
import numpy.typing as npt

# relative
from ....core.common.serde.recursive import RecursiveSerde
from ...adp.entity import Entity
from ...adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from ...common.serde.serializable import bind_protobuf
from ..ancestors import AutogradTensorAncestor
from ..passthrough import AcceptableSimpleType  # type: ignore
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..passthrough import implements  # type: ignore
from ..passthrough import inputs2child  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from .initial_gamma import InitialGammaTensor


@bind_protobuf
class SingleEntityPhiTensor(PassthroughTensor, AutogradTensorAncestor, RecursiveSerde):

    __attr_allowlist__ = ["child", "_min_vals", "_max_vals", "entity", "scalar_manager"]

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
        return self.gamma.publish(acc=acc, sigma=sigma, user_key=user_key)

    @property
    def min_vals(self) -> np.ndarray:

        return self._min_vals

    @property
    def max_vals(self) -> np.ndarray:

        return self._max_vals

    def __repr__(self) -> str:

        """Pretty print some information, optimized for Jupyter notebook viewing."""
        return (
            f"{self.__class__.__name__}(entity={self.entity.name}, child={self.child})"
        )

    def __eq__(self, other: SupportedChainType) -> SingleEntityPhiTensor:

        if is_acceptable_simple_type(other) or self.child.shape == other.child.shape:  # type: ignore
            # if the tensor being compared is also private
            if isinstance(other, SingleEntityPhiTensor):
                if self.entity != other.entity:
                    # this should return a GammaTensor
                    return NotImplemented
                data = self.child == other.child
            else:
                # this can still fail, if shape1 = (1,s), and shape2 = (,s) --> as an example
                data = self.child == other
            min_vals = self.min_vals * 0.0
            max_vals = self.max_vals * 0.0 + 1.0
            entity = self.entity
            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )
        else:
            raise Exception(
                f"Tensor shapes do not match for __eq__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def logical_and(self, other: SupportedChainType) -> SingleEntityPhiTensor:
        if is_acceptable_simple_type(other) or self.child.shape == other.child.shape:
            if isinstance(other, SingleEntityPhiTensor):
                if self.entity != other.entity:
                    return NotImplemented
                data = self.child and other.child
            else:
                data = self.child and other
            min_vals = self.min_vals * 0.0
            max_vals = self.max_vals * 0.0 + 1.0
            entity = self.entity
            return SingleEntityPhiTensor(
                child=data,
                entity=entity,
                min_vals=min_vals,
                max_vals=max_vals,
                scalar_manager=self.scalar_manager,
            )
        else:
            raise Exception(
                f"Tensor shapes do not match for __eq__: {len(self.child)} != {len(other.child)}"
            )

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
        max_vals_gtlt0 = np.max([self.max_vals, -self.min_vals])

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

    def __add__(self, other: SupportedChainType) -> SingleEntityPhiTensor:

        # if the tensor being added is also private
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

        else:
            return NotImplemented

    def __neg__(self) -> SingleEntityPhiTensor:

        data = self.child * -1
        min_vals = self.min_vals * -1
        max_vals = self.max_vals * -1
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

    def __gt__(self, other: SupportedChainType) -> SingleEntityPhiTensor:

        # if the tensor being added is also private
        if isinstance(other, SingleEntityPhiTensor):

            if self.entity != other.entity:
                # this should return a GammaTensor
                return NotImplemented

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

        # if the tensor being added is a public tensor / int / float / etc.
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

    def __mul__(self, other: SupportedChainType) -> SingleEntityPhiTensor:

        if isinstance(other, SingleEntityPhiTensor):

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

    def __sub__(self, other: SupportedChainType) -> SingleEntityPhiTensor:

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
            # Ignoring unsupported operand error b/c other logic is taken care of
            return self * (1 / other)  # type: ignore

    def dot(self, other: SupportedChainType) -> SingleEntityPhiTensor:
        return self.manual_dot(other)

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

    def sum(self, *args: Any, **kwargs: Any) -> SingleEntityPhiTensor:

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

    def transpose(self, *args: Any, **kwargs: Any) -> SingleEntityPhiTensor:

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

    #
    # def _object2proto(self) -> Tensor_PB:
    #     arrays = []
    #     tensors = []
    #     if isinstance(self.child, np.ndarray):
    #         use_tensors = False
    #         arrays = [
    #             serialize(self.child),
    #             serialize(self.min_vals),
    #             serialize(self.max_vals),
    #         ]
    #     else:
    #         use_tensors = True
    #         tensors = [
    #             serialize(self.child),
    #             serialize(self.min_vals),
    #             serialize(self.max_vals),
    #         ]
    #
    #     return Tensor_PB(
    #         obj_type=full_name_with_name(klass=type(self)),
    #         use_tensors=use_tensors,
    #         arrays=arrays,
    #         tensors=tensors,
    #         entity=serialize(self.entity),
    #     )
    #
    # @staticmethod
    # def _proto2object(proto: Tensor_PB) -> SingleEntityPhiTensor:
    #     use_tensors = proto.use_tensors
    #     children = []
    #     if use_tensors:
    #         children = [deserialize(tensor) for tensor in proto.tensors]
    #     else:
    #         children = [deserialize(array) for array in proto.arrays]
    #
    #     child = children.pop(0)
    #     min_vals = children.pop(0)
    #     max_vals = children.pop(0)
    #
    #     entity = deserialize(blob=proto.entity)
    #
    #     return SingleEntityPhiTensor(
    #         child=child, entity=entity, min_vals=min_vals, max_vals=max_vals
    #     )

    # Cant have recursive and custom Tensor_PB
    # @staticmethod
    # def get_protobuf_schema() -> GeneratedProtocolMessageType:
    #     return Tensor_PB


@implements(SingleEntityPhiTensor, np.expand_dims)
def expand_dims(a: npt.ArrayLike, axis: Optional[int] = None) -> SingleEntityPhiTensor:

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
