# future
from __future__ import annotations

# stdlib
from collections.abc import Sequence
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
import numpy.typing as npt

# relative
from ...adp.vm_private_scalar_manager import (
    VirtualMachinePrivateScalarManager as TypeScalarManager,
)
from ...common.serde.serializable import serializable
from ..broadcastable import is_broadcastable
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import implements  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from ..types import AcceptableSimpleType
from .adp_tensor import ADPTensor
from .initial_gamma import InitialGammaTensor
from .single_entity_phi import SingleEntityPhiTensor


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

    # a list of attributes needed for serialization using RecursiveSerde
    __attr_allowlist__ = ["child"]

    def __init__(self, rows: Sequence, check_shape: bool = True):
        """Initialize a RowEntityPhiTensor

        rows: the actual data organized as an iterable (can be any type of iterable)
        check_shape: whether or not we are already confident that the objects in iterable
            'rows' all have the same dimension (check if we're not sure).

        """
        # Container type heirachy: https://docs.python.org/3/library/collections.abc.html
        self.child: Sequence
        super().__init__(rows)

        # include this check because it's expensvie to check and sometimes we can skip it when
        # we already know the rows are identically shaped.
        if check_shape:

            # shape of the first row we use for reference
            shape = rows[0].shape

            # check each row to make sure it's the same shape as the first
            for row in rows[1:]:
                if shape != row.shape:
                    raise Exception(
                        f"All rows in RowEntityPhiTensor must match: {shape} != {row.shape}"
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
        return np.array(
            [[x.entity] * np.array(x.shape).prod() for x in self.child]
        ).reshape(self.shape)

    @property
    def gamma(self) -> InitialGammaTensor:
        return self.create_gamma()

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

    def __eq__(self, other: Any) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] == other)
                else:
                    new_list.append(self.child[i] == other.child[i])
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __eq__: {len(self.child)} != {len(other.child)}"
            )

    def __ne__(self, other: Any) -> RowEntityPhiTensor:
        opposite_result = self.__eq__(other)

        # Normal inversion on (opposite_result.child) might not work on nested lists
        result = []
        for row in opposite_result.child:
            result.append(np.invert(row))

        return RowEntityPhiTensor(rows=result)

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
                new_list = [
                    self.child[i] * other.child[i] for i in range(len(self.child))
                ]
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
    def sum(
        self, *args: Any, axis: Optional[int] = None, **kwargs: Any
    ) -> RowEntityPhiTensor:

        if axis is None or axis == 0:
            return self.gamma.sum(axis=axis)

        new_list = list()
        for row in self.child:
            new_list.append(row.sum(*args, axis=axis - 1, **kwargs))

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

        # if the tensor being compared is a public tensor / int / float / etc.
        if is_acceptable_simple_type(other):
            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] <= other)

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        if len(self.child) == len(other.child):
            # tensors have different entities
            if not (self.entities == other.entities).all():
                raise Exception("Tensor owners do not match")

            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] <= other.child[i])

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                f"Tensor dims do not match for __le__: {len(self.child)} != {len(other.child)}"
            )

    def __lt__(self, other: Any) -> RowEntityPhiTensor:

        # if the tensor being compared is a public tensor / int / float / etc.
        if is_acceptable_simple_type(other):
            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] < other)

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        if len(self.child) == len(other.child):
            # tensors have different entities
            if not (self.entities == other.entities).all():
                raise Exception("Tensor owners do not match")

            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] < other.child[i])

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                f"Tensor dims do not match for __lt__: {len(self.child)} != {len(other.child)}"
            )

    def __gt__(self, other: Any) -> RowEntityPhiTensor:

        # if the tensor being compared is a public tensor / int / float / etc.
        if is_acceptable_simple_type(other):
            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] > other)

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        if len(self.child) == len(other.child):
            # tensors have different entities
            if not (self.entities == other.entities).all():
                raise Exception("Tensor owners do not match")

            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] > other.child[i])

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                f"Tensor dims do not match for __gt__: {len(self.child)} != {len(other.child)}"
            )

    def __ge__(self, other: Any) -> RowEntityPhiTensor:

        # if the tensor being compared is a public tensor / int / float / etc.
        if is_acceptable_simple_type(other):
            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] >= other)

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        if len(self.child) == len(other.child):
            # tensors have different entities
            if not (self.entities == other.entities).all():
                raise Exception("Tensor owners do not match")

            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] >= other.child[i])

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                f"Tensor dims do not match for __ge__: {len(self.child)} != {len(other.child)}"
            )

    def clip(
        self, a_min: npt.ArrayLike, a_max: npt.ArrayLike, *args: Any
    ) -> RowEntityPhiTensor:

        if a_min is None and a_max is None:
            raise Exception("ValueError: clip: must set either max or min")

        new_list = list()
        for row in self.child:
            new_list.append(row.clip(a_min=a_min, a_max=a_max, *args))

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
