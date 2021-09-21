# future
from __future__ import annotations

# stdlib
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
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import implements  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from ..types import AcceptableSimpleType  # type: ignore
from .adp_tensor import ADPTensor
from .initial_gamma import InitialGammaTensor  # type: ignore


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

    def __init__(self, rows: Any, check_shape: bool = True):
        """Initialize a RowEntityPhiTensor

        rows: the actual data organized as an iterable (can be any type of iterable)
        check_shape: whether or not we are already confident that the objects in iterable
            'rows' all have the same dimension (check if we're not sure).

        """

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
        return np.concatenate([x.min_vals for x in self.child]).reshape(self.shape)  # type: ignore

    @property
    def max_vals(self) -> np.ndarray:
        return np.concatenate([x.max_vals for x in self.child]).reshape(self.shape)  # type: ignore

    @property
    def value(self) -> np.ndarray:
        return np.concatenate([x.child for x in self.child]).reshape(self.shape)  # type: ignore

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
        return [len(self.child)] + list(self.child[0].shape)  # type: ignore

    def __eq__(self, other: Any) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):  # type: ignore
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] == other)
                else:
                    new_list.append(self.child[i] == other.child[i])  # type: ignore
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __eq__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def __ne__(self, other: Any) -> RowEntityPhiTensor:
        opposite_result = self.__eq__(other)

        # Normal inversion on (opposite_result.child) might not work on nested lists
        result = []
        for row in opposite_result.child:
            result.append(np.invert(row))

        return RowEntityPhiTensor(rows=result)

    def __add__(  # type: ignore
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:
        # TODO: Catch unacceptable types (str, dict, etc) to avoid errors for other.child below
        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):  # type: ignore
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] + other)
                else:
                    # Private/Public and Private/Private are handled by the underlying SEPT self.child objects.
                    new_list.append(self.child[i] + other.child[i])  # type: ignore
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            # Broadcasting is possible, but we're skipping that for now.
            raise Exception(
                f"Tensor dims do not match for __add__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def __sub__(  # type: ignore
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:
        # TODO: Catch unacceptable types (str, dict, etc) to avoid errors for other.child below
        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):  # type: ignore
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] - other)
                else:
                    new_list.append(self.child[i] - other.child[i])  # type: ignore
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __sub__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def __mul__(  # type: ignore
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):  # type: ignore
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    if isinstance(other, (int, bool, float)):
                        new_list.append(self.child[i] * other)
                    else:
                        new_list.append(self.child[i] * other[i])  # type: ignore
                else:
                    new_list.append(self.child[i] * other.child[i])  # type: ignore
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __mul__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def __pos__(self) -> RowEntityPhiTensor:
        return RowEntityPhiTensor(rows=[+x for x in self.child], check_shape=False)

    def __truediv__(  # type: ignore
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):  # type: ignore
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] / other)
                else:
                    new_list.append(self.child[i] / other.child[i])  # type: ignore
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __truediv__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def repeat(
        self, repeats: Union[int, Tuple[int, ...]], axis: Optional[int] = None
    ) -> RowEntityPhiTensor:

        if axis is None:
            raise Exception(
                "Conservatively, RowEntityPhiTensor doesn't yet support repeat(axis=None)"
            )

        if axis == 0 or axis == -len(self.shape):
            new_list = list()
            for r in range(repeats):  # type: ignore
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

    def reshape(self, *shape: List[int]) -> RowEntityPhiTensor:

        if shape[0] != self.shape[0]:
            raise Exception(
                "For now, you can't reshape the first dimension because that would"
                + "probably require creating a gamma tensor."
            )

        new_list = list()
        for row in self.child:
            new_list.append(row.reshape(shape[1:]))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    # Since this is being used differently compared to supertype, ignoring type annotation errors
    def sum(  # type: ignore
        self, *args: Any, axis: Optional[int] = None, **kwargs: Any
    ) -> RowEntityPhiTensor:

        if axis is None or axis == 0:
            return self.gamma.sum(axis=axis)

        new_list = list()
        for row in self.child:
            new_list.append(row.sum(*args, axis=axis - 1, **kwargs))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    # Since this is being used differently compared to supertype, ignoring type annotation errors
    def transpose(self, *dims: Any) -> RowEntityPhiTensor:  # type: ignore
        if dims[0] != 0:
            raise Exception("Can't move dim 0 in RowEntityPhiTensor at this time")

        new_dims = list(np.array(dims[1:]))

        new_list = list()
        for row in self.child:
            new_list.append(row.transpose(*new_dims))

        return RowEntityPhiTensor(rows=new_list, check_shape=False)

    def __le__(self, other: Any) -> RowEntityPhiTensor:

        # if the tensor being compared is a public tensor / int / float / etc.
        if is_acceptable_simple_type(other):
            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] <= other)

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        if len(self.child) == len(other.child):  # type: ignore
            # tensors have different entities
            if not (self.entities == other.entities).all():
                raise Exception("Tensor owners do not match")

            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] <= other.child[i])  # type: ignore

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                f"Tensor dims do not match for __le__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def __lt__(self, other: Any) -> RowEntityPhiTensor:

        # if the tensor being compared is a public tensor / int / float / etc.
        if is_acceptable_simple_type(other):
            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] < other)

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        if len(self.child) == len(other.child):  # type: ignore
            # tensors have different entities
            if not (self.entities == other.entities).all():
                raise Exception("Tensor owners do not match")

            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] < other.child[i])  # type: ignore

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                f"Tensor dims do not match for __lt__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def __gt__(self, other: Any) -> RowEntityPhiTensor:

        # if the tensor being compared is a public tensor / int / float / etc.
        if is_acceptable_simple_type(other):
            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] > other)

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        if len(self.child) == len(other.child):  # type: ignore
            # tensors have different entities
            if not (self.entities == other.entities).all():
                raise Exception("Tensor owners do not match")

            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] > other.child[i])  # type: ignore

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                f"Tensor dims do not match for __gt__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def __ge__(self, other: Any) -> RowEntityPhiTensor:

        # if the tensor being compared is a public tensor / int / float / etc.
        if is_acceptable_simple_type(other):
            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] >= other)

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        if len(self.child) == len(other.child):  # type: ignore
            # tensors have different entities
            if not (self.entities == other.entities).all():
                raise Exception("Tensor owners do not match")

            new_list = list()
            for i in range(len(self.child)):
                new_list.append(self.child[i] >= other.child[i])  # type: ignore

            return RowEntityPhiTensor(rows=new_list, check_shape=False)

        else:
            raise Exception(
                f"Tensor dims do not match for __ge__: {len(self.child)} != {len(other.child)}"  # type: ignore
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
    for row in a.child:  # type: ignore
        new_rows.append(np.expand_dims(row, axis - 1))  # type: ignore

    return RowEntityPhiTensor(rows=new_rows, check_shape=False)
