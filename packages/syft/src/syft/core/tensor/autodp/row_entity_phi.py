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

# relative
from ....core.adp.vm_private_scalar_manager import (
    VirtualMachinePrivateScalarManager as TypeScalarManager,
)

# syft relative
from ....core.common.serde.recursive import RecursiveSerde
from ...common.serde.serializable import bind_protobuf
from ...tensor.types import AcceptableSimpleType  # type: ignore
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import implements  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore
from .initial_gamma import InitialGammaTensor  # type: ignore


@bind_protobuf
class RowEntityPhiTensor(PassthroughTensor, RecursiveSerde):

    __attr_allowlist__ = ["child"]

    def __init__(self, rows: Any, check_shape: bool = True):
        super().__init__(rows)

        if check_shape:
            shape = rows[0].shape
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
        return [len(self.child)] + list(self.child[0].shape)  # type: ignore

    def __eq__(self, other) -> RowEntityPhiTensor:

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

    def __add__(  # type: ignore
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:

        if is_acceptable_simple_type(other) or len(self.child) == len(other.child):  # type: ignore
            new_list = list()
            for i in range(len(self.child)):
                if is_acceptable_simple_type(other):
                    new_list.append(self.child[i] + other)
                else:
                    new_list.append(self.child[i] + other.child[i])  # type: ignore
            return RowEntityPhiTensor(rows=new_list, check_shape=False)
        else:
            raise Exception(
                f"Tensor dims do not match for __add__: {len(self.child)} != {len(other.child)}"  # type: ignore
            )

    def __sub__(  # type: ignore
        self, other: Union[RowEntityPhiTensor, AcceptableSimpleType]
    ) -> RowEntityPhiTensor:
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
        print(dims)
        if dims[0] != 0:
            raise Exception("Can't move dim 0 in RowEntityPhiTensor at this time")

        new_dims = list(np.array(dims[1:]))

        new_list = list()
        for row in self.child:
            new_list.append(row.transpose(*new_dims))

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
