# stdlib
from functools import partial
from typing import Any
from typing import List
from typing import Union

# third party
import numpy as np
import torch

# syft absolute
from syft.decorators import syft_decorator
from syft.lib.tensor.tensorbase_util import call_func_and_wrap_result

Num = Union[int, float]


class ChildDelegatorTensor:
    wrap_type: Any = None

    def __getattr__(self, name: str) -> Any:
        func_or_attr = getattr(self.child, name)
        if callable(func_or_attr):
            return partial(call_func_and_wrap_result, func_or_attr, self.wrap_type)
        else:
            var = func_or_attr
            return var


class DataTensor(ChildDelegatorTensor):
    @syft_decorator(typechecking=True)
    def __init__(self, child: Union[torch.FloatTensor, torch.IntTensor]) -> None:
        self.child = child
        self.wrap_type = DataTensor

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: "DataTensor") -> "DataTensor":
        return DataTensor(child=self.child + other.child)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: "DataTensor") -> "DataTensor":
        return DataTensor(child=self.child - other.child)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: "DataTensor") -> "DataTensor":
        return DataTensor(child=self.child * other.child)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Union[int, float]) -> "DataTensor":
        return DataTensor(child=self.child / other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __matmul__(self, other: "DataTensor") -> "DataTensor":
        return DataTensor(child=self.child @ other.child)


class FloatTensor(ChildDelegatorTensor):
    @syft_decorator(typechecking=True)
    def __init__(self, child: DataTensor) -> None:
        self.child = child
        self.wrap_type = FloatTensor

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: "FloatTensor") -> "FloatTensor":
        return FloatTensor(child=self.child + other.child)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: "FloatTensor") -> "FloatTensor":
        return FloatTensor(child=self.child - other.child)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: "FloatTensor") -> "FloatTensor":
        return FloatTensor(child=self.child * other.child)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Union[int, float]) -> "FloatTensor":
        return FloatTensor(child=self.child / other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __matmul__(self, other: "FloatTensor") -> "FloatTensor":
        return FloatTensor(child=self.child @ other.child)


class IntegerTensor(ChildDelegatorTensor):
    @syft_decorator(typechecking=True)
    def __init__(self, child: DataTensor) -> None:
        self.child = child
        self.wrap_type = IntegerTensor

    # todo: make sure that operations return the correct tensor type
    # e.g. a div between a IntegerTensor and an Int may produce
    # a FloatTensor

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: "IntegerTensor") -> "IntegerTensor":
        return IntegerTensor(child=self.child + other.child)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: "IntegerTensor") -> "IntegerTensor":
        return IntegerTensor(child=self.child - other.child)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: "IntegerTensor") -> "IntegerTensor":
        return IntegerTensor(child=self.child * other.child)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Union[int, float]) -> "IntegerTensor":
        return IntegerTensor(child=self.child / other)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __matmul__(self, other: "IntegerTensor") -> "IntegerTensor":
        return IntegerTensor(child=self.child @ other.child)


class SyftTensor(ChildDelegatorTensor):
    def __init__(self, child: Union[FloatTensor, IntegerTensor]) -> None:
        self.child = child
        self.wrap_type = SyftTensor

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: "SyftTensor") -> "SyftTensor":
        if isinstance(self.child, FloatTensor) and isinstance(other.child, FloatTensor):
            return SyftTensor(child=self.child + other.child)
        elif isinstance(self.child, IntegerTensor) and isinstance(
            other.child, IntegerTensor
        ):
            return SyftTensor(child=self.child + other.child)
        else:
            raise ValueError()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: "SyftTensor") -> "SyftTensor":
        if isinstance(self.child, FloatTensor) and isinstance(other.child, FloatTensor):
            return SyftTensor(child=self.child - other.child)
        elif isinstance(self.child, IntegerTensor) and isinstance(
            other.child, IntegerTensor
        ):
            return SyftTensor(child=self.child - other.child)
        else:
            raise ValueError()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __mul__(self, other: "SyftTensor") -> "SyftTensor":
        if isinstance(self.child, FloatTensor) and isinstance(other.child, FloatTensor):
            return SyftTensor(child=self.child * other.child)
        elif isinstance(self.child, IntegerTensor) and isinstance(
            other.child, IntegerTensor
        ):
            return SyftTensor(child=self.child * other.child)
        else:
            raise ValueError()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __truediv__(self, other: Union[int, float]) -> "SyftTensor":
        if isinstance(self.child, FloatTensor) and isinstance(other, int):
            return SyftTensor(child=self.child / other)
        if isinstance(self.child, IntegerTensor) and isinstance(other, float):
            return SyftTensor(child=self.child / other)
        else:
            raise ValueError()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __matmul__(self, other: "SyftTensor") -> "SyftTensor":
        if isinstance(self.child, FloatTensor) and isinstance(other, SyftTensor):
            return SyftTensor(child=self.child @ other.child)
        if isinstance(self.child, IntegerTensor) and isinstance(other, SyftTensor):
            return SyftTensor(child=self.child @ other.child)
        else:
            raise ValueError()

    @classmethod
    def FloatTensor(cls, data: Union[List[Num], np.ndarray]) -> "SyftTensor":
        if isinstance(data, list) or isinstance(data, np.ndarray):
            return cls(
                child=FloatTensor(child=DataTensor(child=torch.FloatTensor(data)))
            )
        else:
            raise NotImplementedError()
