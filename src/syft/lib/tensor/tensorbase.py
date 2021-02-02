import torch
from typing import Union, List
from syft.decorators import syft_decorator

Num = Union[int, float]


class DataTensor:
    @syft_decorator(typechecking=True)
    def __init__(self, child: Union[torch.FloatTensor, torch.IntTensor]) -> None:
        self.child = child

    def __add__(self, other: "DataTensor") -> "DataTensor":
        return DataTensor(child=self.child + other.child)


class FloatTensor:
    @syft_decorator(typechecking=True)
    def __init__(self, child: DataTensor) -> None:
        self.child = child

    def __add__(self, other: "FloatTensor") -> "FloatTensor":
        return FloatTensor(child=self.child + other.child)


class IntegerTensor:
    @syft_decorator(typechecking=True)
    def __init__(self, child: DataTensor) -> None:
        self.child = child

    def __add__(self, other: "IntegerTensor") -> "IntegerTensor":
        return IntegerTensor(child=self.child + other.child)


class SyftTensor:
    def __init__(self, child: Union[FloatTensor, IntegerTensor]) -> None:
        self.child = child

    def __add__(self, other: "SyftTensor") -> "SyftTensor":
        if isinstance(self.child, FloatTensor) and isinstance(other.child, FloatTensor):
            return SyftTensor(child=self.child + other.child)
        elif isinstance(self.child, IntegerTensor) and isinstance(
            other.child, IntegerTensor
        ):
            return SyftTensor(child=self.child + other.child)
        else:
            raise ValueError()

    @classmethod
    def FloatTensor(cls, data: List[Num]) -> "SyftTensor":
        if isinstance(data, list):
            return cls(
                child=FloatTensor(child=DataTensor(child=torch.FloatTensor(data)))
            )
        else:
            raise NotImplementedError()
