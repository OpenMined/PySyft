# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

# third party
import numpy as np
import torch

# syft absolute
from syft.lib.tensor.tensorbase_util import call_func_and_wrap_result

Num = Union[int, float]


TENSOR_FORWARD_PROPERTIES = {"shape", "data"}

TENSOR_FORWARD_METHODS = {"squeeze", "unsqueeze", "transpose"}


class ChildDelegatorTensor(type):
    @staticmethod
    def add_dummy_method(method_name: str) -> Callable[..., Callable[..., Any]]:
        def dummy_method(_self: Any, *args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
            method = getattr(_self.child, method_name)
            res = call_func_and_wrap_result(method, _self.__class__, *args, **kwargs)
            return res

        return dummy_method

    @staticmethod
    def add_dummy_property(property_name: str) -> Any:
        def property_getter(_self: Any) -> Any:
            prop = getattr(_self.child, property_name)
            return prop

        def property_setter(_self: Any, new_val: Any) -> None:
            setattr(_self.child, property_name, new_val)

        return property(property_getter, property_setter)

    def __new__(
        cls: Type["ChildDelegatorTensor"],
        name: str,
        bases: Tuple[Any],
        dic: Dict[Any, Any],
    ) -> "ChildDelegatorTensor":
        for method in TENSOR_FORWARD_METHODS:
            if method in dic:
                raise ValueError(f"Attribute {method} already exists in {name}")
            dic[method] = ChildDelegatorTensor.add_dummy_method(method)

        for prop in TENSOR_FORWARD_PROPERTIES:
            if prop in dic:
                raise ValueError(f"Attribute {prop} already exists in {name}")
            dic[prop] = ChildDelegatorTensor.add_dummy_property(prop)

        res = super().__new__(cls, name, bases, dic)
        res = cast(ChildDelegatorTensor, res)
        return res


class DataTensor(metaclass=ChildDelegatorTensor):
    def __init__(self, child: Union[torch.FloatTensor, torch.IntTensor]) -> None:
        self.child = child

    def __add__(self, other: "DataTensor") -> "DataTensor":
        return DataTensor(child=self.child + other.child)

    def __sub__(self, other: "DataTensor") -> "DataTensor":
        return DataTensor(child=self.child - other.child)

    def __mul__(self, other: "DataTensor") -> "DataTensor":
        return DataTensor(child=self.child * other.child)

    def __truediv__(self, other: Union[int, float]) -> "DataTensor":
        return DataTensor(child=self.child / other)

    def __matmul__(self, other: "DataTensor") -> "DataTensor":
        return DataTensor(child=self.child @ other.child)


class FloatTensor(metaclass=ChildDelegatorTensor):
    def __init__(self, child: DataTensor) -> None:
        self.child = child

    def __add__(self, other: "FloatTensor") -> "FloatTensor":
        return FloatTensor(child=self.child + other.child)

    def __sub__(self, other: "FloatTensor") -> "FloatTensor":
        return FloatTensor(child=self.child - other.child)

    def __mul__(self, other: "FloatTensor") -> "FloatTensor":
        return FloatTensor(child=self.child * other.child)

    def __truediv__(self, other: Union[int, float]) -> "FloatTensor":
        return FloatTensor(child=self.child / other)

    def __matmul__(self, other: "FloatTensor") -> "FloatTensor":
        return FloatTensor(child=self.child @ other.child)


class IntegerTensor(metaclass=ChildDelegatorTensor):
    def __init__(self, child: DataTensor) -> None:
        self.child = child

    # todo: make sure that operations return the correct tensor type
    # e.g. a div between a IntegerTensor and an Int may produce
    # a FloatTensor

    def __add__(self, other: "IntegerTensor") -> "IntegerTensor":
        return IntegerTensor(child=self.child + other.child)

    def __sub__(self, other: "IntegerTensor") -> "IntegerTensor":
        return IntegerTensor(child=self.child - other.child)

    def __mul__(self, other: "IntegerTensor") -> "IntegerTensor":
        return IntegerTensor(child=self.child * other.child)

    def __truediv__(self, other: Union[int, float]) -> "IntegerTensor":
        return IntegerTensor(child=self.child / other)

    def __matmul__(self, other: "IntegerTensor") -> "IntegerTensor":
        return IntegerTensor(child=self.child @ other.child)


class SyftTensor(metaclass=ChildDelegatorTensor):
    def __init__(self, child: Union[FloatTensor, IntegerTensor]) -> None:
        self.child = child

    def __add__(self, other: "SyftTensor") -> "SyftTensor":
        if not isinstance(other, SyftTensor):
            raise ValueError("Need to use a SyftTensor")

        if isinstance(self.child, FloatTensor) and isinstance(other.child, FloatTensor):
            return SyftTensor(child=self.child + other.child)
        elif isinstance(self.child, IntegerTensor) and isinstance(
            other.child, IntegerTensor
        ):
            return SyftTensor(child=self.child + other.child)
        else:
            raise ValueError()

    def __sub__(self, other: "SyftTensor") -> "SyftTensor":
        if not isinstance(other, SyftTensor):
            raise ValueError("Need to use a SyftTensor")

        if isinstance(self.child, FloatTensor) and isinstance(other.child, FloatTensor):
            return SyftTensor(child=self.child - other.child)
        elif isinstance(self.child, IntegerTensor) and isinstance(
            other.child, IntegerTensor
        ):
            return SyftTensor(child=self.child - other.child)
        else:
            raise ValueError()

    def __mul__(self, other: "SyftTensor") -> "SyftTensor":
        if not isinstance(other, SyftTensor):
            raise ValueError("Need to use a SyftTensor")

        if isinstance(self.child, FloatTensor) and isinstance(other.child, FloatTensor):
            return SyftTensor(child=self.child * other.child)
        elif isinstance(self.child, IntegerTensor) and isinstance(
            other.child, IntegerTensor
        ):
            return SyftTensor(child=self.child * other.child)
        else:
            raise ValueError()

    def __truediv__(self, other: Union[int, float]) -> "SyftTensor":
        if isinstance(self.child, FloatTensor) and isinstance(other, int):
            return SyftTensor(child=self.child / other)
        if isinstance(self.child, IntegerTensor) and isinstance(other, float):
            return SyftTensor(child=self.child / other)
        else:
            raise ValueError()

    def __matmul__(self, other: "SyftTensor") -> "SyftTensor":
        if not isinstance(other, SyftTensor):
            raise ValueError("Need to use a SyftTensor")

        if isinstance(self.child, FloatTensor) and isinstance(other.child, FloatTensor):
            return SyftTensor(child=self.child @ other.child)
        if isinstance(self.child, IntegerTensor) and isinstance(
            other.child, IntegerTensor
        ):
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

    @classmethod
    def IntegerTensor(cls, data: Union[List[Num], np.ndarray]) -> "SyftTensor":
        if isinstance(data, list) or isinstance(data, np.ndarray):
            return cls(
                child=IntegerTensor(child=DataTensor(child=torch.IntTensor(data)))
            )
        else:
            raise NotImplementedError()
