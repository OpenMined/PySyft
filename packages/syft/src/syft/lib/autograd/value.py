# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from functools import lru_cache
from numbers import Number
from typing import Any
from typing import Dict as TypeDict
from typing import Tuple as TypeTuple
from typing import Union
import uuid

# third party
import numpy as np
import sympy as sym
from sympy.core.basic import Basic

# May 5th, 2020
# APACHE 2.0 License
# Some code modified/reused from https://github.com/sradc/SmallPebble under APACHE 2.0 License terms

__author__ = ["Georgios Kaissis", "Alexander Ziller"]



Numeric = Any  # numbers.Number doesn't work as desired


class Value:
    def __init__(
        self,
        value: Union[Basic, Numeric],
        grads: TypeTuple[TypeTuple[Value, Numeric], ...] = (),
    ) -> None:
        self.value = value
        self.grads = grads
        self.id = uuid.uuid4().hex

    def __add__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return add(self, other)

    def __radd__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return add(self, other)

    def __mul__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return mul(self, other)

    def __rmul__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return mul(self, other)

    def __sub__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return add(self, neg(other))  # subtraction is not commutative

    def __rsub__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return add(other, neg(self))  # subtraction is not commutative

    def __truediv__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return mul(self, inv(other))  # division is not commutative

    def __rtruediv__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return mul(other, inv(self))  # division is not commutative

    def __pow__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return power(self, other)

    def __rpow__(self, other: Union[Basic, Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return power(other, self)

    def __neg__(self) -> Value:
        return neg(self)

    def exp(self) -> Value:
        return _exp(self)

    def __repr__(self) -> str:
        return str(self.value)


def add(a: Value, b: Value) -> Value:
    value = a.value + b.value
    grads = ((a, 1), (b, 1))
    return Value(value, grads)


def mul(a: Value, b: Value) -> Value:
    value = a.value * b.value
    grads = ((a, b.value), (b, a.value))
    return Value(value, grads)


def neg(a: Value) -> Value:
    value = -1 * a.value
    grads = ((a, -1),)
    return Value(value, grads)


def power(a: Value, b: Value) -> Value:
    value = a.value ** b.value
    grads = (
        (a, b.value * (a.value ** (b.value - 1))),
        (
            b,
            (np.log(a.value) if isinstance(a.value, Number) else sym.log(a.value))
            * (a.value ** b.value),
        ),
    )
    return Value(value, grads)


def inv(a: Value) -> Value:
    value = 1.0 / a.value
    grads = ((a, -1 / a.value ** 2),)
    return Value(value, grads)


def _exp(a: Value) -> Value:
    value = np.exp(a.value) if isinstance(a.value, Number) else sym.exp(a.value)
    grads = ((a, value),)
    return Value(value, grads)


def log(a: Value) -> Value:
    value = np.exp(a.value) if isinstance(a.value, Number) else sym.log(a.value)
    grads = ((a, 1.0 / a.value),)
    return Value(value, grads)


def sigmoid(x: Numeric) -> np.float64:
    return 1 / (1 + np.exp(-x))


@lru_cache(maxsize=None)
def grad(value: Value, accumulate: bool = False) -> TypeDict[Value, float]:
    gradients: TypeDict[Value, float] = defaultdict(lambda: 0)

    def _inner(value: Value, weight: float) -> None:

        for parent, grad in value.grads:
            to_par = weight * grad
            gradients[parent] += to_par
            _inner(parent, to_par)

    _inner(value, weight=1)

    if accumulate:
        for parent in gradients.keys():
            _grad = getattr(parent, "_grad", 0)
            if _grad is not None:
                setattr(parent, "_grad", _grad + gradients[parent])
            else:
                setattr(parent, "_grad", gradients[parent])
    else:
        for parent in gradients.keys():
            setattr(parent, "_grad", gradients[parent])

    return dict(gradients)


to_values = np.vectorize(lambda x: Value(x))
