# May 5th, 2020
# APACHE 2.0 License
# Some code modified/reused from https://github.com/sradc/SmallPebble under APACHE 2.0 License terms

__author__ = ["Georgios Kaissis", "Alexander Ziller"]

# stdlib
from collections import defaultdict
from functools import lru_cache
from numbers import Number
import uuid

# third party
import numpy as np
import sympy as sym


class Value:
    def __init__(self, value, grads=()):
        self.value = value
        self.grads = grads
        self.id = uuid.uuid4().hex

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return add(self, other)

    def __radd__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return add(other, self)

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return mul(self, other)

    def __rmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return mul(other, self)

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return add(self, neg(other))

    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return add(other, neg(self))

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return mul(self, inv(other))

    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return mul(other, inv(self))

    def __pow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return power(self, other)

    def __rpow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return power(other, self)

    def __neg__(self):
        return neg(self)

    def exp(self):
        return _exp(self)

    def __repr__(self):
        return str(self.value)


def add(a, b):
    value = a.value + b.value
    grads = ((a, 1), (b, 1))
    return Value(value, grads)


def mul(a, b):
    value = a.value * b.value
    grads = ((a, b.value), (b, a.value))
    return Value(value, grads)


def neg(a):
    value = -1 * a.value
    grads = ((a, -1),)
    return Value(value, grads)


def power(a, b):
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


def inv(a):
    value = 1.0 / a.value
    grads = ((a, -1 / a.value ** 2),)
    return Value(value, grads)


def _exp(a):
    value = np.exp(a.value) if isinstance(a.value, Number) else sym.exp(a.value)
    grads = ((a, value),)
    return Value(value, grads)


def log(a):
    value = np.exp(a.value) if isinstance(a.value, Number) else sym.log(a.value)
    grads = ((a, 1.0 / a.value),)
    return Value(value, grads)


@lru_cache(maxsize=None)
def grad(Value, accumulate=False):
    gradients = defaultdict(lambda: 0)

    def _inner(Value, weight):

        for parent, grad in Value.grads:
            to_par = weight * grad
            gradients[parent] += to_par
            _inner(parent, to_par)

    _inner(Value, weight=1)

    if accumulate:
        for parent in gradients.keys():
            if not hasattr(parent, '_grad'):
                parent._grad = 0
            parent._grad += gradients[parent]
    else:
        for parent in gradients.keys():
            parent._grad = gradients[parent]

    return dict(gradients)


to_values = np.vectorize(lambda x: Value(x))


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))
