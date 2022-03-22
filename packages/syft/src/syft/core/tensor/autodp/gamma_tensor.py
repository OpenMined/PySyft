# future
from __future__ import annotations

# stdlib
from collections import deque
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple

if TYPE_CHECKING:
    # stdlib
    from dataclasses import dataclass
else:
    from flax.struct import dataclass

# third party
import flax
import jax
from jax import numpy as jnp
import numpy as np
from numpy.random import randint
from scipy.optimize import shgo

# relative
from ...adp.data_subject_ledger import DataSubjectLedger
from ...adp.entity_list import EntityList
from ...adp.vectorized_publish import vectorized_publish


def create_lookup_tables(dictionary: dict) -> Tuple[List[str], dict, List[dict]]:
    index2key: List = [str(x) for x in dictionary.keys()]
    key2index: dict = {key: i for i, key in enumerate(index2key)}
    # Note this maps to GammaTensor, not to GammaTensor.value as name may imply
    index2values: List = [dictionary[i] for i in index2key]

    return index2key, key2index, index2values


def create_new_lookup_tables(
    dictionary: dict,
) -> Tuple[Deque[str], dict, Deque[dict], Deque[int]]:
    index2key: Deque = deque()
    key2index: dict = {}
    index2values: Deque = (
        deque()
    )  # Note this maps to GammaTensor, not to GammaTensor.value as name may imply
    index2size: Deque = deque()
    for index, key in enumerate(dictionary.keys()):
        key = str(key)
        index2key.append(key)
        key2index[key] = index
        index2values.append(dictionary[key])
        index2size.append(len(dictionary[key]))

    return index2key, key2index, index2values, index2size


def no_op(x: Dict[str, GammaTensor]) -> Dict[str, GammaTensor]:
    """A Private input will be initialized with this function.
    Whenever you manipulate a private input (i.e. add it to another private tensor),
    the result will have a different function. Thus we can check to see if the f
    """
    return x


@dataclass
class GammaTensor:
    value: jnp.array
    data_subjects: EntityList
    min_val: float = flax.struct.field(pytree_node=False)
    max_val: float = flax.struct.field(pytree_node=False)
    is_linear: bool = True
    func: Callable = flax.struct.field(pytree_node=False, default_factory=lambda: no_op)
    id: str = flax.struct.field(
        pytree_node=False, default_factory=lambda: str(randint(0, 2**32 - 1))
    )  # TODO: Need to check if there are any scenarios where this is not secure
    state: dict = flax.struct.field(pytree_node=False, default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.state) == 0:
            self.state[self.id] = self

    def run(self, state: dict) -> Callable:
        # we hit a private input
        if self.func is no_op:
            return self.func(state[self.id].value)
        return self.func(state)

    def __add__(self, other: Any) -> GammaTensor:
        state = dict()
        state.update(self.state)

        if isinstance(other, GammaTensor):

            def _add(state: dict) -> jax.numpy.DeviceArray:
                return jnp.add(self.run(state), other.run(state))

            state.update(other.state)
            value = self.value + other.value
            min_val = self.min_val + other.min_val
            max_val = self.max_val + other.max_val
        else:

            def _add(state: dict) -> jax.numpy.DeviceArray:
                return jnp.add(self.run(state), other)

            value = self.value + other
            min_val = self.min_val + other
            max_val = self.max_val + other

        return GammaTensor(
            value=value,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_add,
            state=state,
        )

    def __mul__(self, other: Any) -> GammaTensor:
        state = dict()
        state.update(self.state)

        if isinstance(other, GammaTensor):

            def _mul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.multiply(self.run(state), other.run(state))

            state.update(other.state)
            value = self.value * other.value
        else:

            def _mul(state: dict) -> jax.numpy.DeviceArray:
                return jnp.multiply(self.run(state), other)

            value = self.value * other

        return GammaTensor(
            value=value,
            data_subjects=self.data_subjects,
            min_val=0,
            max_val=10,
            func=_mul,
            state=state,
        )

    def sum(self, *args: Tuple[Any, ...], **kwargs: Any) -> GammaTensor:
        def _sum(state: dict) -> jax.numpy.DeviceArray:
            return jnp.sum(self.run(state))

        state = dict()
        state.update(self.state)

        value = jnp.sum(self.value)
        min_val = jnp.sum(self.min_val)
        max_val = jnp.sum(self.max_val)

        return GammaTensor(
            value=value,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_sum,
            state=state,
        )

    def sqrt(self) -> GammaTensor:
        def _sqrt(state: dict) -> jax.numpy.DeviceArray:
            return jnp.sqrt(self.run(state))

        state = dict()
        state.update(self.state)

        value = jnp.sqrt(self.value)
        min_val = jnp.sqrt(self.min_val)
        max_val = jnp.sqrt(self.max_val)

        return GammaTensor(
            value=value,
            data_subjects=self.data_subjects,
            min_val=min_val,
            max_val=max_val,
            func=_sqrt,
            state=state,
        )

    def publish(
        self,
        ledger: DataSubjectLedger,
        sigma: Optional[float] = None,
        output_func: Callable = np.sum,
    ) -> jax.numpy.DeviceArray:
        # TODO: Add data scientist privacy budget as an input argument, and pass it
        # into vectorized_publish
        if sigma is None:
            sigma = self.value.mean() / 4

        return vectorized_publish(
            min_vals=self.min_val,
            max_vals=self.max_val,
            values=self.state["0"],
            data_subjects=self.data_subjects,
            is_linear=self.is_linear,
            sigma=sigma,
            output_func=output_func,
            ledger=ledger,
        )

    def expand_dims(self, axis: int) -> GammaTensor:
        def _expand_dims(state: dict) -> jax.numpy.DeviceArray:
            return jnp.expand_dims(self.run(state), axis)

        state = dict()
        state.update(self.state)

        return GammaTensor(
            value=jnp.expand_dims(self.value, axis),
            data_subjects=self.data_subjects,
            min_val=self.min_val,
            max_val=self.max_val,
            func=_expand_dims,
            state=state,
        )

    def squeeze(self, axis: Optional[int] = None) -> GammaTensor:
        def _squeeze(state: dict) -> jax.numpy.DeviceArray:
            return jnp.squeeze(self.run(state), axis)

        state = dict()
        state.update(self.state)
        return GammaTensor(
            value=jnp.squeeze(self.value, axis),
            data_subjects=self.data_subjects,
            min_val=self.min_val,
            max_val=self.max_val,
            func=_squeeze,
            state=state,
        )

    def __len__(self) -> int:
        return len(self.value)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape

    @property
    def lipschitz_bound(self) -> float:
        # TODO: Check if there are any functions for which lipschitz bounds shouldn't be computed
        # if dis(self.func) == dis(no_op):
        #     raise Exception

        print("Starting JAX JIT")
        fn = jax.jit(self.func)
        print("Traced self.func with jax's jit, now calculating gradient")
        grad_fn = jax.grad(fn)
        print("Obtained gradient, creating lookup tables")
        i2k, k2i, i2v, i2s = create_new_lookup_tables(self.state)

        print("created lookup tables, now getting bounds")
        i2minval = jnp.concatenate([x for x in i2v]).reshape(-1, 1)
        i2maxval = jnp.concatenate([x for x in i2v]).reshape(-1, 1)
        bounds = jnp.concatenate([i2minval, i2maxval], axis=1)
        print("Obtained bounds")
        # sample_input = i2minval.reshape(-1)
        _ = i2minval.reshape(-1)
        print("Obtained all inputs")

        def max_grad_fn(input_values: np.ndarray) -> float:
            vectors = {}
            n = 0
            for i, size_param in enumerate(i2s):
                vectors[i2k[i]] = input_values[n : n + size_param]  # noqa: E203
                n += size_param

            grad_pred = grad_fn(vectors)

            m = 0
            for value in grad_pred.values():
                m = max(m, jnp.max(value))

            # return negative because we want to maximize instead of minimize
            return -m

        print("starting SHGO")
        res = shgo(max_grad_fn, bounds, iters=1, constraints=tuple())
        print("Ran SHGO")
        # return negative because we flipped earlier
        return -float(res.fun)

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype
