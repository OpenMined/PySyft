from __future__ import annotations
import jax
import flax
import numpy as np
from jax import numpy as jnp
from typing import Union
from typing import Callable
from typing import Set
from random import randint


def no_op(x):
    return x


@flax.struct.dataclass
class GammaTensor:
    value: jnp.array
    min_val: float = flax.struct.field(pytree_node=False)
    max_val: float = flax.struct.field(pytree_node=False)
    func: Callable = flax.struct.field(pytree_node=False, default_factory=lambda: no_op)
    id: str = flax.struct.field(pytree_node=False, default_factory=lambda: str(randint(0, 2 ** 32 - 1)))
    state: dict = flax.struct.field(pytree_node=False, default_factory=dict)

    def __post_init__(self):
        if len(self.state) == 0:
            self.state[self.id] = self

    def run(self, state):
        # we hit a private input
        if self.func is no_op:
            return self.func(state[self.id].value)
        return self.func(state)

    def __add__(self, other):
        state = dict()
        state.update(self.state)

        if isinstance(other, GammaTensor):
            adder = lambda state: jnp.add(self.run(state), other.run(state))
            state.update(other.state)
            value = self.value + other.value
        else:
            adder = lambda state: jnp.add(self.run(state), other)
            value = self.value + other

        return GammaTensor(value=value, func=adder, state=state)

    def sum(self):
        sum = lambda state: jnp.sum(self.run(state))
        state = dict()
        state.update(self.state)

        value = jnp.sum(self.value)
        min_val = jnp.sum(self.min_val)
        max_val = jnp.sum(self.max_val)

        return GammaTensor(value=value, min_val=min_val, max_val=max_val, func=sum, state=state)

    def expand_dims(self, axis):
        expand_dims = lambda state: jnp.expand_dims(self.run(state), axis)

        state = dict()
        state.update(self.state)

        return GammaTensor(value=jnp.expand_dims(self.value, axis), min_val=self.min_val, max_val=self.max_val,
                           func=expand_dims, state=state)

    def squeeze(self, axis=None):
        squeeze = lambda state: jnp.squeeze(self.run(state), axis)
        state = dict()
        state.update(self.state)
        return GammaTensor(value=jnp.squeeze(self.value, axis), min_val=self.min_val, max_val=self.max_val,
                           func=squeeze, state=state)

    def __len__(self):
        return len(self.value)


@flax.struct.dataclass
class IntermediateGammaScalar:
    value: float
    min_val: float = flax.struct.field(pytree_node=False)
    max_val: float = flax.struct.field(pytree_node=False)
    entities: jnp.array = flax.struct.field(pytree_node=False)
    is_linear: bool = flax.struct.field(pytree_node=False)
    generative_func: Callable = flax.struct.field(pytree_node=False)

    def __add__(self, other: Union["GammaScalar", "IntermediateGammaScalar"]) -> "IntermediateGammaScalar":
        value = self.value + other.value
        min_val = self.min_val + other.min_val
        max_val = self.max_val + other.max_val

        if isinstance(other, GammaScalar):
            new_entities = jnp.append(self.entities, other.entity)
        else:
            new_entities = jnp.concatenate([self.entities, other.entities])

        return IntermediateGammaScalar(value=value, min_val=min_val, max_val=max_val, entities=new_entities,
                                       is_linear=True, generative_func=None)


@flax.struct.dataclass
class GammaScalar:
    value: float
    min_val: float = flax.struct.field(pytree_node=False)
    max_val: float = flax.struct.field(pytree_node=False)
    entity: int = flax.struct.field(pytree_node=False)
    is_linear: bool = flax.struct.field(pytree_node=False)

    def __add__(self, other: "GammaScalar") -> Union["GammaScalar", "IntermediateGammaScalar"]:
        new_min = self.min_val + other.min_val
        new_max = self.max_val + other.max_val
        new_value = self.value + other.value
        new_entities = jnp.array((self.entity, other.entity))
        return IntermediateGammaScalar(value=new_value, min_val=new_min, max_val=new_max, entities=new_entities,
                                       is_linear=True, generative_func=None)


