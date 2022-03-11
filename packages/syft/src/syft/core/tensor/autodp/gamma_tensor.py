import jax
import flax
import numpy as np
from jax import numpy as jnp
from typing import Union
from typing import Callable
from typing import Set


@flax.struct.dataclass
class IntermediateGammaScalar:
    value: float
    min_val: float = flax.struct.field(pytree_node=False)
    max_val: float = flax.struct.field(pytree_node=False)
    entities: jnp.array = flax.struct.field(pytree_node=False)
    is_linear: bool = flax.struct.field(pytree_node=False)
    generative_func: Callable = flax.struct.field(pytree_node=False)

    # TODO: fix circular import
    def __add__(self, other: Union[GammaScalar, "IntermediateGammaScalar"]) -> "IntermediateGammaScalar":
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


