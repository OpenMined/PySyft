import syft as sy
sy.requires(">=0.8-beta")

import jax
import jax.numpy as jnp
from syft.core.node.new.jax import DeviceArrayObject 
# from syft import jax

x = jnp.arange(10)
y = jnp.ones(10)

from syft.core.node.new.action_object import ActionObject

action_object = ActionObject.from_obj(x) 

def f(x):
  if len(x) < 4:
    return x
  else:
    return 2 * x

f_jit = jax.jit(f)
# print("No wrapper", f_jit(x))
print("Wrapper", f_jit(action_object))
print("No wrapper jnp", jnp.add(x,x))
print("wrapper jnp",jnp.add(action_object,action_object))
