import syft as sy
sy.requires(">=0.8-beta")

import jax
import jax.numpy as jnp
from syft.core.node.new.jax import DeviceArrayObject 
# from syft import jax

x = jnp.arange(10)
# y = jnp.ones(10)

from syft.core.node.new.action_object import ActionObject

action_object = ActionObject.from_obj(x) 
# action_object = DeviceArrayObject(syft_action_data=x)
# print("Type of action object:", type(action_object))

def f(x):
  if len(x) < 4:
    return x
  else:
    return 2 * x

f_jit = jax.jit(f)
device = action_object.device()
print(device)
print(type(device))
print("Wrapper", f_jit(action_object))
