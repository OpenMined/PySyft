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
# result = f_jit(x)
# print("No wrapper", result)
# device = action_object.device()
# x_device = x._device
# action_device = action_object._device
# print("Device", device)
# with jax.default_device as device:
#   print(device)
# print(jax.devices()[0])
# print(action_object.__jax_array__().device())
# print(int(x))
# print(int(action_object))
# print(int(action_object.__jax_array__().device()))
device = action_object.device()
print(device)
print(type(device))
# print("TYPE:", type(action_object.__jax_array__().device()))
print("Wrapper", f_jit(action_object))
# print("No wrapper jnp", jnp.add(x,x))
# print("wrapper jnp",jnp.add(action_object,action_object))
