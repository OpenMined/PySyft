# syft absolute
import syft as sy
from syft.core.node.new.action_types import action_types
from syft.core.node.new.syft_object import SYFT_OBJECT_VERSION_1
sy.requires(">=0.8-beta")

# syft absolute
# NOTE: action objects need to be imported even if they are not used directly.
# Otherwise, they will not be registered in the action_types dict.
from syft.core.node.new.jax import DeviceArrayObject 
from syft.core.node.new.numpy import NumpyArrayObject, NumpyBoolObject, NumpyScalarObject
from syft.core.node.new.action_object import ActionObject

import haiku as hk
import jax
import jax.numpy as jnp


def forward(x):
    mlp = hk.nets.MLP([300, 100, 10])
    return mlp(x)

forward = hk.transform(forward)
rng = hk.PRNGSequence(jax.random.PRNGKey(42))
rng_wrapped = ActionObject.from_obj(rng)

x = jnp.ones([8, 28 * 28])
x_wrapped = ActionObject.from_obj(x)

# print("*** no wrapper ***")
# params = forward.init(next(rng), x)
# logits = forward.apply(params, next(rng), x)

print("*** wrapper ***")
params = forward.init(next(rng), x_wrapped)
# logits = forward.apply(params, next(rng), x_wrapped)