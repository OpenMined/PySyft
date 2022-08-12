# stdlib
from typing import Any

# third party
import jax
from jax import numpy as jnp
import numpy as np

#
# def _add_private(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.add(self.run(state), other.run(state))
#
# def _add_public(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.add(self.run(state), other)
#
# def _sub(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.subtract(self.run(state), other.run(state))
#
# def _sub(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.subtract(self.run(state), other)
# def _mul(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.multiply(self.run(state), other.run(state))
# def _mul(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.multiply(self.run(state), other)
# def _matmul(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.matmul(self.run(state), other.run(state))
# def _matmul(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.matmul(self.run(state), other)
# def _rmatmul(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.matmul(
# def _rmatmul(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.matmul(other, self.run(state))
# def _gt(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.greater(self.run(state), other.run(state))
# def _gt(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.greater(self.run(state), other)
# def _lt(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.less(self.run(state), other.run(state))
# def _lt(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.greater(self.run(state), other)
# def _le(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.less_equal(self.run(state), other.run(state))
# def _le(state: dict) -> jax.numpy.DeviceArray:
#     return jnp.less_equal(self.run(state), other)


mapper = dict()
# mapper["add_public"] = _add_public
# mapper["add_private"] = _add_private
mapper["no_op"] = lambda x: x
