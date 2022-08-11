import jax
from jax import numpy as jnp
import numpy as np
from typing import Any
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


def no_op(x: Any) -> Any:  # GammaTensor -> GammaTensor, but avoiding circular imports
    """A Private input will be initialized with this function.
    Whenever you manipulate a private input (i.e. add it to another private tensor),
    the result will have a different function. Thus we can check to see if the f
    """
    from .gamma_tensor import GammaTensor
    res = x
    if isinstance(x, GammaTensor) and isinstance(x.data_subjects, np.ndarray):
        res = GammaTensor(
            child=x.child,
            data_subjects=np.zeros_like(x.data_subjects, np.int64),
            min_vals=x.min_vals,
            max_vals=x.max_vals,
            func=x.func,
            state=GammaTensor.convert_dsl(x.state),
        )
    else:
        raise NotImplementedError
    return res


mapper = dict()
# mapper["add_public"] = _add_public
# mapper["add_private"] = _add_private
mapper["no_op"] = lambda x: x
