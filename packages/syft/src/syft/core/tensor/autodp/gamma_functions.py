import jax
from jax import numpy as jnp
from . import gamma_tensor as GT 

mapper = dict()
mapper["no_op"] = lambda x: x

def _add(state: dict) -> jax.numpy.DeviceArray:
            return jnp.add(
                *[
                    i.reconstruct() if isinstance(i, GT.GammaTensor) else i
                    for i in state.values()
                ]
            )
mapper["add"] = _add

def _sub(state: dict) -> jax.numpy.DeviceArray:
            return jnp.subtract(*[i.reconstruct() for i in state.values()])
mapper["sub"] = _sub

def _mul_private(state: dict) -> jax.numpy.DeviceArray:
    return jnp.multiply(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["mul_private"] = _mul_private

def _mul_public(state: dict) -> jax.numpy.DeviceArray:
                return jnp.multiply(
                    *[
                        i.reconstruct() if isinstance(i, GT.GammaTensor) else i
                        for i in state.values()
                    ]
                )
mapper["mul_public"] = _mul_public

def _truediv(state: dict) -> jax.numpy.DeviceArray:
    return jnp.divide(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["truediv"] = _truediv

def _matmul_private(state: dict) -> jax.numpy.DeviceArray:
    return jnp.matmul(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["matmul_private"] = _matmul_private

def _matmul_public(state: dict) -> jax.numpy.DeviceArray:
    return jnp.matmul(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["matmul_public"] = _matmul_public

def _rmatmul(state: dict) -> jax.numpy.DeviceArray:
    return jnp.matmul(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["rmatmul"] = _rmatmul

def _gt(state: dict) -> jax.numpy.DeviceArray:
    return jnp.greater(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["gt"] = _gt

def _ge(state: dict) -> jax.numpy.DeviceArray:
    return jnp.greater_equal(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["ge"] = _ge

def _eq(state: dict) -> jax.numpy.DeviceArray:
    return jnp.equal(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["eq"] = _eq

def _ne(state: dict) -> jax.numpy.DeviceArray:
    return jnp.equal(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["ne"] = _ne


def _lt(state: dict) -> jax.numpy.DeviceArray:
    return jnp.less(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["lt"] = _lt

def _le(state: dict) -> jax.numpy.DeviceArray:
    return jnp.less_equal(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["le"] = _le

def _exp(state: dict) -> jax.numpy.DeviceArray:
    return jnp.exp(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["exp"] = _exp

def _log(state: dict) -> jax.numpy.DeviceArray:
    return jnp.log(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["log"] = _log

def _transpose(state: dict) -> jax.numpy.DeviceArray:
    return jnp.transpose(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["transpose"] = _transpose

def _sum(state: dict) -> jax.numpy.DeviceArray:
    return jnp.sum(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["sum"] = _sum

def _ones_like(state: dict) -> jax.numpy.DeviceArray:
    return jnp.ones_like(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["ones_like"] = _ones_like

def _zeros_like(state: dict) -> jax.numpy.DeviceArray:
    return jnp.zeros_like(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["zeros_like"] = _zeros_like

def _ravel(state: dict) -> jax.numpy.DeviceArray:
    return jnp.ravel(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["ravel"] = _ravel

def _resize(state: dict) -> jax.numpy.DeviceArray:
    return jnp.resize(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["resize"] = _resize


def _compress(state: dict) -> jax.numpy.DeviceArray:
    return jnp.compress(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )

mapper["compress"] = _compress

def _squeeze(state: dict) -> jax.numpy.DeviceArray:
    return jnp.squeeze(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )

mapper["squeeze"] = _squeeze

def _any(state: dict) -> jax.numpy.DeviceArray:
    return jnp.any(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["any"] = _any

def _all(state: dict) -> jax.numpy.DeviceArray:
    return jnp.all(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["all"] = _all

def _and(state: dict) -> jax.numpy.DeviceArray:
    return jnp.__and__(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["and"] = _and


def _or(state: dict) -> jax.numpy.DeviceArray:
    return jnp.__or__(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["or"] = _or

def _pos(state: dict) -> jax.numpy.DeviceArray:
    return jnp.pos(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["pos"] = _pos

def _neg(state: dict) -> jax.numpy.DeviceArray:
    return jnp.neg(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["neg"] = _neg

def _mean(state: dict) -> jax.numpy.DeviceArray:
    # TODO: Figure out if any modifications need to be done if adding axis/args/kwargs to source/state
    return jnp.mean(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )

mapper["mean"] = _mean

def _std(
    state: dict, #axis: Union[int, Tuple[int, ...]] = axis
) -> jax.numpy.DeviceArray:
    return jnp.std(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["std"] = _std

def _dot(state: dict) -> jax.numpy.DeviceArray:
    return jnp.dot(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["dot"] = _dot

def _sqrt(state: dict) -> jax.numpy.DeviceArray:
    return jnp.sqrt(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["sqrt"] = _sqrt

def _abs(state: dict) -> jax.numpy.DeviceArray:
    return jnp.abs(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )
mapper["abs"] = _abs

def _clip(state: dict) -> jax.numpy.DeviceArray:
    return jnp.clip(
        *[
            i.reconstruct() if isinstance(i, GT.GammaTensor) else i
            for i in state.values()
        ]
    )

mapper["clip"] = _clip


