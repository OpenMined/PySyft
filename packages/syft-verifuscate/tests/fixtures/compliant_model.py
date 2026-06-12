"""A tiny verifuscate-compliant Flax-style module (green-path test fixture).

Uses only: allow-listed jax/flax calls by name, self.* calls, operator bundles
(arithmetic/indexing/comparison/metadata), comprehensions, and bare-name calls — and NO named methods
on opaque values, no .T, no .append, no dynamic features. So it passes `verify` cleanly.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

CONFIG = dict(dim=8, layers=2, eps=1e-6)


def scale_pattern(n):
    base = (1.0,) * n
    return base[:n]


class RMSNorm(nn.Module):
    def setup(self):
        self.weight = self.param("w", nn.initializers.ones, (CONFIG["dim"],))

    def __call__(self, x):
        sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(sq + CONFIG["eps"]) * (1.0 + self.weight)


class Block(nn.Module):
    cfg: dict

    def setup(self):
        self.norm = RMSNorm()
        self.layers = [RMSNorm() for _ in range(self.cfg["layers"])]

    def __call__(self, x):
        h = self.norm(x)
        out = x + h
        if x.shape[-1] > 0:
            out = out * 0.5
        return jnp.sum(out, axis=-1) + out[..., 0]
