# third party
from jax.config import config

# this ensures that jax_enable_x64 is set before we import and use it
config.update("jax_enable_x64", True)
