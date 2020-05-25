import logging
from syft import dependency_check

logger = logging.getLogger(__name__)

__all__ = []

if dependency_check.tensorflow_available:
    from syft.frameworks import tensorflow

    __all__.append(tensorflow.__name__)

if dependency_check.tfe_available:
    from syft.frameworks import keras  # noqa: F401

    __all__.append("keras")

if dependency_check.torch_available:
    from syft.frameworks import torch  # noqa: F401

    __all__.append("torch")
