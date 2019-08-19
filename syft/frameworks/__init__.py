import logging
from syft import dependency_check

logger = logging.getLogger(__name__)

__all__ = list()

if dependency_check.tensorflow_available:
    from syft.frameworks import tensorflow

    __all__.append("tensorflow")

if dependency_check.tfe_available:
    from syft.frameworks import keras

    __all__.append("keras")

if dependency_check.torch_available:
    from syft.frameworks import torch

    __all__.append("torch")
