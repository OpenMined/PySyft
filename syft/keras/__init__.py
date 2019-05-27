from syft import TorchHook
from syft.keras import model, layers
from syft.keras.hook import KerasHook


def get_hooks(torch=None):
    if torch is None:
        import torch
    return TorchHook(torch)

__all__ = [
    'model',
    'layers',
]
