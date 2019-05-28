from syft.keras import model, layers
from syft.keras.hook import KerasHook


def get_hooks(torch=None, keras=None):
    # TODO[jason]: this is currently very ugly, sync on this with Andrew & Theo
    from syft.frameworks.torch import TorchHook
    if torch is None:
        import torch
    if keras is None:
        from tensorflow import keras
    k_hook = KerasHook(keras)
    t_hook = TorchHook(torch)
    return k_hook, t_hook

__all__ = [
    'model',
    'layers',
]
