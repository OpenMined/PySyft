from . import model, layers
from syft.frameworks.keras.hook import KerasHook


def get_hooks(torch=None, keras=None):
    # TODO[jason]: this is currently very ugly, in preparation for merging keras with existing workers
    #              for now, we ignore this in favor of TFWorker; sync on this with the team
    from syft.frameworks.torch import TorchHook
    if torch is None:
        import torch
    if keras is None:
        from tensorflow import keras
    k_hook = KerasHook(keras)
    t_hook = TorchHook(torch)
    return k_hook, t_hook

__all__ = [
    'get_hooks'
    'KerasHook',
]
