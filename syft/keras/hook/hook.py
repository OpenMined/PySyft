from functools import wraps
import inspect
import re

import tf_encrypted as tfe

from syft.workers import BaseWorker


class KerasHook:
    def __init__(self, keras, local_worker: BaseWorker = None):
        self.keras = keras
        self.local_worker = local_worker
        self._hook_layers()
        self._hook_sequential()

    def _hook_layers(self):
        for layer_cls in _filter_nonlayers(self.keras.layers, tfe.keras.layers):
            registered_cls = _add_constructor_registration(layer_cls)

    def _hook_sequential(self):  # TODO
        seq_cls = getattr(self.keras, 'Sequential')
        setattr(seq_cls, 'share', _sequential_share_method)


def _add_constructor_registration(layer_cls):
    layer_cls._native_keras_constructor = layer_cls.__init__

    def _syft_keras_constructor(self, *args, **kwargs):
        self._syft_args_store = args
        self._syft_kwargs_store = kwargs
        self._native_keras_constructor(*args, **kwargs)

    setattr(layer_cls, "__init__", _syft_keras_constructor)

def _sequential_share_method():  # Simulate Yann's function here
    raise NotImplementedError()


def _filter_nonlayers(layers_module, tfe_layers_module):
    # We recognize Layer classes based on their compliance with PEP8.
    pattern = re.compile('[A-Z_][a-zA-Z0-9]+$')
    for attr_name in dir(layers_module):
        match_result = pattern.match(attr_name)
        if match_result is None:
            continue
        else:
            layer_type = match_result.group(0)
            if hasattr(tfe.keras.layers, layer_type):
                yield getattr(layers_module, layer_type)
