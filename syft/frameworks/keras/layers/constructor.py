import functools
import inspect
import re

import tf_encrypted as tfe


def add_constructor_registration(layer_cls):
    """
    This method rewires the layer's constructor to record arguments passed to it.
    """
    layer_cls._native_keras_constructor = layer_cls.__init__
    sig = inspect.signature(layer_cls.__init__)

    @functools.wraps(layer_cls.__init__)
    def syft_keras_constructor(self, *args, **kwargs):
        self._constructor_parameters_store = sig.bind(self, *args, **kwargs)
        self._native_keras_constructor(*args, **kwargs)

    setattr(layer_cls, "__init__", syft_keras_constructor)


def filter_layers(layers_module, tfe_layers_module):
    """
    Returns all layer types in module.
    """
    # We recognize Layer classes based on their compliance with PEP8.
    pattern = re.compile("[A-Z_][a-zA-Z0-9]+$")
    for attr_name in dir(layers_module):
        match_result = pattern.match(attr_name)
        if match_result is None:
            continue
        else:
            layer_type = match_result.group(0)
            if hasattr(tfe.keras.layers, layer_type):
                yield getattr(layers_module, layer_type)
