import tf_encrypted as tfe

from syft.frameworks.keras.model import serve, share, shutdown
from syft.frameworks.keras.layers import add_constructor_registration, filter_nonlayers



class KerasHook:
    def __init__(self, keras):
        self.keras = keras
        self._hook_layers()
        self._hook_sequential()

    def _hook_layers(self):
        for layer_cls in filter_nonlayers(self.keras.layers, tfe.keras.layers):
            registered_cls = add_constructor_registration(layer_cls)

    def _hook_sequential(self):
        seq_cls = getattr(self.keras, 'Sequential')
        setattr(seq_cls, 'share', share)
        setattr(seq_cls, 'serve', serve)
        setattr(seq_cls, 'shutdown', shutdown)
