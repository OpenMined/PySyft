import tensorflow as tf


class Sequential(object):
    def __init__(self, layers=None):

        self._layers = []
        
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)

    def share(self, prot=None):
        
        tfe_layers = []

        #Once tfe.keras is ready will have to specify the protocol
        # when instanciating the model
        # with prot:
        for layer in self._layers:
            tfe_layer_cls = getattr(tf.keras.layers, layer.name)
            tfe_layers.append(tfe_layer_cls(**layer.arguments))

        tfe_model = tf.keras.Sequential(tfe_layers)

        return tfe_model
        



