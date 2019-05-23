import tensorflow as tf


class Sequential(object):
    def __init__(self, layers=None):

        self._layers = []
        self.weights = None
        
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)

    def set_weights(self, weights):
        self.weights = weights

    def share(self, prot=None):
        
        tfe_layers = []

        # Once tfe.keras is ready will have to specify the protocol
        # when instantiating the model
        # with prot:
        for layer in self._layers:
            tfe_layer_cls = getattr(tf.keras.layers, layer.name)
            tfe_layers.append(tfe_layer_cls(*layer.args, **layer.kargs))

        tfe_model = tf.keras.Sequential(tfe_layers)

        if not self.weights:
            raise ValueError("Please make sure to set the weights before sharing" 
                             "the model using the method `set_weights`")
        else:
            tfe_model.set_weights(self.weights)

        return tfe_model
        



