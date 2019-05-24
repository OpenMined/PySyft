import tf_encrypted as tfe
import tensorflow as tf
from keras.models import Sequential
from tf_encrypted.keras.engine.sequential import Sequential as tfe_Sequential
from tf_encrypted.keras.engine.input_layer import InputLayer as tfe_InputLayer



class Sequential(Sequential):
    def __init__(self, *args, **kargs):
        super(Sequential, self).__init__(*args, **kargs)


    def share(self, prot=None):
        
        self.tfe_layers = []

        # NOTE[Yann]: run with keras until tfe is ready
        with prot:
            for keras_layer in self._layers:
                print(keras_layer)
                # Don't need to instantiate 'InputLayer'. Will be automatically
                # created when `mode.predict()`.
                if _get_layer_name(keras_layer) != 'InputLayer':
                    
                    tfe_layer  = _instantiate_tfe_layer(keras_layer)

                    self.tfe_layers.append(tfe_layer)

            # Create tfe.keras model
            tfe_model = tfe_Sequential(self.tfe_layers)

        return tfe_model


def _instantiate_tfe_layer(keras_layer):
    
    # Get dictionary with layer attributes
    keras_layer_attr = keras_layer.__dict__
    
    keras_layer_name = _get_layer_name(keras_layer)
    
    # Identify the TFE layer corresponding to the Keras layer
    tfe_layer_cls = getattr(tfe.keras.layers, keras_layer_name)
    
    # Extract argument list expected by layer __init__
    tfe_arg_list = list(tfe_layer_cls.__dict__['__init__'].__code__.co_varnames)

    tfe_arg_list.remove('self')
    tfe_arg_list.remove('kwargs')
    #tfe_arg_list.remove('dilation_rate')

    # Load weights from Keras layer into TFE layer
    if 'kernel_initializer' in tfe_arg_list:
        k_initializer = tf.keras.initializers.Constant(keras_layer.get_weights()[0])
        keras_layer_attr['kernel_initializer'] = k_initializer
    
    # [NOTE] This might break if use_bias=False
    if 'bias_initializer' in tfe_arg_list:
        b_initializer = tf.keras.initializers.Constant(keras_layer.get_weights()[1])
        keras_layer_attr['bias_initializer'] = b_initializer

    # When the keras layer has been instantiated, the activation attribute 
    # gets assigned an activation function (e.g. keras.activations.relu())
    # Needs to be converted back to activation name (e.g. `relu`), so it
    # uses TFE activation.
    if 'activation' in tfe_arg_list:
        activation_name = keras_layer_attr['activation'].__name__
        keras_layer_attr['activation'] = activation_name
    
    tfe_kwargs = {k: keras_layer_attr[k] for k in tfe_arg_list}
    
    return tfe_layer_cls(**tfe_kwargs)
        

def _get_layer_name(keras_layer_cls):
    return keras_layer_cls.__class__.__name__
