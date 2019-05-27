import tf_encrypted as tfe
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tf_encrypted.keras.engine.sequential import Sequential as tfe_Sequential
from tf_encrypted.keras.engine.input_layer import InputLayer as tfe_InputLayer



# When instantiating tfe layers, exclude not supported arguments in TFE.
args_not_supported_by_tfe = ['self', 'kwargs',
                            'activity_regularizer', 'kernel_regularizer',
                            'bias_regularizer', 'kernel_constraint',
                            'bias_constraint','dilation_rate']



class Sequential(Sequential):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__(*args, **kwargs)


    def share(self, prot=None, target_graph=None):

        # Store Keras weights before loading them in the TFE layers.
        stored_keras_weights = dict()

        # TODO(Morten) we could optimize runtime by running a single model.get_weights instead
        for keras_layer in self._layers:
            stored_keras_weights[keras_layer.name] = keras_layer.get_weights()

        if prot is None:
            # If no protocol is specified we use the default
            prot = tfe.get_protocol()

        if target_graph is None:
            # By default we create a new graph for the shared model
            target_graph = tf.Graph()

        with target_graph.as_default():

            with prot:

                tfe_model = tfe_Sequential()

                for keras_layer in self._layers:
                    # Don't need to instantiate 'InputLayer'. Will be automatically
                    # created when `mode.predict()`.
                    print(keras_layer.name, _get_layer_type(keras_layer))
                    if _get_layer_type(keras_layer) == 'InputLayer':
                        batch_input_shape = keras_layer._batch_input_shape
                        supply_shape_next = True
                        if not any(batch_input_shape):
                            raise NotImplementedError("Please provide batch_size or batch_input_shape "
                                                    "keyword arguments when creating your syft.keras "
                                                    "model")
                        continue
                    if supply_shape_next:
                        tfe_layer  = _instantiate_tfe_layer(keras_layer, batch_input_shape, stored_keras_weights)
                        supply_shape_next = False
                    else:
                        tfe_layer  = _instantiate_tfe_layer(keras_layer, None, stored_keras_weights)

                    tfe_model.add(tfe_layer)

        # TODO(Morten) we should keep a reference to the graph instead of returning it
        return target_graph, tfe_model


def _instantiate_tfe_layer(keras_layer, batch_input_shape, stored_keras_weights):

    # Get dictionary with layer attributes
    keras_layer_attr = keras_layer.__dict__
    keras_layer_args = keras_layer._syft_args_store
    keras_layer_kwargs = keras_layer._syft_kwargs_store

    keras_layer_name = _get_layer_type(keras_layer)

    # Identify the TFE layer corresponding to the Keras layer
    tfe_layer_cls = getattr(tfe.keras.layers, keras_layer_name)

    # Extract argument list expected by layer __init__
    import inspect
    print(inspect.getfullargspec(tfe_layer_cls.__init__))
    tfe_arg_list = list(tfe_layer_cls.__dict__['__init__'].__code__.co_varnames)
    print("parameters", list(inspect.signature(tfe_layer_cls.__init__).parameters.keys()))
    print("arg_list", tfe_arg_list)

    # Remove arguments currenlty not supported by TFE layers
    tfe_arg_list = trim_tfe_args(tfe_arg_list, args_not_supported_by_tfe)

    # Load weights from Keras layer into TFE layer
    if 'kernel_initializer' in tfe_arg_list:
        kernel_weights = stored_keras_weights[keras_layer_attr['_name']][0]
        k_initializer = tf.keras.initializers.Constant(kernel_weights)
        keras_layer_attr['kernel_initializer'] = k_initializer

    if 'bias_initializer' in tfe_arg_list and keras_layer_attr['use_bias']:
        bias_weights = stored_keras_weights[keras_layer_attr['_name']][1]
        b_initializer = tf.keras.initializers.Constant(bias_weights)
        keras_layer_attr['bias_initializer'] = b_initializer

    # When the keras layer has been instantiated, the activation attribute
    # gets assigned an activation function (e.g. keras.activations.relu())
    # Needs to be converted back to activation name (e.g. `relu`), so it
    # uses TFE activation.
    if 'activation' in tfe_arg_list:
        activation_name = keras_layer_attr['activation'].__name__
        keras_layer_attr['activation'] = activation_name

    tfe_kwargs = {k: keras_layer_attr[k] for k in tfe_arg_list}

    if batch_input_shape is not None:
      tfe_kwargs['batch_input_shape'] = batch_input_shape

    return tfe_layer_cls(**tfe_kwargs)


def _get_layer_type(keras_layer_cls):
    return keras_layer_cls.__class__.__name__

def trim_tfe_args(tfe_args, not_supported_args):

    return [a for a in tfe_args if a not in not_supported_args]
