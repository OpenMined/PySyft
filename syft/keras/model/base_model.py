import inspect

import tensorflow as tf
from tensorflow.keras.models import Sequential

import tf_encrypted as tfe
from tf_encrypted.keras.engine.sequential import Sequential as tfe_Sequential
from tf_encrypted.keras.engine.input_layer import InputLayer as tfe_InputLayer

# When instantiating tfe layers, exclude not supported arguments in TFE.
_args_not_supported_by_tfe = [
    'activity_regularizer',
    'kernel_regularizer',
    'bias_regularizer',
    'kernel_constraint',
    'bias_constraint',
    'dilation_rate',
]


class Sequential(Sequential):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__(*args, **kwargs)

    def share(self, prot=None, target_graph=None):

        # Store Keras weights before loading them in the TFE layers.
        stored_keras_weights = {}

        # TODO(Morten) we could optimize runtime by running a single model.get_weights instead
        for keras_layer in self.layers:
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
                for keras_layer in self.layers:
                    # Don't need to instantiate 'InputLayer'. Will be automatically
                    # created when `mode.predict()`.
                    tfe_layer = _instantiate_tfe_layer(keras_layer, stored_keras_weights)
                    tfe_model.add(tfe_layer)

        # TODO(Morten) we should keep a reference to the graph instead of returning it
        return target_graph, tfe_model


def _instantiate_tfe_layer(keras_layer, stored_keras_weights):

    # Get original layer's constructor parameters
    # This method is added by the KerasHook object in syft.keras
    constructor_params = keras_layer._constructor_parameters_store
    constructor_params.apply_defaults()
    _trim_bound_tf_args(constructor_params, _args_not_supported_by_tfe)

    # Identify tf.keras layer type and pull its attributes
    keras_layer_name = _get_layer_type(keras_layer)
    keras_layer_attr = keras_layer.__dict__

    # Identify the corresponding tfe.keras layer
    tfe_layer_cls = getattr(tfe.keras.layers, keras_layer_name)

    # Extract argument list expected by layer __init__
    tfe_arg_list = list(inspect.signature(tfe_layer_cls.__init__).parameters.keys())

    # Remove arguments currently not supported by TFE layers
    tfe_arg_list = _trim_tfe_args(tfe_arg_list, _args_not_supported_by_tfe)

    # Load weights from Keras layer into TFE layer
    # TODO[jason]: generalize to n weights -- will require special handling for
    #              optional weights like bias (or batchnorm trainables)
    if 'kernel_initializer' in constructor_params.arguments:
        kernel_weights = stored_keras_weights[keras_layer.name][0]
        k_initializer = tf.keras.initializers.Constant(kernel_weights)
        constructor_params.arguments['kernel_initializer'] = k_initializer

    if 'use_bias' in constructor_params.arguments:
        if constructor_params.arguments['use_bias']:
            bias_weights = stored_keras_weights[keras_layer.name][1]
            b_initializer = tf.keras.initializers.Constant(bias_weights)
            constructor_params.arguments['bias_initializer'] = b_initializer

    unpacked_kwargs = constructor_params.arguments.pop('kwargs')
    tfe_kwargs = {**constructor_params.arguments, **unpacked_kwargs}

    return tfe_layer_cls(**tfe_kwargs)

def _get_layer_type(keras_layer_cls):
    return keras_layer_cls.__class__.__name__

def _trim_bound_tf_args(bound_args, not_supported_args):
    filter_list = not_supported_args + ['self']
    for arg_name in filter_list:
        try:
            del bound_args.arguments[arg_name]
        except KeyError:
            continue

def _trim_tfe_args(tfe_args, not_supported_args):
    filter_list = not_supported_args + ['self', 'kwargs']
    return [a for a in tfe_args if a not in filter_list]
