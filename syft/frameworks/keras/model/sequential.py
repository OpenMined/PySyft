from collections import defaultdict, OrderedDict
import inspect
import subprocess

import tensorflow as tf

import tf_encrypted as tfe

# When instantiating tfe layers, exclude not supported arguments in TFE.
_args_not_supported_by_tfe = [
    'activity_regularizer',
    'kernel_regularizer',
    'bias_regularizer',
    'kernel_constraint',
    'bias_constraint',
    'dilation_rate',
]


def share(model, *workers, prot=None, target_graph=None):

    # Handle input combinations to produce protocol
    prot = _sanitize_share_argspec(workers, prot)

    # NOTE If protocol not set, getting errors with
    # tfe.define_private_placeholder in tfe.keras.Sequential
    # TODO[jason]: investigate
    tfe.set_protocol(prot)

    # Launch tf servers and store the subprocess results on the model
    _launch_tfserver_subprocesses(model)

    if target_graph is None:
        # By default we create a new graph for the shared model
        target_graph = tf.Graph()

    # Store Keras weights before loading them in the TFE layers.
    stored_keras_weights = {}

    # TODO(Morten) we could optimize runtime by running a single model.get_weights instead
    for keras_layer in model.layers:
        stored_keras_weights[keras_layer.name] = keras_layer.get_weights()

    with target_graph.as_default():
        tfe_model, batch_input_shape = _rebuild_tfe_model(model, stored_keras_weights)

    # Set up a new tfe.serving.QueueServer for the shared TFE model
    q_input_shape = batch_input_shape
    q_output_shape = model.output_shape

    with target_graph.as_default():
        server = tfe.serving.QueueServer(
            input_shape=q_input_shape,
            output_shape=q_output_shape,
            computation_fn=tfe_model,
        )

        initializer = tf.global_variables_initializer()

    model._server = server

    sess = tfe.Session(graph=target_graph)
    tf.Session.reset(sess.target)
    sess.run(initializer, tag='init')

    model._tfe_session = sess


def serve(model, num_steps=5):

    def step_fn():
        print("Served")

    model._server.run(
        model._tfe_session,
        num_steps=num_steps,
        step_fn=step_fn,
    )


_protocol_map = defaultdict(
    tfe.get_protocol,
    {'securenn': tfe.protocol.SecureNN, 'pond': tfe.protocol.Pond},
)


def _sanitize_share_argspec(workers, prot):
    if not workers and prot is None:
        raise RuntimeError("Invalid arguments supplied to model.share(): "
                           "must supply TensorflowServerWorkers or a protocol object that corresponds")

    if workers:
        if len(workers) != 3:
            raise RuntimeError("TF Encrypted expects three parties for its sharing protocols.")

        players = OrderedDict([('server{}'.format(i), workers[i].host) for i in range(len(workers))])
        config = tfe.RemoteConfig(players)
        config.save('/tmp/tfe.config')
        tfe.set_config(config)


    if prot is None:
        # Default is SecureNN
        prot = 'securenn'
    if isinstance(prot, str):
        # here lies the dumbest, sneakiest bug of my life, so far...
        # if you move this constructor call into _protocol_map,
        # the protocol objects will be initialized before set_config above is called,
        # which means any direct calls to prot will use the default device names
        # (which come from LocalConfig). so we return the class from _protocol_map
        # and instantiate it here
        prot = _protocol_map[prot]()
    return prot


def _launch_tfserver_subprocesses(model):
    subprocess_calls = {}
    cmd = "python -m tf_encrypted.player --config /tmp/tfe.config {}"
    hostmap = tfe.get_config().hostmap
    for player_name in hostmap.keys():
        subprocess_calls[player_name] = subprocess.Popen(cmd.format(player_name).split(' '))
    setattr(model, '_subprocess_calls', subprocess_calls)


def _rebuild_tfe_model(keras_model, stored_keras_weights):
    tfe_model = tfe.keras.Sequential()

    for keras_layer in keras_model.layers:
        tfe_layer = _instantiate_tfe_layer(keras_layer, stored_keras_weights)
        tfe_model.add(tfe_layer)

        if hasattr(tfe_layer, '_batch_input_shape'):
            batch_input_shape = tfe_layer._batch_input_shape

    return tfe_model, batch_input_shape


def _instantiate_tfe_layer(keras_layer, stored_keras_weights):

    # Get original layer's constructor parameters
    # This method is added by the KerasHook object in syft.keras
    constructor_params = keras_layer._constructor_parameters_store
    constructor_params.apply_defaults()
    _trim_params(constructor_params.arguments, _args_not_supported_by_tfe + ['self'])

    # Identify tf.keras layer type, and grab the corresponding tfe.keras layer
    keras_layer_type = _get_layer_type(keras_layer)
    try:
        tfe_layer_cls = getattr(tfe.keras.layers, keras_layer_type)
    except AttributeError:
        # TODO: rethink how we warn the user about this, maybe codegen a list of
        #       supported layers in a doc somewhere
        raise RuntimeError("TF Encrypted doesn't yet support the "
                           "{lcls} layer.".format(lcls=keras_layer_type))

    # Extract argument list expected by layer __init__
    # TODO[jason]: find a better way
    tfe_params = list(inspect.signature(tfe_layer_cls.__init__).parameters.keys())

    # Remove arguments currently not supported by TFE layers
    _trim_params(tfe_params, _args_not_supported_by_tfe + ['self', 'kwargs'])

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


def _trim_params(params, filter_list):
    for arg_name in filter_list:
        try:
            del params[arg_name]
        except TypeError:
            # params is a list of strings
            if arg_name in params:
                params.remove(arg_name)
        except KeyError:
            continue
