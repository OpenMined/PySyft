from collections import defaultdict, OrderedDict
import inspect

import tensorflow as tf
import tf_encrypted as tfe

# When instantiating tfe layers, exclude not supported arguments in TFE.
_args_not_supported_by_tfe = [
    "activity_regularizer",
    "kernel_regularizer",
    "bias_regularizer",
    "kernel_constraint",
    "bias_constraint",
    "dilation_rate",
]


def share(model, *workers, target_graph=None):  # pragma: no cover
    """
    Secret share the model between `workers`.

    This is done by rebuilding the model as a TF Encrypted model inside `target_graph`
    and pushing this graph to TensorFlow servers running on the workers.
    """

    # Store Keras weights before loading them in the TFE layers.
    # TODO(Morten) we could optimize runtime by running a single model.get_weights instead
    stored_keras_weights = {
        keras_layer.name: keras_layer.get_weights() for keras_layer in model.layers
    }

    # Handle input combinations to configure TFE
    player_to_worker_mapping = _configure_tfe(workers)

    if target_graph is None:
        # By default we create a new graph for the shared model
        target_graph = tf.Graph()

    with target_graph.as_default():
        tfe_model, batch_input_shape = _rebuild_tfe_model(model, stored_keras_weights)

        # Set up a new tfe.serving.QueueServer for the shared TFE model
        q_input_shape = batch_input_shape
        q_output_shape = model.output_shape

        server = tfe.serving.QueueServer(
            input_shape=q_input_shape, output_shape=q_output_shape, computation_fn=tfe_model
        )

        initializer = tf.global_variables_initializer()

    model._server = server
    model._workers = workers

    # Tell the TFE workers to launch TF servers
    for player_name, worker in player_to_worker_mapping.items():
        worker.start(player_name, *workers)

    # Push and initialize shared model on servers
    sess = tfe.Session(graph=target_graph)
    tf.Session.reset(sess.target)
    sess.run(initializer)

    model._tfe_session = sess


def serve(model, num_requests=5):
    global request_ix
    request_ix = 1

    def step_fn():
        global request_ix
        print("Served encrypted prediction {i} to client.".format(i=request_ix))
        request_ix += 1

    model._server.run(model._tfe_session, num_steps=num_requests, step_fn=step_fn)


def shutdown_workers(model):
    if model._tfe_session is not None:
        sess = model._tfe_session
        model._tfe_session = None
        sess.close()
        del sess
    for worker in model._workers:
        worker.stop()


def _configure_tfe(workers):

    if not workers or len(workers) != 3:
        raise RuntimeError("TF Encrypted expects three parties for its sharing protocols.")

    tfe_worker_cls = workers[0].__class__
    config, player_to_worker_mapping = tfe_worker_cls.config_from_workers(workers)
    tfe.set_config(config)

    prot = tfe.protocol.SecureNN(
        config.get_player("server0"), config.get_player("server1"), config.get_player("server2")
    )
    tfe.set_protocol(prot)

    return player_to_worker_mapping


def _rebuild_tfe_model(keras_model, stored_keras_weights):
    """
    Rebuild the plaintext Keras model as a TF Encrypted Keras model
    from the plaintext weights in `stored_keras_weights` using the
    current TensorFlow graph, and the current TF Encrypted protocol
    and configuration.
    """

    tfe_model = tfe.keras.Sequential()

    for keras_layer in keras_model.layers:
        tfe_layer = _instantiate_tfe_layer(keras_layer, stored_keras_weights)
        tfe_model.add(tfe_layer)

        if hasattr(tfe_layer, "_batch_input_shape"):
            batch_input_shape = tfe_layer._batch_input_shape

    return tfe_model, batch_input_shape


def _instantiate_tfe_layer(keras_layer, stored_keras_weights):

    # Get original layer's constructor parameters
    # This method is added by the KerasHook object in syft.keras
    constructor_params = keras_layer._constructor_parameters_store
    constructor_params.apply_defaults()
    _trim_params(constructor_params.arguments, _args_not_supported_by_tfe + ["self"])

    # Identify tf.keras layer type, and grab the corresponding tfe.keras layer
    keras_layer_type = _get_layer_type(keras_layer)
    try:
        tfe_layer_cls = getattr(tfe.keras.layers, keras_layer_type)
    except AttributeError:
        # TODO: rethink how we warn the user about this, maybe codegen a list of
        #       supported layers in a doc somewhere
        raise RuntimeError(
            "TF Encrypted does not yet support the " "{lcls} layer.".format(lcls=keras_layer_type)
        )

    # Extract argument list expected by layer __init__
    # TODO[jason]: find a better way
    tfe_params = list(inspect.signature(tfe_layer_cls.__init__).parameters.keys())

    # Remove arguments currently not supported by TFE layers
    _trim_params(tfe_params, _args_not_supported_by_tfe + ["self", "kwargs"])

    # Load weights from Keras layer into TFE layer
    # TODO[jason]: generalize to n weights -- will require special handling for
    #              optional weights like bias (or batchnorm trainables)
    if "kernel_initializer" in constructor_params.arguments:
        kernel_weights = stored_keras_weights[keras_layer.name][0]
        k_initializer = tf.keras.initializers.Constant(kernel_weights)
        constructor_params.arguments["kernel_initializer"] = k_initializer

    if "use_bias" in constructor_params.arguments:
        if constructor_params.arguments["use_bias"]:
            bias_weights = stored_keras_weights[keras_layer.name][1]
            b_initializer = tf.keras.initializers.Constant(bias_weights)
            constructor_params.arguments["bias_initializer"] = b_initializer

    unpacked_kwargs = constructor_params.arguments.pop("kwargs")
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
