import pytest

import numpy as np
import syft as sy
from syft import dependency_check

if dependency_check.keras_available:
    import tensorflow as tf
    import tf_encrypted as tfe


@pytest.mark.skipif(not dependency_check.keras_available, reason="tf_encrypted not installed")
def test_instantiate_tfe_layer():

    from syft.frameworks.keras.model.sequential import _instantiate_tfe_layer

    hook = sy.KerasHook(tf.keras)

    input_shape = [4, 5]
    input_data = np.ones(input_shape)
    kernel = np.random.normal(size=[5, 5])
    initializer = tf.keras.initializers.Constant(kernel)

    d_tf = tf.keras.layers.Dense(
        5, kernel_initializer=initializer, batch_input_shape=input_shape, use_bias=True
    )

    with tf.Session() as sess:
        x = tf.Variable(input_data, dtype=tf.float32)
        y = d_tf(x)
        sess.run(tf.global_variables_initializer())
        expected = sess.run(y)

    stored_keras_weights = {d_tf.name: d_tf.get_weights()}

    with tf.Graph().as_default():
        p_x = tfe.define_private_variable(input_data)
        d_tfe = _instantiate_tfe_layer(d_tf, stored_keras_weights)

        out = d_tfe(p_x)

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())

            actual = sess.run(out.reveal())

    np.testing.assert_allclose(actual, expected, rtol=0.001)


@pytest.mark.skipif(not dependency_check.keras_available, reason="tf_encrypted not installed")
def test_share():

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    hook = sy.KerasHook(tf.keras)

    input_shape = (4, 5)
    kernel = np.random.normal(size=(5, 5))
    initializer = tf.keras.initializers.Constant(kernel)
    dummy_input = np.ones(input_shape).astype(np.float32)

    model = Sequential()

    model.add(
        Dense(5, kernel_initializer=initializer, batch_input_shape=input_shape, use_bias=True)
    )
    output_shape = model.output_shape
    result_public = model.predict(dummy_input)

    client = sy.TFEWorker(host=None)
    alice = sy.TFEWorker(host=None)
    bob = sy.TFEWorker(host=None)
    carol = sy.TFEWorker(host=None)
    cluster = sy.TFECluster(alice, bob, carol)

    cluster.start()

    model.share(cluster)

    with model._tfe_graph.as_default():
        client.connect_to_model(input_shape, output_shape, cluster, sess=model._tfe_session)

    client.query_model_async(dummy_input)

    model.serve(num_requests=1)

    result_private = client.query_model_join().astype(np.float32)
    np.testing.assert_allclose(result_private, result_public, atol=0.01)

    model.stop()

    cluster.stop()
