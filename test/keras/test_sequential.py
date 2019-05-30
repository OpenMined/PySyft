import pytest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
import syft as sy


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


def test_share():

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    hook = sy.KerasHook(tf.keras)

    input_shape = [4, 5]
    input_data = np.ones(input_shape)
    kernel = np.random.normal(size=[5, 5])
    initializer = tf.keras.initializers.Constant(kernel)

    model = Sequential()

    model.add(
        Dense(5, kernel_initializer=initializer, batch_input_shape=input_shape, use_bias=True)
    )

    AUTO = True
    alice = sy.TFEWorker(host="localhost:4000", auto_managed=AUTO)
    bob = sy.TFEWorker(host="localhost:4001", auto_managed=AUTO)
    carol = sy.TFEWorker(host="localhost:4002", auto_managed=AUTO)

    model.share(alice, bob, carol)

    model.serve(num_requests=0)

    model.shutdown_workers()
