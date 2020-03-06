import syft.tensorflow_ as tf_
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops


def test_constant_and_eagertensor_constructors():
    x = tf.constant([1, 2, 3, 4])

    assert isinstance(x, tf.Tensor)
    assert isinstance(x, ops.EagerTensor)

    assert (x.numpy() == np.array([1, 2, 3, 4])).all()


def test_variable_constructor():
    x = tf.Variable([1, 2, 3, 4])

    assert isinstance(x, tf.Variable)
    assert isinstance(x.value(), tf.Tensor)

    assert (x.numpy() == np.array([1, 2, 3, 4])).all()
