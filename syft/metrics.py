""" Based upon https://github.com/keras-team/keras/blob/master/keras/metrics.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import six

def binary_accuracy(y_true, y_pred):
    return np.mean(np.equal(y_true, np.round(y_pred)).astype(float), axis=-1)

def categorical_accuracy(y_true, y_pred):
    return np.mean((np.equal(np.argmax(y_true, axis=-1),
                            np.argmax(y_pred, axis=-1))).astype(float))

def sparse_categorical_accuracy(y_true, y_pred):
    return np.mean(np.equal(np.max(y_true, axis=-1),
                            np.argmax(y_pred, axis=-1)).astype(float))

def top_k_categorical_accuracy(y_true, y_pred, k=5):
    pass

def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    pass

def mean_squared_error(y_true, y_pred):
    pass

def mean_absolute_error(y_true, y_pred):
    pass

def mean_absolute_percentage_error(y_true, y_pred):
    pass

def mean_squared_logarithmic_error(y_true, y_pred):
    pass

def cosine_proximity(y_true, y_pred):
    pass


# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity
accuracy = acc = categorical_accuracy

def get(identifier):
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        function_name = identifier
        fn = globals().get(function_name)
        if fn is None:
            raise ValueError('Unknown metric function : ' + function_name)
        return fn
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)