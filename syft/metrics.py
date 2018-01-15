""" Based upon https://github.com/keras-team/keras/blob/master/keras/metrics.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import six
import math

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def binary_accuracy(y_true, y_pred):
    return np.mean(np.equal(y_true, np.round(y_pred)).astype(float), axis=-1)

def categorical_accuracy(y_true, y_pred):
    return np.mean((np.equal(np.argmax(y_true, axis=-1),
                            np.argmax(y_pred, axis=-1))).astype(float))

def sparse_categorical_accuracy(y_true, y_pred):
    return np.mean(np.equal(np.max(y_true, axis=-1),
                            np.argmax(y_pred, axis=-1)).astype(float))

def top_k_categorical_accuracy(y_true, y_pred, k=5):
    top_k = np.argsort(-y_pred)[:,:k]
    return np.mean([np.in1d(np.argmax(y_true[i]), top_k[i]) for i in range(y_true.shape[0])])

def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    top_k = np.argsort(-y_pred)[:,:k]
    return np.mean([np.in1d(np.max(y_true[i]), top_k[i]) for i in range(y_true.shape[0])])

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.absolute(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return 100*mean_absolute_error(y_true, y_pred)

def mean_squared_logarithmic_error(y_true, y_pred):
    return np.mean(np.power(np.log1p(y_true)-np.log1p(y_pred), 2))

def cosine_proximity(y_true, y_pred):
    cosine_prox = 0
    for i in range(y_true.shape[0]):
        cosine_prox += cosine_similarity(y_true[i], y_pred[i])
    return cosine_prox / float(y_true.shape[0])


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