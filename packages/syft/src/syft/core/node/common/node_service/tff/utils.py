# stdlib
import ast
import io
import os
import shutil
from typing import Callable
from typing import Union

# third party
import numpy as np

# relative
from ...client import Client
from .tff_messages import TFFMessageWithReply

try:
    # third party
    import tensorflow as tf
    import tensorflow_federated as tff
    from tensorflow_federated.python.learning.model_utils import ModelWeights
except Exception:  # nosec
    # no tff
    pass


def train_model(
    model_fn: Callable, params: dict, domain: Client, timeout: int = 300
) -> "Union[tf.keras.Model, dict]":
    # disabled in order to keep the names of the layers of the keras model
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    model = model_fn()

    # Save the model using a functional TFF model
    train_data_pointer = domain.store.get(params["train_data_id"])
    train_data_shape = list(train_data_pointer.shape)  # type: ignore
    train_data_shape[0] = 1

    label_data_pointer = domain.store.get(params["label_data_id"])
    label_data_shape = list(label_data_pointer.shape)  # type: ignore
    if len(label_data_shape) == 1:
        label_data_shape = [1, 1]
    else:
        label_data_shape[0] = 1

    input_spec = (
        tf.TensorSpec(shape=tuple(train_data_shape), dtype=tf.int64, name=None),
        tf.TensorSpec(shape=tuple(label_data_shape), dtype=tf.int64, name=None),
    )

    functional_model = tff.learning.models.functional_model_from_keras(
        keras_model=model,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=input_spec,
    )
    tff.learning.models.save_functional_model(
        functional_model=functional_model, path="_tmp"
    )

    # Create one file from the new directory and read the bytes
    shutil.make_archive("_archive", "zip", "_tmp")
    with open("_archive.zip", "rb") as f:
        model_bytes = f.read()
    os.remove("_archive.zip")
    shutil.rmtree("_tmp")

    # Send the train message to the domain
    msg = TFFMessageWithReply(params=params, model_bytes=model_bytes)
    reply_msg = domain.send_immediate_msg_with_reply(msg, timeout=timeout)

    # Read the serialized weights
    payload = ast.literal_eval(str(reply_msg.payload))  # type: ignore

    memfile_trainable = io.BytesIO()
    memfile_trainable.write(payload["trainable"])
    memfile_trainable.seek(0)
    trainable_weights = np.load(memfile_trainable, allow_pickle=True)

    memfile_non_trainable = io.BytesIO()
    memfile_non_trainable.write(payload["non_trainable"])
    memfile_non_trainable.seek(0)
    non_trainable_weights = np.load(memfile_non_trainable, allow_pickle=True)

    # Create a new model with the trained weights
    modelweights = ModelWeights(list(trainable_weights), list(non_trainable_weights))
    model = model_fn()
    modelweights.assign_weights_to(model)

    return model, payload["metrics"]
