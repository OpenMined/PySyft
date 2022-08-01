import io
from socket import timeout
import numpy as np
import os
import shutil
from .tff_messages import TFFMessageWithReply
from typing import OrderedDict
try:
    import tensorflow as tf
    import tensorflow_federated as tff
    from tensorflow_federated.python.learning.model_utils import ModelWeights 
except:
    pass


def train_model(model_fn, params, domain, timeout=300):
    # disabled in order to keep the names of the layers of the keras model 
    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
    model = model_fn()

    # Save the model using a functional TFF model
    train_data_shape = list(domain.store.get(params['train_data_id']).shape)
    train_data_shape[0] = 1
    
    label_data_shape = list(domain.store.get(params['label_data_id']).shape)
    if len(label_data_shape) == 1:
        label_data_shape = [1,1]
    else:
        label_data_shape[0] = 1
        
    input_spec = (tf.TensorSpec(shape=tuple(train_data_shape), dtype=tf.int64, name=None), 
                tf.TensorSpec(shape=tuple(label_data_shape), dtype=tf.int64, name=None))

    functional_model = tff.learning.models.functional_model_from_keras(keras_model=model, 
                                                                    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
                                                                    input_spec=input_spec)
    tff.learning.models.save_functional_model(functional_model=functional_model, path='_tmp')

    # Create one file from the new directory and read the bytes
    shutil.make_archive('_archive', 'zip', '_tmp')
    with open('_archive.zip', 'rb') as f:
        model_bytes = f.read()
    os.remove('_archive.zip')
    shutil.rmtree('_tmp')

    # Send the train message to the domain
    msg = TFFMessageWithReply(params=params, model_bytes=model_bytes)
    reply_msg = domain.send_immediate_msg_with_reply(msg, timeout=timeout)
    
    # Read the serialized weights 
    payload = eval(reply_msg.payload)
    
    memfile_trainable = io.BytesIO()
    memfile_trainable.write(payload['trainable'])
    memfile_trainable.seek(0)
    trainable_weights = np.load(memfile_trainable, allow_pickle=True)
    
    memfile_non_trainable = io.BytesIO()
    memfile_non_trainable.write(payload['non_trainable'])
    memfile_non_trainable.seek(0)
    non_trainable_weights = np.load(memfile_non_trainable, allow_pickle=True)

    # Create a new model with the trained weights
    modelweights = ModelWeights(list(trainable_weights), list(non_trainable_weights))
    model = model_fn()
    modelweights.assign_weights_to(model)
    
    return model, payload['metrics']