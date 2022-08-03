import pytest
import tensorflow_federated as tff
import tensorflow as tf
import syft as sy
import os
import json
import pandas as pd
from PIL import Image
from enum import Enum
from collections import defaultdict
import numpy as np
from syft.core.adp.data_subject_list import DataSubjectArray
  
def create_keras_model():
      return tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(100,)),
            tf.keras.layers.Dense(10, kernel_initializer='zeros'),
            tf.keras.layers.Softmax(),
      ])

def load_dataset(domain):
    data_subjects = DataSubjectArray.from_objs(['Test'])

    train_data = np.random.randint(256,size=(1,100))
    label_data = np.array([0])

    train_data_subjects = np.broadcast_to(np.array(data_subjects), train_data.shape)

    train_image_data = sy.Tensor(train_data).private(
    min_val=0, max_val=255, data_subjects=train_data_subjects
    )
    train_label_data = sy.Tensor(label_data).private(
        min_val=0, max_val=5, data_subjects=data_subjects
    )

    domain.load_dataset(
        name='Test',
        assets={
            'images': train_image_data,
            "labels": train_label_data
        },
        description="Small dataset for TFF testing"
    )

@pytest.mark.tff
def test_tff():
    assert tff.federated_computation(lambda: 'Hello World')() == b'Hello World'
    
@pytest.mark.tff
def test_training():
    domain = sy.login(email="info@openmined.org", password="changethis", port=9081)    
    load_dataset(domain)
    
    # Set params
    model_fn = create_keras_model 
    params = {
    'rounds': 1,
    'no_clients': 5,
    'noise_multiplier': 0.05,
    'clients_per_round': 2,
    'dataset_id': domain.datasets[0].id,
    'train_data_id': domain.datasets[0]['images'].id_at_location.to_string(),
    'label_data_id': domain.datasets[0]['labels'].id_at_location.to_string()
    }
    model, metrics = sy.tff.train_model(model_fn, params, domain)
    
    # Check Results
    l1 = model.layers
    l2 = create_keras_model().layers
    assert ([type(x) for x in l1] == [type(x) for x in l2])