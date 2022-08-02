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
from syft.core.adp.data_subject_list import DataSubjectList

def create_keras_model():
  return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(64,64,1), name='input'),
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu", name='conv1'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
      #   tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name='conv2'),
      #   tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
      #   tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv3'),
      #   tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
      #   tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", name='conv4'),
      #   tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool4'),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, kernel_initializer='zeros'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(6, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
  ])
  
def get_label_mapping():
    # the data uses the following mapping
    mapping = {
        "AbdomenCT": 0, 
        "BreastMRI": 1, 
        "CXR": 2, 
        "ChestCT": 3, 
        "Hand": 4, 
        "HeadCT": 5
    }
    return mapping

def load_dataset(domain):
    df = pd.read_pickle("./MedNIST.pkl")
    mapping = get_label_mapping()

    subset_idx = []
    size = 100
    step = 10000
    for i in range(len(mapping)):
        subset_idx.extend(list(range(step * i, step* i + size)))


    data_subjects = DataSubjectList.from_series(df['patient_id'][subset_idx])

    images = df['image'][subset_idx]
    images = np.dstack(images.values).astype(np.int64)
    images = np.rollaxis(images,-1)

    labels = df['label'][subset_idx].to_numpy().astype("int64")
    train_image_data = sy.Tensor(images).annotated_with_dp_metadata(
    min_val=0, max_val=255, data_subjects=data_subjects
    )
    train_label_data = sy.Tensor(labels).annotated_with_dp_metadata(
        min_val=0, max_val=5, data_subjects=data_subjects
    )

    domain.load_dataset(
        name='Mixed MedNIST 64 fo real',
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
    domain = sy.login(email="info@openmined.org", password="changethis", port=9082)    
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