import os
import json
import time
from colorama import Fore, Style
import sys
import numpy as np

from filelock import FileLock
from grid import ipfsapi
from pathlib import Path

import utils

import keras

def keras2ipfs(model):
    return get_ipfs_api().add_bytes(serialize_keras_model(model))

def ipfs2keras(model_addr):
    model_bin = get_ipfs_api().cat(model_addr)
    return deserialize_keras_model(model_bin)

def serialize_keras_model(model):
    lock = FileLock('temp_model.h5.lock')
    with lock:
        model.save('temp_model.h5')
        with open('temp_model.h5', 'rb') as f:
            model_bin = f.read()
            f.close()
        return model_bin


def deserialize_keras_model(model_bin):
    lock = FileLock('temp_model2.h5.lock')
    with lock:
        with open('temp_model2.h5', 'wb') as g:
            g.write(model_bin)
            g.close()
        model = keras.models.load_model('temp_model2.h5')
        return model

def save_best_keras_model_for_task(task, model):
    ensure_exists(f'{Path.home()}/.openmined/models.json', {})
    with open(f"{Path.home()}/.openmined/models.json", "r") as model_file:
        models = json.loads(model_file.read())

    models[task] = keras2ipfs(model)

    with open(f"{Path.home()}/.openmined/models.json", "w") as model_file:
        json.dump(models, model_file)

def best_keras_model_for_task(task, return_model=False):
    if not os.path.exists(f'{Path.home()}/.openmined/models.json'):
        return None

    with open(f'{Path.home()}/.openmined/models.json', 'r') as model_file:
        models = json.loads(model_file.read())
        if task in models.keys():
            if return_model:
                return ipfs2keras(models[task])
            else:
                return models[task]

    return None

