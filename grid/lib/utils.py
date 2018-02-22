from filelock import Timeout, FileLock
from grid import ipfsapi
from pathlib import Path
import keras
import os
import json
from colorama import Fore, Back, Style
import sys


def get_ipfs_api(ipfs_addr='127.0.0.1', port=5001):
    try:
        return ipfsapi.connect(ipfs_addr, port)
    except:
        print(f'\n{Fore.RED}ERROR: {Style.RESET_ALL}could not connect to IPFS.  Is your daemon running with pubsub support at {ipfs_addr} on port {port}')
        sys.exit()

def save_adapter(addr):
    adapter_bin = get_ipfs_api().cat(addr)
    ensure_exists(f'{Path.home()}/grid/adapters/adapter.py', adapter_bin)

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

# def load_tasks():

def save_best_model_for_task(task, model):
    ensure_exists(f'{Path.home()}/.openmined/models.json', {})
    with open(f"{Path.home()}/.openmined/models.json", "r") as model_file:
        models = json.loads(model_file.read())

    models[task] = keras2ipfs(model)

    with open(f"{Path.home()}/.openmined/models.json", "w") as model_file:
        json.dump(models, model_file)


def best_model_for_task(task, return_model=False):
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

def load_task(name):
    if not os.path.exists(f'{Path.home()}/.openmined/tasks.json'):
        return None

    with open(f'{Path.home()}/.openmined/tasks.json', 'r') as task_file:
        tasks = json.loads(task_file.read())

    for task in tasks:
        if task['name'] == name:
            return task

def store_task(name, address):
    ensure_exists(f'{Path.home()}/.openmined/tasks.json', [])
    with open(f"{Path.home()}/.openmined/tasks.json", "r") as task_file:
        tasks = json.loads(task_file.read())

    task = {
        'name': name,
        'address': address
    }

    if task not in tasks:
        print("storing task", task['name'])
        tasks.append(task)

        with open(f"{Path.home()}/.openmined/tasks.json", "w") as task_file:
            json.dump(tasks, task_file)


def ensure_exists(path, default_contents=None):
    """
    Ensure that a path exists.  You can pass as many subdirectories as you
    want without verifying that the parent exists.

    E.g.

    ensure_exists('~/.openmined/adapters/config.json', {}) will ensure that
    the file `~/.openmined/adapters/config.json` gets created and that the file
    contents will become an empty object
    """
    parts = path.split('/')
    f = parts.pop()
    full_path = ""

    print(f'all parts .... {parts}')

    for p in parts:
        # convert ~ to the users home directory
        if p == '~':
            p = str(Path.home())

        full_path = f'{full_path}{p}/'
        print(f'full path {full_path}')
        if not os.path.exists(full_path):
            print('making dir.... {full_path}')
            os.makedirs(full_path)

    full_path = f'{full_path}{f}'
    if not os.path.exists(full_path):
        if isinstance(default_contents, bytes):
            with open(full_path, 'wb') as f:
                f.write(default_contents)
                f.close()
        else:
            with open(full_path, 'w') as f:
                if isinstance(default_contents, str):
                    f.write(default_contents)
                elif isinstance(default_contents, list) or isinstance(default_contents, dict):
                    json.dump(default_contents, f)
                else:
                    # Not sure what this is, try to tostring it.
                    f.write(str(default_contents))

                f.close()
