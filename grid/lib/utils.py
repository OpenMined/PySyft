from filelock import FileLock
from grid import ipfsapi
from pathlib import Path
import keras
import os
import json
import time
from colorama import Fore, Style
import sys
import numpy as np


def get_ipfs_api(ipfs_addr='127.0.0.1', port=5001, max_tries=10):
    print(f'\n{Fore.BLUE}UPDATE: {Style.RESET_ALL}Connecting to IPFS... this can take a few seconds...')

    # out = ipfsapi.connect(ipfs_addr, port)
    # print(f'\n{Fore.GREEN}SUCCESS: {Style.RESET_ALL}Connected!!!')
    # return out

    try:
        out = ipfsapi.connect(ipfs_addr, port)
        print(f'\n{Fore.GREEN}SUCCESS: {Style.RESET_ALL}Connected!!! - My ID: ' + str(out.config_show()['Identity']['PeerID']))
        return out
    except:
        print(f'\n{Fore.RED}ERROR: {Style.RESET_ALL}could not connect to IPFS.  Is your daemon running with pubsub support at {ipfs_addr} on port {port}? Let me try to start IPFS for you... (this will take ~15 seconds)')
        os.system('ipfs daemon --enable-pubsub-experiment  > ipfs.log 2> ipfs.log.err &')
        for i in range(15):
            sys.stdout.write('.')
            time.sleep(1)

            try:
                out = ipfsapi.connect(ipfs_addr, port)
                print(f'\n{Fore.GREEN}SUCCESS: {Style.RESET_ALL}Connected!!! - My ID: ' + str(out.config_show()['Identity']['PeerID']))
                return out
            except:
                ""

    for try_index in range(max_tries):
        try:
            out = ipfsapi.connect(ipfs_addr, port)
            print(f'\n{Fore.GREEN}SUCCESS: {Style.RESET_ALL}Connected!!! - My ID: ' + str(out.config_show()['Identity']['PeerID']))
            return out
        except:
            print(f'\n{Fore.RED}ERROR: {Style.RESET_ALL}still could not connect to IPFS.  Is your daemon running with pubsub support at {ipfs_addr} on port {port}?')
            time.sleep(5)

    print(f'\n{Fore.RED}ERROR: {Style.RESET_ALL}could not connect to IPFS. Failed after ' + str(max_tries) + ' attempts... Is IPFS installed? Consult the README at https://github.com/OpenMined/Grid')
    sys.exit()


def save_adapter(addr):
    adapter_bin = get_ipfs_api().cat(addr)
    ensure_exists(f'{Path.home()}/grid/adapters/adapter.py', adapter_bin)


def keras2ipfs(model):
    return get_ipfs_api().add_bytes(serialize_keras_model(model))


def ipfs2keras(model_addr):
    model_bin = get_ipfs_api().cat(model_addr)
    return deserialize_keras_model(model_bin)


def serialize_numpy(tensor):
    # nested lists with same data, indices
    return json.dumps(tensor.tolist())


def deserialize_numpy(json_array):
    return np.array(json.loads(json_array)).astype('float')


def serialize_torch_model(model, **kwargs):
    """
    kwargs are the arguments needed to instantiate the model
    """
    state = {'state_dict': model.state_dict(), 'kwargs': kwargs}
    torch.save(state, 'temp_model.pth.tar')
    with open('temp_model.pth.tar', 'rb') as f:
        model_bin = f.read()
    return model_bin


def deserialize_torch_model(model_bin, model_class, **kwargs):
    """
    model_class is needed since PyTorch uses pickle for serialization
        see https://discuss.pytorch.org/t/loading-pytorch-model-without-a-code/12469/2 for details
    kwargs are the arguments needed to instantiate the model from model_class
    """
    with open('temp_model2.pth.tar', 'wb') as g:
        g.write(model_bin)
    state = torch.load()
    model = model_class(**state['kwargs'])
    model.load_state_dict(state['state_dict'])
    return model


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


def store_whoami(info):
    ensure_exists(f'{Path.home()}/.openmined/whoami.json', info)


def load_whoami():
    if not os.path.exists(f'{Path.home()}/.openmined/whoami.json'):
        return None

    with open(f'{Path.home()}/.openmined/whoami.json', 'r') as cb:
        return json.loads(cb.read())


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
