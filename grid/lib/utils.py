from grid import ipfsapi
from pathlib import Path
import os
import json
import time
from colorama import Fore, Style
import sys
import numpy as np


def get_ipfs_api(mode, ipfs_addr='127.0.0.1', port=5001, max_tries=25):
    print(
        f'\n{Fore.BLUE}UPDATE: {Style.RESET_ALL}Connecting to IPFS... this can take a few seconds...'
    )

    api = _attempt_ipfs_connection(ipfs_addr, port, 0, 1)
    if api:
        id = get_id(mode, api)
        print(
            f'\n{Fore.GREEN}SUCCESS: {Style.RESET_ALL}Connected!!! - My ID: {id}'
        )
        return api

    print(
        f'\n{Fore.RED}ERROR: {Style.RESET_ALL}could not connect to IPFS.  Is your daemon running with pubsub support at {ipfs_addr} on port {port}? Let me try to start IPFS for you... (this will take ~15 seconds)'
    )
    os.system(
        'ipfs daemon --enable-pubsub-experiment  > ipfs.log 2> ipfs.log.err &')

    api = _attempt_ipfs_connection(ipfs_addr, port, 0, max_tries, _write_dot)
    if api:
        id = get_id(mode, api)
        print(
            f'\n{Fore.GREEN}SUCCESS: {Style.RESET_ALL}Connected!!! - My ID: {id}'
        )
        return api

    print(
        f'\n{Fore.RED}ERROR: {Style.RESET_ALL}could not connect to IPFS. Failed after {max_tries} attempts... Is IPFS installed? Consult the README at https://github.com/OpenMined/Grid'
    )
    sys.exit()


def _write_dot():
    sys.stdout.write('.')


def _attempt_ipfs_connection(ipfs_addr,
                             port,
                             current_tries=0,
                             max_tries=10,
                             progress_fn=None):
    current_tries += 1
    try:
        api = ipfsapi.connect(ipfs_addr, port)
        return api
    except:
        if current_tries == max_tries:
            return False

        if progress_fn:
            progress_fn()

        time.sleep(1)
        return _attempt_ipfs_connection(ipfs_addr, port, current_tries,
                                        max_tries)


def get_ipfs_id(api):
    return api.config_show()['Identity']['PeerID']


def get_id(node_type, api):
    peer_id = get_ipfs_id(api)
    return derive_id(node_type, peer_id)


def derive_id(node_type, peer_id):
    node_type = node_type.lower()
    return f'{node_type}:{peer_id}'


def save_adapter(ipfs, addr):
    adapter_bin = ipfs.cat(addr)
    ensure_exists(f'{Path.home()}/grid/adapters/adapter.py', adapter_bin)


def serialize_numpy(tensor):
    # nested lists with same data, indices
    return json.dumps(tensor.tolist())


def deserialize_numpy(json_array):
    return np.array(json.loads(json_array)).astype('float')


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

    task = {'name': name, 'address': address}

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
                elif isinstance(default_contents, list) or isinstance(
                        default_contents, dict):
                    json.dump(default_contents, f)
                else:
                    # Not sure what this is, try to tostring it.
                    f.write(str(default_contents))

                f.close()
