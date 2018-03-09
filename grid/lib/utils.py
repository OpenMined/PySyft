from grid import ipfsapi
from pathlib import Path
import os
import json
import time
from colorama import Fore, Style
import sys
import numpy as np


def get_ipfs_api(ipfs_addr='127.0.0.1', port=5001, max_tries=10):
    print(f'\n{Fore.BLUE}UPDATE: {Style.RESET_ALL}Connecting to IPFS... this can take a few seconds...')

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
