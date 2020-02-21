import functools
import multiprocessing
import threading
import os
import re
import ast
from dill.source import getsource
from RestrictedPython import compile_restricted
from RestrictedPython import safe_builtins

import syft as sy
from syft.messaging.message import CryptenInit
from syft.frameworks import crypten as syft_crypt

import crypten
from crypten.communicator import DistributedCommunicator


def _check_func_def(func_src):
    # The body should contain one element
    tree = ast.parse(func_src)
    if len(tree.body) != 1:
        return (False, "")
    # The body should contain a function defintion
    func_def = tree.body[0]
    if type(func_def) is not ast.FunctionDef:
        return (False, "")

    return (True, func_def.name)


def _launch(func_src, rank, world_size, master_addr, master_port, queue, func_args, func_kwargs):
    communicator_args = {
        "RANK": rank,
        "WORLD_SIZE": world_size,
        "RENDEZVOUS": "env://",
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": master_port,
        "DISTRIBUTED_BACKEND": "gloo",
    }
    for key, val in communicator_args.items():
        os.environ[key] = str(val)

    # func_src should contain one and only one function definition
    is_func, func_name = _check_func_def(func_src)
    if not is_func:
        queue.put(-1)
        return

    # TODO: error handling
    exec_globals = {'__builtins__': safe_builtins}
    exec_globals['crypten'] = crypten
    exec_locals = {}
    compiled = compile_restricted(func_src)
    exec(compiled, exec_globals, exec_locals)
    func = exec_locals[func_name]
    crypten.init()
    return_value = func(*func_args, **func_kwargs)
    crypten.uninit()

    queue.put(return_value)


def _new_party(func_src, rank, world_size, master_addr, master_port, func_args, func_kwargs):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_launch,
        args=(func_src, rank, world_size, master_addr, master_port, queue, func_args, func_kwargs),
    )
    return process, queue


def run_party(func_src, rank, world_size, master_addr, master_port, func_args, func_kwargs):
    """Start crypten party localy and run computation.

    Args:
        func_src (str): function source code.
        rank (int): rank of the crypten party.
        world_size (int): number of crypten parties involved in the computation.
        master_addr (str): IP address of the master party (party with rank 0).
        master_port (int or str): port of the master party (party with rank 0).
        func_args (list): arguments to be passed to func.
        func_kwargs (dict): keyword arguments to be passed to func.

    Returns:
        The return value of the executed function.
    """

    process, queue = _new_party(
        func_src, rank, world_size, master_addr, master_port, func_args, func_kwargs
    )
    was_initialized = DistributedCommunicator.is_initialized()
    if was_initialized:
        crypten.uninit()
    process.start()
    process.join()
    if was_initialized:
        crypten.init()
    return queue.get()


def _send_party_info(worker, rank, msg, return_values):
    """Send message to worker with necessary information to run a crypten party.
    Add response to return_values dictionary.

    Args:
        worker (BaseWorker): worker to send the message to.
        rank (int): rank of the crypten party.
        msg (CryptenInit): message containing the rank, world_size, master_addr and master_port.
        return_values (dict): dictionnary holding return values of workers.
    """

    response = worker.send_msg(msg, worker)
    return_values[rank] = response.contents


def toy_func():
    alice_tensor = syft_crypt.load("crypten_data", 1, "alice")
    bob_tensor = syft_crypt.load("crypten_data", 2, "bob")

    crypt = crypten.cat([alice_tensor, bob_tensor], dim=0)
    return crypt.get_plain_text().tolist()


def run_multiworkers(workers: list, master_addr: str, master_port: int = 15987):
    """Defines decorator to run function across multiple workers.

    Args:
        workers (list): workers (parties) to be involved in the computation.
        master_addr (str): IP address of the master party (party with rank 0).
        master_port (int, str): port of the master party (party with rank 0), default is 15987.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # TODO:
            # - check if workers are reachable / they can handle the computation
            # - check return code of processes for possible failure

            world_size = len(workers) + 1
            return_values = {rank: None for rank in range(world_size)}
            
            # Get func_src without decorators
            re_decorator = r'@[^\(]+\([^\)]*\)'
            func_src = re.sub(re_decorator, '', getsource(func))

            # Start local party
            process, queue = _new_party(func_src, 0, world_size, master_addr, master_port, (), {})
            was_initialized = DistributedCommunicator.is_initialized()
            if was_initialized:
                crypten.uninit()
            process.start()
            # Run TTP if required
            # TODO: run ttp in a specified worker
            if crypten.mpc.ttp_required():
                ttp_process, _ = _new_party(
                    getsource(crypten.mpc.provider.TTPServer),
                    world_size,
                    world_size,
                    master_addr,
                    master_port,
                    (),
                    {},
                )
                ttp_process.start()

            # Send messages to other workers so they start their parties
            threads = []
            for i in range(len(workers)):
                rank = i + 1
                msg = CryptenInit((func_src, rank, world_size, master_addr, master_port))
                thread = threading.Thread(
                    target=_send_party_info, args=(workers[i], rank, msg, return_values)
                )
                thread.start()
                threads.append(thread)

            # Wait for local party and sender threads
            process.join()
            local_return = queue.get()
            # TODO: check if bad function definition (or other error)
            return_values[0] = local_return
            for thread in threads:
                thread.join()
            if was_initialized:
                crypten.init()

            return return_values

        return wrapper

    return decorator
