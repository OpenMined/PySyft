import functools
import multiprocessing
import threading
import os
import re
import ast
from dill.source import getsource
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    guarded_setattr,
)
from RestrictedPython.PrintCollector import PrintCollector

import torch
import crypten
import syft as sy
from syft.messaging.message import CryptenInit
from syft.frameworks import crypten as syft_crypt
from crypten.communicator import DistributedCommunicator


PACK_OTHER = -1
PACK_TORCH_TENSOR = 0
PACK_CRYPTEN_MODEL = 1


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

    # Provide it in global for replicating tutorial 7 of crypten.
    # The parties need to know about the classes so here we define them statically.
    class ExampleNet(torch.nn.Module):
        def __init__(self):
            super(ExampleNet, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, padding=0)
            self.fc1 = torch.nn.Linear(16 * 12 * 12, 100)
            self.fc2 = torch.nn.Linear(
                100, 2
            )  # For binary classification, final layer needs only 2 outputs

        def forward(self, x):
            out = self.conv1(x)
            out = torch.nn.functional.relu(out)
            out = torch.nn.functional.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = torch.nn.functional.relu(out)
            out = self.fc2(out)
            return out

    # Update load function
    setattr(crypten, "load", syft_crypt.load)

    # TODO: error handling
    exec_globals = {"__builtins__": safe_builtins}
    exec_globals["crypten"] = crypten
    exec_globals["torch"] = torch
    exec_globals["syft"] = sy
    exec_globals["ExampleNet"] = ExampleNet
    exec_globals["_getiter_"] = default_guarded_getiter
    exec_globals["_getitem_"] = default_guarded_getitem
    exec_globals["_getattr_"] = getattr
    # for a, b in
    exec_globals["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    # for a in
    exec_globals["_unpack_sequence_"] = guarded_unpack_sequence
    # unrestricted write of attr
    exec_globals["_write_"] = lambda x: x
    # Collecting printed strings and saved in printed local variable
    exec_globals["_print_"] = PrintCollector
    exec_globals["__name__"] = "__main__"

    exec_locals = {}
    compiled = compile_restricted(func_src)
    exec(compiled, exec_globals, exec_locals)  # nosec
    func = exec_locals[func_name]

    crypten.init()
    print(f"Starting func at {rank}")
    return_value = func(*func_args, **func_kwargs)
    print(f"Exited func at {rank}")
    crypten.uninit()

    print(f"Packing value in _launch with {rank}")
    return_value = _pack_values(return_value)
    print(f"Queuing in _launch with {rank}")
    queue.put(return_value)
    print(f"Queued in _launch with {rank}")


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
    print(f"Returing from run_party with {rank}")
    return queue.get()


def _send_party_info(worker, rank, msg, return_values, model=None):
    """Send message to worker with necessary information to run a crypten party.
    Add response to return_values dictionary.

    Args:
        worker (BaseWorker): worker to send the message to.
        rank (int): rank of the crypten party.
        msg (CryptenInit): message containing the rank, world_size, master_addr and master_port.
        return_values (dict): dictionnary holding return values of workers.
    """
    print(f"Sending info to {rank}")
    response = worker.send_msg(msg, worker)
    print(f"Got response from {rank}")
    return_values[rank] = _unpack_values(response.contents, model)
    print(f"Unpacked from {rank}")


def _pack_values(values):
    """Pack return values to be passed into a queue then sent over the wire.
    The main goal here is to be able to return torch tensors."""

    packed_values = []
    # single value
    if not isinstance(values, tuple):
        packed_values.append(_pack_value(values))
    # multiple values
    else:
        for value in values:
            packed_values.append(_pack_value(value))
    return packed_values


def _pack_value(value):
    if isinstance(value, torch.Tensor):
        return (PACK_TORCH_TENSOR, value.tolist())

    elif isinstance(value, crypten.nn.Module):
        if value.encrypted:
            raise TypeError("Cannot pack an encrypted crypten model.")
        params = []
        for p in value.parameters():
            params.append(p.tolist())

        return (PACK_CRYPTEN_MODEL, params)

    return (PACK_OTHER, value)


def _unpack_values(values, model=None):
    """Unpack return values that are fetched from the queue."""

    unpacked_values = []
    for value in values:
        unpacked_values.append(_unpack_value(value, model))
    # single value
    if len(unpacked_values) == 1:
        return unpacked_values[0]
    # multiple values
    else:
        return tuple(unpacked_values)


def _unpack_value(value, model=None):
    value_type = value[0]
    if value_type == PACK_OTHER:
        return value[1]
    elif value_type == PACK_TORCH_TENSOR:
        return torch.tensor(value[1])
    elif value_type == PACK_CRYPTEN_MODEL:
        if model is None:
            raise TypeError("model can't be None when value is a crypten model.")
        params = value[1]
        for p, p_val in zip(model.parameters(), params):
            # Can't set value for leaf variable that requires grad
            requires_grad = p.requires_grad
            p.requires_grad = False
            p.set_(torch.tensor(p_val))
            p.requires_grad = requires_grad

        return syft_crypt.crypten_to_syft_model(model)


def toy_func():
    alice_tensor = syft_crypt.load("crypten_data", 1, "alice")
    bob_tensor = syft_crypt.load("crypten_data", 2, "bob")

    crypt = crypten.cat([alice_tensor, bob_tensor], dim=0)
    return crypt.get_plain_text().tolist()


def _get_model():
    class ExampleNet(torch.nn.Module):
        def __init__(self):
            super(ExampleNet, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, padding=0)
            self.fc1 = torch.nn.Linear(16 * 12 * 12, 100)
            self.fc2 = torch.nn.Linear(
                100, 2
            )  # For binary classification, final layer needs only 2 outputs

        def forward(self, x):
            out = self.conv1(x)
            out = torch.nn.functional.relu(out)
            out = torch.nn.functional.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = torch.nn.functional.relu(out)
            out = self.fc2(out)
            return out

    dummy_input = torch.empty(1, 1, 28, 28)
    model = crypten.nn.from_pytorch(ExampleNet(), dummy_input)
    return model


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
            re_decorator = r"@[^\(]+\([^\)]*\)"
            func_src = re.sub(re_decorator, "", getsource(func))

            # Start local party
            process, queue = _new_party(func_src, 0, world_size, master_addr, master_port, (), {})
            was_initialized = DistributedCommunicator.is_initialized()
            if was_initialized:
                crypten.uninit()
            process.start()
            
            # TODO: can't do this before starting the local process ! Even outside the func (weird bug)
            model = _get_model()
            
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
                    target=_send_party_info, args=(workers[i], rank, msg, return_values, model)
                )
                thread.start()
                threads.append(thread)

            # Wait for local party and sender threads
            print("Waiting for local process")
            # TODO: joining the process hangs even when the process's function ends !
            # First guess is because we didn't get from the queue
            # process.join()
            print("Exited local process")
            local_return = _unpack_values(queue.get(), model)
            # TODO: check if bad function definition (or other error)
            return_values[0] = local_return
            for thread in threads:
                thread.join()
            if was_initialized:
                crypten.init()

            return return_values

        return wrapper

    return decorator
