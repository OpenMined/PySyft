import functools
import multiprocessing
import threading
import os

import syft as sy
from syft.messaging.message import CryptenInit

import crypten
from crypten.communicator import DistributedCommunicator


def _launch(func, rank, world_size, master_addr, master_port, queue, func_args, func_kwargs):
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

    crypten.init()
    return_value = func(*func_args, **func_kwargs).tolist()
    crypten.uninit()

    queue.put(return_value)


def _new_party(func, rank, world_size, master_addr, master_port, func_args, func_kwargs):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_launch,
        args=(func, rank, world_size, master_addr, master_port, queue, func_args, func_kwargs),
    )
    return process, queue


def run_party(func, rank, world_size, master_addr, master_port, func_args, func_kwargs):
    """Start crypten party localy and run computation.

    Args:
        func (function): computation to be done.
        rank (int): rank of the crypten party.
        world_size (int): number of crypten parties involved in the computation.
        master_addr (str): IP address of the master party (party with rank 0).
        master_port (int or str): port of the master party (party with rank 0).
        func_args (list): arguments to be passed to func.
        func_kwargs (dict): keyword arguments to be passed to func.

    Returns:
        The return value of func.
    """

    process, queue = _new_party(
        func, rank, world_size, master_addr, master_port, func_args, func_kwargs
    )
    was_initialized = DistributedCommunicator.is_initialized()
    if was_initialized:
        crypten.uninit()
    process.start()
    process.join()
    if was_initialized:
        crypten.init()
    res = queue.get()
    return res


def _send_party_info(worker, rank, msg, return_values):
    """Send message to worker with necessary information to run a crypten party.
    Add response to return_values dictionary.

    Args:
        worker (BaseWorker): worker to send the message to.
        rank (int): rank of the crypten party.
        msg (CryptenInitMessage): message containing the rank, world_size, master_addr and master_port.
        return_values (dict): dictionnary holding return values of workers.
    """

    response = worker.send_msg(msg, worker)
    return_values[rank] = response.object


def run_multiworkers(workers: list, master_addr: str, master_port: int = 15463):
    """Defines decorator to run function across multiple workers.

    Args:
        workers (list): workers (parties) to be involved in the computation.
        master_addr (str): IP address of the master party (party with rank 0).
        master_port (int, str): port of the master party (party with rank 0), default is 15987.
    """

    def decorator(plan):
        @functools.wraps(plan)
        def wrapper(*args, **kwargs):
            # TODO:
            # - check if workers are reachable / they can handle the computation
            # - check return code of processes for possible failure

            world_size = len(workers) + 1
            return_values = {rank: None for rank in range(world_size)}

            crypten.init()
            plan.build()
            crypten.uninit()

            # Mark the plan so the other workers will use that tag to retrieve the plan
            plan.tags = ["crypten_plan"]

            rank_to_worker_id = dict(
                zip(range(1, len(workers) + 1), [worker.id for worker in workers])
            )

            sy.local_worker._set_rank_to_worker_id(rank_to_worker_id)

            for worker in workers:
                plan.send(worker)

            # Start local party
            process, queue = _new_party(plan, 0, world_size, master_addr, master_port, (), {})

            was_initialized = DistributedCommunicator.is_initialized()
            if was_initialized:
                crypten.uninit()
            process.start()

            # Run TTP if required
            # TODO: run ttp in a specified worker
            if crypten.mpc.ttp_required():
                ttp_process, _ = _new_party(
                    crypten.mpc.provider.TTPServer,
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
                msg = CryptenInit((rank_to_worker_id, world_size, master_addr, master_port))
                thread = threading.Thread(
                    target=_send_party_info, args=(workers[i], rank, msg, return_values)
                )
                thread.start()
                threads.append(thread)

            # Wait for local party and sender threads
            process.join()
            return_values[0] = queue.get()
            for thread in threads:
                thread.join()
            if was_initialized:
                crypten.init()

            return return_values

        return wrapper

    return decorator
