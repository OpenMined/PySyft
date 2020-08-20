import functools
import multiprocessing
import os

import torch as th
import syft as sy
from syft.messaging.message import CryptenInitPlan, CryptenInitJail
from syft.frameworks.crypten import jail, utils

import crypten
from syft.frameworks.crypten.hook.hook import hook_plan_building, unhook_plan_building
from crypten.communicator import DistributedCommunicator


def _launch(
    cid, func, rank, world_size, master_addr, master_port, queue, func_args, func_kwargs
):  # pragma: no cover

    # set CrypTen computation id
    sy.frameworks.crypten.CID = cid

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
    return_value = func(*func_args, **func_kwargs)
    crypten.uninit()

    return_value = utils.pack_values(return_value)
    queue.put(return_value)


def _new_party(cid, func, rank, world_size, master_addr, master_port, func_args, func_kwargs):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_launch,
        args=(cid, func, rank, world_size, master_addr, master_port, queue, func_args, func_kwargs),
    )
    return process, queue


def run_party(cid, func, rank, world_size, master_addr, master_port, func_args, func_kwargs):
    """Start crypten party localy and run computation.
    Args:
        cid (int): CrypTen computation id.
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
        cid, func, rank, world_size, master_addr, master_port, func_args, func_kwargs
    )
    was_initialized = DistributedCommunicator.is_initialized()
    if was_initialized:
        crypten.uninit()
    process.start()
    # wait for response
    res = queue.get()
    if was_initialized:
        crypten.init()
    return res


def _send_party_info(worker, rank, msg, return_values, model=None):  # pragma: no cover
    """Send message to worker with necessary information to run a crypten party.
    Add response to return_values dictionary.
    Args:
        worker (BaseWorker): worker to send the message to.
        rank (int): rank of the crypten party.
        msg (CryptenInitMessage): message containing the rank, world_size, master_addr and
            master_port.
        return_values (dict): dictionnary holding return values of workers.
        model: crypten model to unpack parameters to (if received).
    """

    response = worker.send_msg(msg, worker)
    return_values[rank] = utils.unpack_values(response.object, model)


def run_multiworkers(
    workers: list, master_addr: str, master_port: int = 15463, model=None, dummy_input=None
):
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

            if len(workers) != len(set(worker.id for worker in workers)):  # noqa: C401
                raise RuntimeError("found workers with same ID but IDs must be unique")

            if model is not None:
                if not isinstance(model, th.nn.Module):
                    raise TypeError("model must be a torch.nn.Module")
                if dummy_input is None:
                    raise ValueError("must provide dummy_input when model is set")
                if not isinstance(dummy_input, th.Tensor):
                    raise TypeError("dummy_input must be a torch.Tensor")
                onnx_model = utils.pytorch_to_onnx(model, dummy_input)
            else:
                onnx_model = None

            crypten_model = None if onnx_model is None else utils.onnx_to_crypten(onnx_model)

            world_size = len(workers)
            manager = multiprocessing.Manager()
            return_values = manager.dict({rank: None for rank in range(world_size)})

            rank_to_worker_id = dict(zip(range(0, len(workers)), [worker.id for worker in workers]))

            # TODO: run ttp in a specified worker
            # if crypten.mpc.ttp_required():
            #     ttp_process, _ = _new_party(
            #         crypten.mpc.provider.TTPServer,
            #         world_size,
            #         world_size,
            #         master_addr,
            #         master_port,
            #         (),
            #         {},
            #     )
            #     ttp_process.start()

            if isinstance(func, sy.Plan):
                plan = func

                # This is needed because at building we use a set of methods defined in syft
                # (ex: load)
                hook_plan_building()
                was_initialized = DistributedCommunicator.is_initialized()
                if not was_initialized:
                    crypten.init()

                # We can build the plan only using a crypten model such that the actions
                # traced inside the plan would know about it's existance
                if crypten_model is None:
                    plan.build()
                else:
                    plan.build(crypten_model)

                if not was_initialized:
                    crypten.uninit()
                unhook_plan_building()

                # Mark the plan so the other workers will use that tag to retrieve the plan
                plan.tags = ["crypten_plan"]

                for worker in workers:
                    plan.send(worker)

                msg = CryptenInitPlan(
                    (rank_to_worker_id, world_size, master_addr, master_port), onnx_model
                )

            else:  # func
                jail_runner = jail.JailRunner(func=func)
                ser_jail_runner = jail.JailRunner.simplify(jail_runner)

                msg = CryptenInitJail(
                    (rank_to_worker_id, world_size, master_addr, master_port),
                    ser_jail_runner,
                    onnx_model,
                )

            # Send messages to other workers so they start their parties
            threads = []
            for i in range(len(workers)):
                rank = i
                thread = multiprocessing.Process(
                    target=_send_party_info,
                    args=(workers[i], rank, msg, return_values, crypten_model),
                )
                thread.start()
                threads.append(thread)

            # wait for workers running the parties return a response
            for thread in threads:
                thread.join()

            return return_values

        return wrapper

    return decorator
