#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pickle

import numpy
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from .communicator import Communicator, _logging


class DistributedCommunicator(Communicator):
    """
    Implementation of the Communicator class via torch.distributed. Use this
    communicator to communicate between different processes, potentially,
    running on different nodes.
    """

    BYTES_PER_ELEMENT = 8
    __instance = None

    def __init__(self, init_ttp=False, store = None):
        # no need to do anything if we already initialized the communicator:
        if not dist.is_initialized():
            # get configuration variables from environmens:
            for key in ["distributed_backend", "rendezvous", "world_size", "rank"]:
                if key.upper() not in os.environ:
                    raise ValueError("Environment variable %s must be set." % key)
                setattr(self, key.lower(), os.environ[key.upper()])

            # make sure world size and rank are integers; comms stats are reset:
            self.world_size = int(self.world_size)
            self.rank = int(self.rank)
            self.reset_communication_stats()

            # logging:
            logging.info("==================")
            logging.info("DistributedCommunicator with rank %d" % self.rank)
            logging.info("==================")

            # initialize process group:
            total_ws = self.world_size + 1 if init_ttp else self.world_size
            if store is None:
                dist.init_process_group(
                    backend=self.distributed_backend,
                    init_method=self.rendezvous,
                    world_size=total_ws,
                    rank=self.rank,
                )
            else:
                self.store = store
                dist.init_process_group(
                    backend=self.distributed_backend,
                    store=store,
                    world_size=total_ws,
                    rank=self.rank,
                )
            self.ttp_group = dist.new_group([i for i in range(total_ws)])
            self.main_group = dist.new_group([i for i in range(self.world_size)])
            self.ttp_initialized = init_ttp
            logging.info("World size = %d" % self.world_size)

    @classmethod
    def is_initialized(cls):
        return dist.is_initialized()

    @classmethod
    def initialize(cls, rank, world_size, init_ttp=False):
        import os

        if os.name == "nt":
            raise OSError(
                "Multiprocessing is not supported on Windows. "
                + "Please initialize CrypTen via crypten.init_thread() instead."
            )

        # set default arguments for communicator:
        default_args = {
            "DISTRIBUTED_BACKEND": "gloo",
            "RENDEZVOUS": "file:///tmp/sharedfile",
            "WORLD_SIZE": world_size,
            "RANK": rank,
        }
        for key, val in default_args.items():
            if key not in os.environ:
                os.environ[key] = str(val)

        store = dist.TCPStore(
            os.getenv("MASTER_IP"),
            int(os.getenv("MASTER_PORT")),
            rank,
            rank == 0
        )
        cls.instance = DistributedCommunicator(init_ttp=init_ttp, store=store)

    @classmethod
    def get(cls):
        return cls.instance

    @classmethod
    def shutdown(cls):
        if dist.get_rank() == 0 and cls.instance.ttp_initialized:
            cls.instance.send_obj(
                "terminate", cls.instance.get_ttp_rank(), cls.instance.ttp_group
            )
        dist.destroy_process_group(cls.instance.main_group)
        dist.destroy_process_group(cls.instance.ttp_group)
        dist.destroy_process_group()
        cls.instance = None

    @_logging
    def send(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.send(tensor, dst, group=self.main_group)

    @_logging
    def recv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = tensor.clone()
        dist.recv(result, src=src, group=self.main_group)
        return result

    @_logging
    def isend(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.isend(tensor, dst, group=self.main_group)

    @_logging
    def irecv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.irecv(tensor, src=src, group=self.main_group)

    @_logging
    def scatter(self, scatter_list, src, size=None):
        """Scatters a list of tensors to all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        if src != self.get_rank():
            if size is None:
                size = scatter_list[self.get_rank()].size()
            tensor = torch.empty(size=size, dtype=torch.long)
            dist.scatter(tensor, [], src, group=self.main_group)
        else:
            tensor = scatter_list[self.get_rank()]
            dist.scatter(tensor, [t for t in scatter_list], src, group=self.main_group)
        return tensor

    @_logging
    def reduce(self, tensor, dst, op=ReduceOp.SUM):
        """Reduces the tensor data across all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = tensor.clone()
        dist.reduce(result, dst, op=op, group=self.main_group)
        return result if dst == self.get_rank() else None

    @_logging
    def all_reduce(self, tensor, op=ReduceOp.SUM):
        """Reduces the tensor data across all parties; all get the final result."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = tensor.clone()
        dist.all_reduce(result, op=op, group=self.main_group)
        return result

    @_logging
    def gather(self, tensor, dst):
        """Gathers a list of tensors in a single party."""
        assert dist.is_initialized(), "initialize the communicator first"
        if self.get_rank() == dst:
            result = []
            for _ in range(self.get_world_size()):
                result.append(torch.empty(size=tensor.size(), dtype=torch.long))
            dist.gather(tensor, result, dst, group=self.main_group)
            return result
        dist.gather(tensor, [], dst, group=self.main_group)
        return [None]

    @_logging
    def all_gather(self, tensor):
        """Gathers tensors from all parties in a list."""
        assert dist.is_initialized(), "initialize the communicator first"
        result = []
        for _ in range(self.get_world_size()):
            result.append(torch.empty(size=tensor.size(), dtype=torch.long))
        dist.all_gather(result, tensor, group=self.main_group)
        return result

    @_logging
    def broadcast(self, tensor, src):
        """Broadcasts the tensor to all parties."""
        assert dist.is_initialized(), "initialize the communicator first"
        dist.broadcast(tensor, src, group=self.main_group)
        return tensor

    @_logging
    def barrier(self):
        """Synchronizes all processes.

        This collective blocks processes until the whole group enters this
        function.
        """
        assert dist.is_initialized(), "initialize the communicator first"
        dist.barrier(group=self.main_group)

    @_logging
    def send_obj(self, obj, dst, group=None):
        if group is None:
            group = self.main_group

        buf = pickle.dumps(obj)
        size = torch.tensor(len(buf), dtype=torch.int32)
        arr = torch.from_numpy(numpy.frombuffer(buf, dtype=numpy.int8))

        r0 = dist.isend(size, dst=dst, group=group)
        r1 = dist.isend(arr, dst=dst, group=group)

        r0.wait()
        r1.wait()

    @_logging
    def recv_obj(self, src, group=None):
        if group is None:
            group = self.main_group

        size = torch.tensor(1, dtype=torch.int32)
        dist.irecv(size, src=src, group=group).wait()

        data = torch.zeros(size=(size,), dtype=torch.int8)
        dist.irecv(data, src=src, group=group).wait()
        buf = data.numpy().tobytes()
        return pickle.loads(buf)

    @_logging
    def oblivious_transfer(self, value, sender, receiver):
        """Performs a 1-out-of-2 Oblivious Transfer (OT) between the
        `sender` and `receiver` parties.
        """
        rank = dist.rank()
        if rank != sender and rank != receiver:
            return None

        # TODO: Make this transfer oblivious by implementing OT protocol
        # WARNING: This is currently insecure since transfers are not oblivious.
        if rank == sender:
            assert isinstance(value, list), "OT sender must input a list of values"

            # Receive index from receiver
            index = self.recv_obj(receiver)
            result = torch.where(index, value[0], value[1])
            self.send_obj(result, receiver)
            return None
        else:
            self.send_obj(value, sender)
            result = self.recv_obj(sender)
            return result

    def get_world_size(self):
        """Returns the size of the world."""
        assert dist.is_initialized(), "initialize the communicator first"
        return self.world_size

    def get_rank(self):
        """Returns the rank of the current process."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.get_rank()

    def get_ttp_rank(self):
        """Returns the rank of the Trusted Third Party"""
        return self.get_world_size()

    def get_distributed_backend(self):
        """Returns name of torch.distributed backend used."""
        assert dist.is_initialized(), "initialize the communicator first"
        return dist.get_backend()
