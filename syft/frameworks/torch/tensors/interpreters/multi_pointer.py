import torch
from typing import List
from typing import Union

import syft as sy
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor
from syft.workers import BaseWorker
from syft.frameworks.torch.overload_torch import overloaded


class MultiPointerTensor(AbstractTensor):
    ""

    def __init__(
        self,
        location: BaseWorker = None,
        id_at_location: Union[str, int] = None,
        register: bool = False,
        owner: BaseWorker = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        point_to_attr: str = None,
        tags: List[str] = None,
        description: str = None,
        children: List[AbstractTensor] = [],
    ):

        super().__init__(tags, description)

        self.location = location
        self.id_at_location = id_at_location
        self.owner = owner
        self.id = id
        self.garbage_collect_data = garbage_collect_data
        self.point_to_attr = point_to_attr

        self.child = {}
        for c in children:
            assert c.shape == children[0].shape
            self.child[c.location.id] = c

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        for v in self.child.values():
            out += "\n\t-> " + str(v)
        return out

    def __eq__(self, other):
        return torch.eq(self, other)

    def __add__(self, other):
        """
        Adding a MultiPointer (MPT) and an AdditiveShared Tensor (AST) should return an
        AdditiveShared Tensor, so if we have this configuration, we permute self and
        other to use the fact that other.__add__(...) return an object of type other

        Else, we just redirect to .add which works well
        """
        if isinstance(other, AdditiveSharingTensor):
            return other.__add__(self)
        else:
            return self.add(other)

    def __mul__(self, other):
        """
        See __add__ for details but, MPT * AST should return AST
        """
        if isinstance(other, AdditiveSharingTensor):
            return other.__mul__(self)
        else:
            return self.mul(other)

    @property
    def shape(self) -> torch.Size:
        """This method returns the shape of the data being pointed to.
        This shape information SHOULD be cached on self._shape, but
        occasionally this information may not be present. If this is the
        case, then it requests the shape information from the remote object
        directly (which is inefficient and should be avoided)."""

        return list(self.child.values())[0].shape

    def get(self, sum_results: bool = False) -> torch.Tensor:

        results = list()
        for v in self.child.values():
            results.append(v.get())

        if sum_results:
            return sum(results)

        return results

    def virtual_get(self, sum_results: bool = False):
        """Get the value of the tensor without calling get - Only for VirtualWorkers"""

        results = list()
        for v in self.child.values():
            value = v.location._objects[v.id_at_location]
            results.append(value)

        if sum_results:
            return sum(results)

        return results

    @staticmethod
    def dispatch(args, worker):
        """
        utility function for handle_func_command which help to select
        shares (seen as elements of dict) in an argument set. It could
        perhaps be put elsewhere

        Args:
            args: arguments to give to a functions
            worker: owner of the shares to select

        Return:
            args where the MultiPointerTensor are replaced by
            the appropriate share
        """
        return map(lambda x: x[worker] if isinstance(x, dict) else x, args)

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a Syft Tensor,
        Replace in the args all the LogTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a Syft Tensor on top of all tensors found in
        the response.

        Args:
            command: instruction of a function command: (command name,
            <no self>, arguments[, kwargs])

        Returns:
            the response of the function command
        """

        cmd, _, args, kwargs = command

        tensor = args[0]

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args, **kwargs)
        except AttributeError:
            pass

        # TODO: I can't manage the import issue, can you?
        # Replace all LoggingTensor with their child attribute
        new_args, new_kwargs, new_type = sy.frameworks.torch.hook_args.hook_function_args(
            cmd, args, kwargs
        )

        results = {}
        for worker, share in new_args[0].items():
            new_type = type(share)
            new_args_worker = tuple(MultiPointerTensor.dispatch(new_args, worker))

            # build the new command
            new_command = (cmd, None, new_args_worker, new_kwargs)

            # Send it to the appropriate class and get the response
            results[worker] = new_type.handle_func_command(new_command)

        # Put back MultiPointerTensor on the tensors found in the response
        response = sy.frameworks.torch.hook_args.hook_response(
            cmd, results, wrap_type=cls, wrap_args=tensor.get_class_attributes()
        )

        return response
