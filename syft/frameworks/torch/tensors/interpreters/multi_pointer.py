import torch
from typing import List
from typing import Union

import syft as sy
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.workers import BaseWorker
from syft.frameworks.torch.overload_torch import overloaded

from syft.workers import AbstractWorker


class MultiPointerTensor(AbstractTensor):
    """
    The MultiPointerTensor gathers together several pointers to different locations which can
    be operated all at once. The pointers can be referencing the same value or a different one
    depending on the usage.

    The MultiPointerTensor has the same structure that the AdditiveSharedTensor: its child
    attribute is a dictionary {worker.id: Pointer}

    MultiPointerTensor can be directly instantiated using x.send(worker1, worker2, etc) where
    x is a syft or torch tensor. In that case, the value of x will be sent and replicated to
    each workers.
    """

    def __init__(
        self,
        owner: BaseWorker = None,
        id: Union[str, int] = None,
        tags: List[str] = None,
        description: str = None,
        children: List[AbstractTensor] = [],
    ):
        """Initializes an Multi Pointer Tensor, whose behaviour is to keep references
        to several pointers and to distribute computations to all of them in an easy
        way.

        Args:
            owner: an optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the MultiPointerTensor.
            tags: an optional set of hashtags corresponding to this tensor
                which this tensor should be searchable for
            description: an optional string describing the purpose of the
                tensor
            children: an optional list of children which are PointerTensors
        """

        super().__init__(tags, description)

        self.owner = owner
        self.id = id

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

    @property
    @overloaded.method
    def grad(self, self_):
        results = {}
        all_none = True
        for worker, pointer in self_.items():
            pointer_grad = pointer.grad

            if pointer_grad is None:
                results[worker] = None
            else:
                results[worker] = pointer_grad
                all_none = False

        return results if not all_none else None

    def __eq__(self, other):
        return torch.eq(self, other)

    def __add__(self, other):
        """
        Adding a MultiPointer (MPT) and an AdditiveShared Tensor (AST) should return an
        AdditiveShared Tensor, so if we have this configuration, we permute self and
        other to use the fact that other.__add__(...) return an object of type other

        Else, we just redirect to .add which works well
        """
        if isinstance(other, sy.AdditiveSharingTensor):
            return other.__add__(self)
        else:
            return self.add(other)

    def __mul__(self, other):
        """
        See __add__ for details but, MPT * AST should return AST
        """
        if isinstance(other, sy.AdditiveSharingTensor):
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

    def dim(self) -> int:
        """This method fixes the error that the result of dim was a list of ints
        stored inside a multipointer tensor"""
        return len(self.shape)

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
        new_args, new_kwargs, new_type = sy.frameworks.torch.hook_args.unwrap_args_from_function(
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

    def set_garbage_collect_data(self, value):
        shares = self.child
        for _, share in shares.items():
            share.child.garbage_collect_data = value

    @staticmethod
    def simplify(tensor: "MultiPointerTensor") -> tuple:
        """
        This function takes the attributes of a MultiPointerTensor and saves them in a tuple
        Args:
            tensor (MultiPointerTensor): a MultiPointerTensor
        Returns:
            tuple: a tuple holding the unique attributes of the additive shared tensor
        Examples:
            data = simplify(tensor)
        """

        chain = None
        if hasattr(tensor, "child"):
            chain = sy.serde._simplify(tensor.child)
        return (tensor.id, chain)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "MultiPointerTensor":
        """
        This function reconstructs a MultiPointerTensor given it's attributes in form of a tuple.
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the MultiPointerTensor
        Returns:
            MultiPointerTensor: a MultiPointerTensor
        Examples:
            multi_pointer_tensor = detail(data)
        """

        tensor_id, chain = tensor_tuple

        tensor = sy.MultiPointerTensor(owner=worker, id=tensor_id)

        if chain is not None:
            chain = sy.serde._detail(worker, chain)
            tensor.child = chain

        return tensor
