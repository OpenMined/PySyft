import torch
import numpy as np
import math

from typing import List
from typing import Union
import weakref

import syft
from syft.exceptions import InvalidTensorForRemoteGet
from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.frameworks.torch.pointers import PointerTensor
from syft.workers import BaseWorker
from syft.frameworks.torch.tensors.interpreters.crt_precision import _moduli_for_fields

from syft.exceptions import PureTorchTensorFoundError

from syft.frameworks.torch.overload_torch import overloaded


def _get_maximum_precision():
    """This function returns the maximum value allowed for precision fractions before the chain decides to use LPT.

    This function can be overridden if the setup requires the use of LargePrecisionTensor from a smaller precision.

    The default value is the size of torch.long

    Returns:
        The maximum value for precision allowed in this setup
    """
    return default_pytorch_maximum_precision()


def default_pytorch_maximum_precision():
    """Dealing with integers > 2**62-1 is not fun with precision tensors.
    """
    return 62


class TorchTensor(AbstractTensor):
    """Add methods to this tensor to have them added to every torch.Tensor object.

    This tensor is simply a more convenient way to add custom functions to
    all Torch tensor types. When you add a function to this tensor, it will
    be added to EVERY native torch tensor type (i.e. torch.Torch) automatically
    by the TorchHook (which is in frameworks/torch/hook.py).

    Note: all methods from AbstractTensor will also be included because this
    tensor extends AbstractTensor. So, if you're looking for a method on
    the native torch tensor API but it's not listed here, you might try
    checking AbstractTensor.
    """

    def has_child(self):
        return hasattr(self, "child")

    def describe(self, description):
        self.description = description
        return self

    def tag(self, *_tags):
        if self.tags is None:
            tags = list()
        else:
            tags = list(self.tags)

        for new_tag in _tags:
            tags.append(new_tag)

        self.tags = set(tags)
        return self

    @property
    def tags(self):
        if self.has_child():
            return self.child.tags
        else:
            if not hasattr(self, "_tags"):
                self._tags = None
            return self._tags

    @tags.setter
    def tags(self, new_tags):
        if self.has_child():
            if new_tags is not None:
                self.child.tags = set(new_tags)
            else:
                self.child.tags = set()
        else:
            self._tags = new_tags

    @property
    def description(self):
        if self.has_child():
            return self.child.description
        else:
            if not hasattr(self, "_description"):
                self._description = None
            return self._description

    @description.setter
    def description(self, new_desc):
        if self.has_child():
            self.child.description = new_desc
        else:
            self._description = new_desc

    @property
    def shape(self):
        if self.is_wrapper:
            return self.child.shape
        else:
            return self.native_shape

    @property
    def data(self):
        if self.is_wrapper:
            return self.child.data
        else:
            return self.native_data

    @property
    def grad(self):
        if self.is_wrapper:
            child_grad = self.child.grad
            if child_grad is None:
                return None
            else:
                if child_grad.is_wrapper:
                    return child_grad
                else:
                    return child_grad.wrap()
        else:
            to_return = self.native_grad

            # good to ensure that the ID stays consistent
            # not 100% this is required but it's at least
            # good practice
            try:
                to_return.id = self.grad_id
            except AttributeError:
                if to_return is not None and hasattr(to_return, "id"):
                    self.grad_id = to_return.id

            return to_return

    @grad.setter
    def grad(self, new_grad):

        # If grad is not a pure torch tensor you need to store the chain in a
        # specific place otherwise it will get deleted
        if new_grad is not None and (
            not isinstance(new_grad, torch.Tensor) or hasattr(new_grad, "child")
        ):
            self.child.grad = new_grad  # .wrap()
        else:
            if self.native_grad is not None:
                with torch.no_grad():
                    self.native_grad = new_grad
            elif new_grad is not None:
                self.native_grad = new_grad
        return self

    def __str__(self) -> str:
        if self.has_child():
            if self.is_wrapper:
                return "(Wrapper)>" + self.child.__str__()
            else:
                return type(self).__name__ + ">" + self.child.__str__()
        else:
            return self.native___str__()

    def __repr__(self) -> str:
        if self.has_child():
            if self.is_wrapper:
                return "(Wrapper)>" + self.child.__str__()
            else:
                return type(self).__name__ + ">" + self.child.__repr__()
        else:
            out = self.native___repr__()

            big_repr = False

            if self.tags is not None and len(self.tags):
                big_repr = True
                out += "\n\tTags: "
                for tag in self.tags:
                    out += str(tag) + " "

            if self.description is not None:
                big_repr = True
                out += "\n\tDescription: " + str(self.description).split("\n")[0] + "..."

            if big_repr:
                out += "\n\tShape: " + str(self.shape)

            return out

    def __eq__(self, other):
        return self.eq(other)

    @property
    def id(self):
        if self.is_wrapper:
            return self.child.id
        else:
            try:
                return self._id
            except AttributeError:
                self._id = syft.ID_PROVIDER.pop()
                return self._id

    @id.setter
    def id(self, new_id):
        if self.is_wrapper:
            self.child.id = new_id
        else:
            self._id = new_id

    def _is_parameter(self):
        """
        Utility method to test if the tensor is in fact a Parameter
        """
        return isinstance(self, torch.nn.Parameter)

    @staticmethod
    @overloaded.module
    def torch(module):
        def roll(tensor, shifts, **kwargs):
            int_shifts = int(shifts.item())
            return torch.native_roll(tensor, int_shifts, **kwargs)

        module.roll = roll

    @classmethod
    def handle_func_command(cls, command):
        """
        Operates as a router for functions. A function call always starts
        by being handled here and 3 scenarii must be considered:

        Real Torch tensor:
            The arguments of the function are real tensors so we should
            run the native torch command

        Torch wrapper:
            The arguments are just wrappers at the top of a chain
            (ex: wrapper>LoggingTensor>Torch tensor), so just forward
            the instruction to the next layer type in the chain (in
            the example above to LoggingTensor.handle_func_command),
            get the response and replace a wrapper on top of all tensors
            found in the response.

        Syft Tensor:
            The arguments are syft tensors of same type: this can happen
            if at any node of the chain where some function is forwarded,
            the handle_func_command modify the function and make a new
            call but keeps the arguments "un-wrapped". Making a new call
            means that by default the command is treated here in the
            global router.

        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """
        cmd, _, args, kwargs = command

        try:  # will work if tensors are wrappers

            # Replace all torch tensor with their child attribute
            # Note that we return also args_type which helps handling case 3 in the docstring
            new_args, new_kwargs, new_type, args_type = syft.frameworks.torch.hook_args.unwrap_args_from_function(
                cmd, args, kwargs, return_args_type=True
            )
            # This handles case 3: it redirects the command to the appropriate class depending
            # of the syft type of the arguments and returns
            if args_type not in (torch.Tensor, torch.nn.Parameter):
                return args_type.handle_func_command(command)

            # build the new command
            new_command = (cmd, None, new_args, new_kwargs)
            # Send it to the appropriate class and get the response
            response = new_type.handle_func_command(new_command)
            # Put back the wrappers where needed
            response = syft.frameworks.torch.hook_args.hook_response(
                cmd, response, wrap_type=args_type
            )
        except PureTorchTensorFoundError:  # means that it's not a wrapper but a pure tensor

            # Check that the function has not been overwritten
            try:
                # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
                command = cls.rgetattr(cls, cmd)
                return command(*args, **kwargs)
            except AttributeError:
                pass

            # TODO: clean this line
            cmd = (
                "syft.local_worker.hook."
                + ".".join(cmd.split(".")[:-1])
                + ".native_"
                + cmd.split(".")[-1]
            )
            # Run the native function with the new args
            # Note the the cmd should already be checked upon reception by the worker
            # in the execute_command function
            if isinstance(args, tuple):
                response = eval(cmd)(*args, **kwargs)
            else:
                response = eval(cmd)(args, **kwargs)

        return response

    def send(
        self,
        *location,
        inplace: bool = False,
        local_autograd=False,
        preinitialize_grad=False,
        no_wrap=False,
        garbage_collect_data=True,
    ):
        """Gets the pointer to a new remote object.

        One of the most commonly used methods in PySyft, this method serializes
        the object upon which it is called (self), sends the object to a remote
        worker, creates a pointer to that worker, and then returns that pointer
        from this function.

        Args:
            location: The BaseWorker object which you want to send this object
                to. Note that this is never actually the BaseWorker but instead
                a class which instantiates the BaseWorker abstraction.
            inplace: if true, return the same object instance, else a new wrapper
            local_autograd: Use autograd system on the local machine instead of PyTorch's
                autograd on the workers.
            preinitialize_grad: Initialize gradient for AutogradTensors to a tensor
            no_wrap: If True, wrap() is called on the created pointer
            garbage_collect_data: argument passed down to create_pointer()

        Returns:
            A torch.Tensor[PointerTensor] pointer to self. Note that this
            object will likely be wrapped by a torch.Tensor wrapper.
        """

        # If you send a pointer p1, you want the pointer to pointer p2 to control
        # the garbage collection and not the remaining old p1 (here self). Because if
        # p2 is not GCed, GCing p1 shouldn't delete the remote tensor, but if you
        # want to do so, as p2 is not GCed, you can still do `del p2`.
        # This allows to chain multiple .send().send() calls.

        if len(location) == 1:

            location = location[0]

            if hasattr(self, "child") and isinstance(self.child, PointerTensor):
                self.child.garbage_collect_data = False
                if self._is_parameter():
                    self.data.child.garbage_collect_data = False

            ptr = self.owner.send(
                self,
                location,
                local_autograd=local_autograd,
                preinitialize_grad=preinitialize_grad,
                garbage_collect_data=garbage_collect_data,
            )

            ptr.description = self.description
            ptr.tags = self.tags

            # The last pointer should control remote GC, not the previous self.ptr
            if hasattr(self, "ptr") and self.ptr is not None:
                ptr_ = self.ptr()
                if ptr_ is not None:
                    ptr_.garbage_collect_data = False

            # we need to cache this weak reference to the pointer so that
            # if this method gets called multiple times we can simply re-use
            # the same pointer which was previously created
            self.ptr = weakref.ref(ptr)

            if self._is_parameter():
                if inplace:
                    with torch.no_grad():
                        self.set_()
                    self.data = ptr
                    output = self
                else:
                    if no_wrap:
                        raise ValueError("Parameters can't accept no_wrap=True")
                    wrapper = torch.Tensor()
                    param_wrapper = torch.nn.Parameter(wrapper)
                    with torch.no_grad():
                        param_wrapper.set_()
                    param_wrapper.data = ptr
                    output = param_wrapper
            else:
                if inplace:
                    self.set_()
                    self.child = ptr
                    return self
                else:
                    output = ptr if no_wrap else ptr.wrap()

            if self.requires_grad:
                # This is for AutogradTensor to work on MultiPointerTensors
                # With pre-initialized gradients, this should get it from AutogradTensor.grad
                if preinitialize_grad:
                    grad = output.child.grad
                else:
                    grad = output.attr("grad")

                output.grad = grad

                # Because of the way PyTorch works, .grad is prone to
                # create entirely new Python objects for the tensor, which
                # inadvertently deletes our custom attributes (like .child)
                # But, if we keep a backup reference around, PyTorch seems
                # to re-use it, which means .grad keeps the attributes we
                # want it to keep. #HackAlert
                output.backup_grad = grad

                if local_autograd:
                    output = syft.AutogradTensor(
                        data=output, preinitialize_grad=preinitialize_grad
                    ).on(output)

        else:

            children = list()
            for loc in location:
                children.append(self.clone().send(loc, no_wrap=True))

            output = syft.MultiPointerTensor(children=children)

            if not no_wrap:
                output = output.wrap()

        return output

    def send_(self, *location):
        """
        Calls send() with inplace option, but only with a single location
        :param location: workers locations
        :return:
        """
        if len(location) > 1:
            raise NotImplementedError("Inplace send to several workers is currently not supported.")

        return self.send(*location, inplace=True)

    def create_pointer(
        self,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
        shape=None,
        local_autograd=False,
        preinitialize_grad=False,
    ) -> PointerTensor:
        """Creates a pointer to the "self" torch.Tensor object.

        Returns:
            A PointerTensor pointer to self. Note that this
            object will likely be wrapped by a torch.Tensor wrapper.
        """
        if id_at_location is None:
            id_at_location = self.id

        if ptr_id is None:
            if location is not None and location.id != self.owner.id:
                ptr_id = self.id
            else:
                ptr_id = syft.ID_PROVIDER.pop()

        if shape is None:
            shape = self.shape

        ptr = syft.PointerTensor.create_pointer(
            self,
            location,
            id_at_location,
            register,
            owner,
            ptr_id,
            garbage_collect_data,
            shape,
            local_autograd,
            preinitialize_grad,
        )

        return ptr

    def mid_get(self):
        """This method calls .get() on a child pointer and correctly registers the results"""
        if not hasattr(self, "child"):
            raise InvalidTensorForRemoteGet(self)

        self.child.mid_get()

    def remote_get(self):
        """Assuming .child is a PointerTensor, this method calls .get() on the tensor
        that the .child is pointing to (which should also be a PointerTensor)

        TODO: make this kind of message forwarding generic?
        """
        if not hasattr(self, "child"):
            raise InvalidTensorForRemoteGet(self)

        self.child.remote_get()

        return self

    def get(self, *args, inplace: bool = False, **kwargs):
        """Requests the tensor/chain being pointed to, be serialized and return
            Args:
                args: args to forward to worker
                inplace: if true, return the same object instance, else a new wrapper
                kwargs: kwargs to forward to worker
            Raises:
                GetNotPermittedError: Raised if get is not permitted on this tensor
        """
        # Transfer the get() to the child attribute which is a pointer

        # if (self.has_child()):
        #     if (isinstance(self.child, syft.frameworks.torch.tensors.FixedPrecisionTensor)):
        #         if (hasattr(self.child, "child")):
        #             if (hasattr(self.child.child, "child")):
        #                 if(isinstance(self.child.child.child, syft.frameworks.torch.tensors.AdditiveSharingTensor)):
        #                     self.child.child =  self.child.child.get()
        #                     return self

        tensor = self.child.get(*args, **kwargs)

        # Clean the wrapper
        delattr(self, "child")

        # Parameters use .data instead of children
        # so we need to have special support to make sure
        # that Parmaeters operate inline (because they're
        # typically being managed inside of a model/optimizer
        # so not using the same wrapper can cause the model/
        # optimizer to lose track of where the actual weights
        # are.
        if isinstance(self, torch.nn.Parameter):
            if inplace:
                self.data = tensor.data
                self.grad = tensor.grad
                return self
            else:
                return tensor

        if inplace:
            self.set_(tensor)
            if hasattr(tensor, "child"):
                self.child = tensor.child
            return self
        else:
            return tensor

    def get_(self, *args, **kwargs):
        """
        Calls get() with inplace option set to True
        """
        return self.get(*args, inplace=True, **kwargs)

    def allowed_to_get(self) -> bool:
        """This function returns true always currently. Will return false in the future
        if get is not allowed to be called on this tensor
        """
        return True

    def move(self, location):
        self.child = self.child.move(location)
        return self

    def attr(self, attr_name):
        """"""

        if self.is_wrapper:
            attr_val = self.child.attr(attr_name)

            if attr_name == "grad":
                self.grad = attr_val
        else:
            attr_val = getattr(self, attr_name)

        return attr_val

    def enc_fix_prec(self):
        return self.child.fix_precision()

    def float_prec(self):
        return self.child.float_precision()

    float_precision = float_prec

    def float_prec_(self):
        tensor = self.float_prec()
        if hasattr(tensor, "child"):
            self.child = tensor.child
        elif self._is_parameter():
            self.data = tensor
        else:
            del self.child
            self.set_(tensor)
        return self

    float_precision_ = float_prec_

    def fix_prec(self, *args, storage="auto", field_type="int100", **kwargs):
        if self.is_wrapper:
            self.child = self.child.fix_prec(*args, **kwargs)
            return self

        base = kwargs.get("base", 10)
        prec_fractional = kwargs.get("precision_fractional", 3)

        max_precision = _get_maximum_precision()
        need_large_prec = self._requires_large_precision(max_precision, base, prec_fractional)

        if storage == "crt":
            assert (
                "field" not in kwargs
            ), 'When storage is set to "crt", choose the field size with the field_type argument'

            possible_field_types = list(_moduli_for_fields.keys())
            assert (
                field_type in possible_field_types
            ), f"Choose field_type in {possible_field_types} to build CRT tensors"

            residues = {}
            for mod in _moduli_for_fields[field_type]:
                residues[mod] = (
                    syft.FixedPrecisionTensor(*args, field=mod, **kwargs)
                    .on(self)
                    .child.fix_precision(check_range=False)
                    .wrap()
                )

            return syft.CRTPrecisionTensor(residues, *args, **kwargs).wrap()

        if need_large_prec or storage == "large":
            return (
                syft.LargePrecisionTensor(*args, **kwargs)
                .on(self)
                .child.fix_large_precision()
                .wrap()
            )
        else:
            assert not need_large_prec, "This tensor needs large precision to be correctly stored"
            return syft.FixedPrecisionTensor(*args, **kwargs).on(self).enc_fix_prec().wrap()

    fix_precision = fix_prec

    def fix_prec_(self, *args, **kwargs):
        tensor = self.fix_prec(*args, **kwargs)
        self.child = tensor.child
        return self

    fix_precision_ = fix_prec_

    def _requires_large_precision(self, max_precision, base, precision_fractional):
        """Check if any of the elements in the tensor would require large precision.
        """
        base_fractional = math.log2(base ** precision_fractional)
        # We need to use NumPy here as log2 is not yet implemented for LongTensor PyTorch objects
        return np.any(
            np.log2(np.abs(self.clone().detach().numpy()) + 1) + base_fractional > max_precision
        )

    def share(
        self,
        *owners: List[BaseWorker],
        field: Union[int, None] = None,
        crypto_provider: Union[BaseWorker, None] = None,
        requires_grad: bool = False,
        no_wrap: bool = False,
    ):
        """This is a pass through method which calls .share on the child.

        Args:
            owners (list): A list of BaseWorker objects determining who to send shares to.
            field (int or None): The arithmetic field where live the shares.
            crypto_provider (BaseWorker or None): The worker providing the crypto primitives.
            requires_grad (bool): Should we add AutogradTensor to allow gradient computation,
                default is False.
        """

        shared_tensor = self
        if self.has_child():
            self.child = self.child.share(*owners, field=field, crypto_provider=crypto_provider)
            if no_wrap:
                return self.child
        else:
            shared_tensor = (
                syft.AdditiveSharingTensor(
                    field=field, crypto_provider=crypto_provider, owner=self.owner
                )
                .on(self)
                .child.init_shares(*owners)
            )
            if not no_wrap:
                shared_tensor = shared_tensor.wrap()

        if requires_grad:
            shared_tensor = syft.AutogradTensor().on(shared_tensor)

        return shared_tensor

    def share_(self, *args, **kwargs):
        """
        Allows to call .share() as an inplace operation
        """
        tensor = self.share(*args, **kwargs)
        self.child = tensor.child
        return self

    def combine(self, *pointers):
        """This method will combine the child pointer with another list of pointers

        Args:
            *pointers a list of pointers to be combined into a MultiPointerTensor

        """

        assert isinstance(self.child, PointerTensor)

        ps = list(pointers)
        ps.append(self)

        return syft.combine_pointers(*ps)
