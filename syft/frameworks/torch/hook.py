import inspect
import re
import logging
import types
import syft
from ... import workers

from ...workers import BaseWorker
from .tensors import TorchTensor
from .torch_attributes import TorchAttributes
from .tensors.abstract import initialize_tensor


class TorchHook:
    """A Hook which Overrides Methods on PyTorch Tensors.

    The purpose of this class is to:
        * extend torch methods to allow for the moving of tensors from one
        worker to another.
        * override torch methods to execute commands on one worker that are
        called on tensors controlled by the local worker.

    This class is typically the first thing you will initialize when using
    PySyft with PyTorch because it is responsible for augmenting PyTorch with
    PySyft's added functionality (such as remote execution).

    Args:
        local_worker (workers.BaseWorker): Optional, you can optionally provide
            a local worker as a parameter which TorchHook will assume to be the
            worker owned by the local machine. If you leave it empty,
            TorchClient will automatically initialize a
            :class:`.workers.VirtualWorker` under the assumption you're looking
            to do local experimentation or development.
        is_client (bool): Optional, defaults to True. Whether or not the
            TorchHook is being initialized as an end-user client. This can
            impact whether or not variables are deleted when they fall out of
            scope. If you set this incorrectly on a end user client, Tensors
            and Variables will never be deleted. If you set this incorrectly
            on a remote machine (not a client), tensors will not get saved.
            It's really only important if you're not initializing the local
            worker yourself.
        verbose (bool): Optional, defaults to True. Whether or not to print
            operations as they occur.
        queue_size (int): optional, defaults to 0. The max length of the list
            storing messages to be sent.

    Example:
        >>> import syft as sy
        >>> hook = sy.TorchHook()
        Hooking into Torch...
        Overloading Complete.
        >>> x = sy.Tensor([-2,-1,0,1,2,3])
        >>> x
        -2
        -1
        0
        1
        2
        3
        [syft.core.frameworks.torch.tensor.FloatTensor of size 6]
    """

    def __init__(
        self, torch, local_worker: BaseWorker = None, is_client: bool = True, verbose: bool = True
    ):
        """Initializes the hook.

        Initialize the hook and define all the attributes pertaining to the
        torch hook in a special TorchAttibute class, that will be added in the
        syft.torch attributes. Hence, this parameters are now conveyed by the
        syft module.
        """
        # Save the provided torch module as an attribute of the hook
        self.torch = torch

        # Save the local worker as an attribute
        self.local_worker = local_worker

        if hasattr(torch, "torch_hooked"):
            logging.warning("Torch was already hooked... skipping hooking process")
            self.local_worker = syft.local_worker
            return
        else:
            torch.torch_hooked = True

        # Add all the torch attributes in the syft.torch attr
        syft.torch = TorchAttributes(torch, self)

        if self.local_worker is None:
            # Every TorchHook instance should have a local worker which is
            # responsible for interfacing with other workers. The worker
            # interface is what allows the Torch specific code in TorchHook to
            # be agnostic to the means by which workers communicate (such as
            # peer-to-peer, sockets, through local ports, or all within the
            # same process)
            self.local_worker = workers.VirtualWorker(
                hook=self, is_client_worker=is_client, id="me"
            )
        else:
            self.local_worker.hook = self

        self.to_auto_overload = {}

        self._hook_native_tensor(torch.Tensor, TorchTensor)

        # Add the local_worker to syft so that it can be found if the hook is
        # called several times
        syft.local_worker = self.local_worker

    def _hook_native_tensor(self, tensor_type: type, syft_type: type):
        """Adds PySyft Tensor Functionality to the given native tensor type.

        Overloads the given native Torch tensor to add PySyft Tensor
        Functionality. Overloading involves modifying the tensor type with
        PySyft's added functionality. You may read about what kind of
        modifications are made in the methods that this method calls.

        Args:
            tensor_type (type): The type of tensor being hooked (in this
                refactor this is only ever torch.Tensor, but in previous
                versions of PySyft this iterated over all tensor types.
            syft_type (type): The abstract type whose methods should all be
                added to the tensor_type class. In practice this is only ever
                TorchTensor. Read more about it there.
        """
        # Reinitialize init method of Torch tensor with Syft init
        self._add_registration_to___init__(tensor_type, torch_tensor=True)

        # Overload Torch tensor properties with Syft properties
        self._hook_properties(tensor_type)

        # Returns a list of methods to be overloaded, stored in the dict to_auto_overload
        # with tensor_type as a key
        self.to_auto_overload[tensor_type] = self._which_methods_should_we_auto_overload(
            tensor_type, syft_type
        )

        # Rename native functions
        self._rename_native_functions(tensor_type)

        # Overload auto overloaded with Torch methods
        self._add_methods_from__torch_tensor(tensor_type, syft_type)

    def _add_registration_to___init__(hook_self, tensor_type: type, torch_tensor: bool = False):
        """Adds several attributes to the tensor.

        Overloads tensor_type.__init__ to add several attributes to the tensor
        as well as (optionally) registering the tensor automatically.
        TODO: auto-registration is disabled at the moment, this might be bad.

        Args:
            tensor_type (type): The type of tensor being hooked (in this
                refactor this is only ever torch.Tensor, but in previous
                versions of PySyft this iterated over all tensor types.
            torch_tensor (bool): Optional, If set to true, skip running the
                native initialization logic. TODO: this flag might never
                get used.
        """
        if "native___init__" not in dir(tensor_type):
            tensor_type.native___init__ = tensor_type.__init__

        def new___init__(cls, *args, owner=None, id=None, register=True, **kwargs):

            initialize_tensor(
                hook_self=hook_self,
                cls=cls,
                id=id,
                torch_tensor=torch_tensor,
                init_args=args,
                init_kwargs=kwargs,
            )

            # if register:
            #     owner.register_object(cls, id=id)

        tensor_type.__init__ = new___init__

    @staticmethod
    def _hook_properties(tensor_type: type):
        """Overloads tensor_type properties.

        This method gets called only on torch.Tensor. If you're not sure how
        properties work, read:
        https://www.programiz.com/python-programming/property

        Args:
            tensor_type (type): The tensor type which is having properties
                added to it, typically just torch.Tensor.
        """

        @property
        def location(self):
            return self.child.location

        tensor_type.location = location

        @property
        def id_at_location(self):
            return self.child.id_at_location

        tensor_type.id_at_location = id_at_location

        @property
        def id(self):
            if self.is_wrapper:
                return self.child.id
            else:
                return self._id

        @id.setter
        def id(self, new_id):
            if self.is_wrapper:
                self.child.id = new_id
            else:
                self._id = new_id
            return self

        tensor_type.id = id

        @property
        def is_wrapper(self):
            if not hasattr(self, "_is_wrapper"):
                self._is_wrapper = False
            return self._is_wrapper

        @is_wrapper.setter
        def is_wrapper(self, it_is_a_wrapper):
            self._is_wrapper = it_is_a_wrapper
            return self

        tensor_type.is_wrapper = is_wrapper

    def _which_methods_should_we_auto_overload(self, tensor_type: type, syft_type: type):
        """Creates list of Torch methods to auto overload.

        By default, it looks for the intersection between the methods of
        tensor_type and torch_type minus those in the exception list
        (syft.torch.exclude).

        Args:
            tensor_type (type): Iterating through a tensor type's properties
            syft_type (type): iterate through all attributes in this type.

        Returns:
            A list of methods to be overloaded.
        """

        to_overload = []

        for attr in dir(syft_type):

            # Conditions for overloading the method
            if attr in syft.torch.exclude:
                continue
            if not hasattr(tensor_type, attr):
                continue

            lit = getattr(tensor_type, attr)
            is_base = attr in dir(object)
            is_desc = inspect.ismethoddescriptor(lit)
            is_func = isinstance(lit, types.FunctionType)
            try:
                is_service_func = "HookService" in lit.__qualname__
            except AttributeError:
                is_service_func = False
            is_overloaded = re.match("native*", attr) is not None

            if (is_desc or (is_func and not is_service_func)) and not is_base and not is_overloaded:
                to_overload.append(attr)

        return to_overload

    def _rename_native_functions(self, tensor_type: type):
        """Renames functions that are not auto overloaded as native functions.

        Args:
            tensor_type (type): The tensor whose native methods are getting
                renamed. Typically just torch.Tensor.
        """
        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)

            # if we haven't already overloaded this function
            if f"native_{attr}" not in dir(tensor_type):
                setattr(tensor_type, f"native_{attr}", lit)

            setattr(tensor_type, attr, None)

    @staticmethod
    def _add_methods_from__torch_tensor(tensor_type: type, syft_type: type):
        """Adds methods from the TorchTensor class to the native torch tensor.

        The class TorchTensor is a proxy to avoid extending directly the torch
        tensor class.

        Args:
            tensor_type (type): The tensor type to which we are adding methods
                from TorchTensor class.
        """
        exclude = [
            "__class__",
            "__delattr__",
            "__dir__",
            "__doc__",
            "__dict__",
            "__format__",
            "__getattribute__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__weakref__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__setattr__",
            "__sizeof__",
            "__subclasshook__",
            "_get_type",
            "__str__",
            "__repr__",
            "__eq__",
            "__gt__",
            "__ge__",
            "__lt__",
            "__le__",
        ]
        # For all methods defined in TorchTensor which are not internal methods (like __class__etc)
        for attr in dir(syft_type):
            if attr not in exclude:
                # Add to the native tensor this method
                setattr(tensor_type, attr, getattr(TorchTensor, attr))
