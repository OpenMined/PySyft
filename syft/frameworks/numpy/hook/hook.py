import copy
from functools import wraps
import logging
from math import inf
import weakref

import numpy

import syft
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.hook.hook import FrameworkHook
from syft.generic.frameworks.remote import Remote
from syft.frameworks.numpy.tensors.interpreters.native import NumpyTensor
#from syft.frameworks.torch.tensors.interpreters.paillier import PaillierTensor
#from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
#from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
#from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
#from syft.frameworks.torch.tensors.interpreters.large_precision import LargePrecisionTensor
from syft.frameworks.numpy.numpy_attributes import NumpyAttributes
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.tensor import _apply_args
from syft.workers.base import BaseWorker
from syft.workers.virtual import VirtualWorker
from syft.messaging.plan import Plan





class NumpyHook(FrameworkHook):
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
        local_worker: An optional BaseWorker instance that lets you provide a
            local worker as a parameter which TorchHook will assume to be the
            worker owned by the local machine. If you leave it empty,
            TorchClient will automatically initialize a
            :class:`.workers.VirtualWorker` under the assumption you're looking
            to do local experimentation or development.
        is_client: An optional boolean parameter (default True), indicating
            whether TorchHook is being initialized as an end-user client.This
            can impact whether or not variables are deleted when they fall out
            of scope. If you set this incorrectly on a end user client, Tensors
            and Variables will never be deleted. If you set this incorrectly on
            a remote machine (not a client), tensors will not get saved. It's
            really only important if you're not initializing the local worker
            yourself.
        verbose: An optional boolean parameter (default True) to indicate
            whether or not to print the operations as they occur.
        queue_size: An integer optional parameter (default 0) to specify the
            max length of the list that stores the messages to be sent.

    Example:
        >>> import torch as th
        >>> import syft as sy
        >>> hook = sy.TorchHook(th)
        Hooking into Torch...
        Overloading Complete.
        # constructing a normal torch tensor in pysyft
        >>> x = th.Tensor([-2,-1,0,1,2,3])
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
        self, numpy, local_worker: BaseWorker = None, is_client: bool = True, verbose: bool = True
    ):
        """Initializes the hook.

        Initialize the hook and define all the attributes pertaining to the
        torch hook in a special TorchAttibute class, that will be added in the
        syft.torch attributes. Hence, this parameters are now conveyed by the
        syft module.
        """
        # Wrap the provided numpy module and save it as an attribute of the hook
        self.numpy = numpy
        self.framework = self.numpy

        # Save the local worker as an attribute
        self.local_worker = local_worker

        if hasattr(numpy, "numpy_hooked"):
            logging.warning("Numpy was already hooked... skipping hooking process")
            self.local_worker = syft.local_worker
            return
        else:
            numpy.numpy_hooked = True

        # Add all the torch attributes in the syft.torch attr
        syft.numpy = NumpyAttributes(self.numpy, self)
        syft.framework = syft.numpy

        # Hook some torch methods such that tensors could be created directy at workers
        self._hook_worker_methods()

        if self.local_worker is None:
            # Every TorchHook instance should have a local worker which is
            # responsible for interfacing with other workers. The worker
            # interface is what allows the Torch specific code in TorchHook to
            # be agnostic to the means by which workers communicate (such as
            # peer-to-peer, sockets, through local ports, or all within the
            # same process)
            self.local_worker = VirtualWorker(hook=self, is_client_worker=is_client, id="me")
        else:
            self.local_worker.hook = self

        self.to_auto_overload = {}

        self.args_hook_for_overloaded_attr = {}

        self._hook_native_tensor(syft.ndarray, NumpyTensor)

        # Add all hooked tensor methods to pointer but change behaviour to have the cmd sent
        #self._hook_pointer_tensor_methods(syft.ndarray)

        # Add all hooked tensor methods to AdditiveSharingTensor tensor but change behaviour
        # to all shares (when it makes sense, otherwise the method is overwritten in the
        # AdditiveSharingTensor class)
        """
        self._hook_additive_shared_tensor_methods()
        """

        # Add all hooked tensor methods to multi_pointer to change behavior to have the cmd
        # sent to all child pointers.
        #self._hook_multi_pointer_tensor_methods(syft.ndarray)

        # Add all hooked tensor methods to Logging tensor but change behaviour to just forward
        # the cmd to the next child (behaviour can be changed in the SyftTensor class file)
        """
        self._hook_syft_tensor_methods(LoggingTensor)
        """

        # Add all hooked tensor methods to Paillier tensor but change behaviour to just forward
        # the cmd to the next child (behaviour can be changed in the SyftTensor class file)
        """
        self._hook_syft_tensor_methods(PaillierTensor)
        """

        # Add all hooked tensor methods to FixedPrecisionTensor tensor but change behaviour
        # to just forward the cmd to the next child (behaviour can be changed in the
        # SyftTensor class file)
        """
        self._hook_syft_tensor_methods(FixedPrecisionTensor)
        """

        # Add all hooked tensor methods to AutogradTensor tensor but change behaviour
        # to just forward the cmd to the next child (behaviour can be changed in the
        # SyftTensor class file)
        """
        self._hook_syft_tensor_methods(AutogradTensor)
        """

        # Add all hooked tensor methods to AdditiveSharingTensor tensor but change behaviour
        # to just forward the cmd to the next child (behaviour can be changed in the
        # SyftTensor class file)
        """
        self._hook_syft_tensor_methods(AdditiveSharingTensor)
        """

        # Add all hooked tensor methods to LargePrecisionTensor tensor
        """
        self._hook_syft_tensor_methods(LargePrecisionTensor)
        """

        # Hook the tensor constructor function
        self._hook_array()

        # Hook torch functions from modules like torch.add OR torch.nn.functional (containing relu, etc.)
        #self._hook_numpy_module()

        # Add the local_worker to syft so that it can be found if the hook is
        # called several times
        syft.local_worker = self.local_worker
        syft.hook = self


    def create_shape(cls, shape_dims):
        return torch.Size(shape_dims)


    def create_wrapper(cls, wrapper_type):
        # Note this overrides FrameworkHook.create_wrapper, so it must conform to
        # that classmethod's signature
        assert (
            wrapper_type is None or wrapper_type == syft.ndarray
        ), "NumpyHook only uses syft.ndarray wrappers"

        return syft.ndarray((0,))


    def create_zeros(cls, *shape, dtype=None, **kwargs):
        return torch.zeros(*shape, dtype=dtype, **kwargs)

    def _hook_native_tensor(self, tensor_type: type, syft_type: type):
        """Adds PySyft Tensor Functionality to the given native tensor type.

        Overloads the given native Torch tensor to add PySyft Tensor
        Functionality. Overloading involves modifying the tensor type with
        PySyft's added functionality. You may read about what kind of
        modifications are made in the methods that this method calls.

        Args:
            tensor_type: The type of tensor being hooked (in this refactor
                this is only ever torch.Tensor, but in previous versions of
                PySyft this iterated over all tensor types.
            syft_type: The abstract type whose methods should all be added to
                the tensor_type class. In practice this is always TorchTensor.
                Read more about it there.
        """
        # Reinitialize init method of Torch tensor with Syft init
        self._add_registration_to___init__(tensor_type, is_tensor=True)

        # Overload Torch tensor properties with Syft properties
        self._hook_properties(tensor_type)

        # Returns a list of methods to be overloaded, stored in the dict to_auto_overload
        # with tensor_type as a key
        self.to_auto_overload[tensor_type] = self._which_methods_should_we_auto_overload(
            tensor_type
        )

        # [We don't rename native methods as torch tensors are not hooked] Rename native functions
        # #self._rename_native_functions(tensor_type)

        # Overload auto overloaded with Torch methods
        self._transfer_methods_to_native_tensor(tensor_type, syft_type)

        self._hook_native_methods(tensor_type)

    def __hook_properties(self, tensor_type):
        super()._hook_properties(tensor_type)
        tensor_type.native_shape = tensor_type.shape

    def _hook_syft_tensor_methods(self, syft_type: type):
        tensor_type = syft.ndarray
        super()._hook_syft_tensor_methods(tensor_type, syft_type)

    def _hook_worker_methods(self):
        class Numpy(object):
            name = "numpy"

            def __init__(self, worker, *args, **kwargs):
                self.worker = weakref.ref(worker)

        Remote.register_framework(Numpy)

        for attr in syft.numpy.worker_methods:
            new_method = self._get_hooked_base_worker_method(attr)
            setattr(Numpy, attr, new_method)

    def _get_hooked_base_worker_method(hook_self, attr):
        @wraps(attr)
        def overloaded_attr(self_numpy, *args, **kwargs):
            ptr = hook_self.local_worker.send_command(
                recipient=self_numpy.worker(),
                message=("{}.{}".format("numpy", attr), None, args, kwargs),
            )

            return ptr.wrap()

        return overloaded_attr

    def _hook_additive_shared_tensor_methods(self):
        """
        Add hooked version of all methods of the torch Tensor to the
        Additive Shared tensor: instead of performing the native tensor
        method, it will be forwarded to each share when it is relevant
        """

        tensor_type = syft.ndarray
        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(AdditiveSharingTensor):
                new_method = self._get_hooked_additive_shared_method(attr)
                setattr(AdditiveSharingTensor, attr, new_method)


    def _hook_numpy_module(self):
        """Overloads functions in the main torch modules.
        The way this is accomplished is by first moving all existing module
        functions in the torch module to native_<function_name_here>.

        Example:
            the real :func:`torch.cat` will become :func:`torch.native_cat`
            and :func:`torch.cat` will have our hooking code.
        """

        numpy_modules = syft.numpy.numpy_modules

        for module_name, numpy_module in numpy_modules.items():
            for func in dir(numpy_module):

                # Some functions we want to ignore (not override). Such functions have been hard
                # coded into the torch_attribute exclude (see TorchAttribute class)
                if func in syft.numpy.exclude:
                    continue

                # ignore dunder functions
                if "__" in func:
                    continue

                # ignore capitalized func values which are Classes not functions
                if func[0].isupper():
                    continue

                # ignore hidden functins
                if func[0] == "_":
                    continue

                # If we haven't already overloaded this function
                if "native_" in func or f"native_{func}" in dir(numpy_module):
                    continue

                self._perform_function_overloading(module_name, numpy_module, func)

    @classmethod
    def _get_hooked_func(cls, public_module_name, func_api_name, attr):
        """Torch-specific implementation. See the subclass for more."""
        if attr.__module__ is None:
            attr.__module__ = "numpy"

        return super()._get_hooked_func(attr.__module__, func_api_name, attr)

    def _get_hooked_additive_shared_method(hook_self, attr):
        """
        Hook a method to send it multiple remote workers

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        def dispatch(args, k):
            return map(lambda x: x[k] if isinstance(x, dict) else x, args)

        @wraps(attr)
        def overloaded_attr(self, *args, **kwargs):
            """
            Operate the hooking
            """

            # Replace all syft tensor with their child attribute
            new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                attr, self, args, kwargs
            )

            results = {}
            for k, v in new_self.items():
                results[k] = v.__getattribute__(attr)(*dispatch(new_args, k), **new_kwargs)

            # Put back AdditiveSharingTensor on the tensors found in the response
            response = hook_args.hook_response(
                attr,
                results,
                wrap_type=AdditiveSharingTensor,
                wrap_args=self.get_class_attributes(),
            )

            return response

        return overloaded_attr

    def _hook_array(hook_self):
        """Hooks the function numpy.array()
        We need to do this seperately from hooking the class because internally
        torch does not pick up the change to add the args
        Args:
            hook_self: the hook itself
        """

        if "native_array" not in dir(hook_self.numpy):
            hook_self.numpy.native_array = hook_self.numpy.array

        def new_array(*args, owner=None, id=None, register=True, **kwargs):
            current_array = hook_self.numpy.native_array(*args, **kwargs).view(syft.ndarray)
            _apply_args(hook_self, current_array, owner, id)
            if register:
                current_array.owner.register_obj(current_array)

            return current_array

        hook_self.numpy.array = new_array

    @classmethod
    def _transfer_methods_to_native_tensor(cls, tensor_type: type, syft_type: type):
        """Adds methods from the TorchTensor class to the native torch tensor.

        The class TorchTensor is a proxy to avoid extending directly the torch
        tensor class.

        Args:
            tensor_type: The tensor type to which we are adding methods
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
            "__module__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__setattr__",
            "__sizeof__",
            "__subclasshook__",
            "_get_type",
            # "__eq__", # FIXME it now overwritten in native.py to use torch.eq, because of pb between == & __eq__ See #2030
            "__gt__",
            "__ge__",
            "__lt__",
            "__le__",
        ]
        cls._transfer_methods_to_framework_class(tensor_type, syft_type, exclude)
