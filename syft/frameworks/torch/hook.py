import random
import inspect
import re
import logging
import types
import syft
from syft import workers
from syft.frameworks.torch.tensors import SyftTensor
from syft.frameworks.torch.torch_attributes import TorchAttributes


class TorchHook:
    r"""
    A Hook which Overrides Methods on PyTorch Tensors -

    The purpose of this class is to:

        * extend torch methods to allow for the moving of tensors
           from one worker to another
        * override torch methods to execute commands on one worker
           that are called on tensors controlled by the local worker.

    This class is typically the first thing you will initialize when
    using PySyft with PyTorch because it is responsible for augmenting
    PyTorch with PySyft's added functionality (such as remote execution).

    Overloading a pytorch tensor method in pysyft is done by creating the
    overloaded method in a syft.frameworks.tensor class, and if needed,
    add this to the hook below.
    """

    def __init__(self, torch, local_worker=None, is_client=True):
        """
        Init the hook and define all the attribute pertaining to the torch hook in a
        special TorchAttibute class, that will be added in the syft.torch attributes.
        Hence, this parameters are now conveyed by the syft module.

        Args:
             torch: represents torch module provided by the user,
               which will be hooked. The hook will add an torch_hooked variable
               to the module, which is needed to check whether the module is already
               hooked or not.

             local_worker (workers.BaseWorker):
               you can optionally provide a local worker as a parameter which
               TorchHook will assume to be the worker owned by the local machine.
               If you leave it empty, TorchClient will automatically initialize
               a :class:`workers.VirtualWorker` under the assumption you're
               looking to do local experimentation/development.

             is_client (bool):  whether or not the TorchHook is
               being initialized as an end-user client. This can impact whether
               or not variables are deleted when they fall out of scope. If you set
               this incorrectly on a end user client, Tensors and Variables will
               never be deleted. If you set this incorrectly on a remote machine
               (not a client), tensors will not get saved. It's really only
               important if you're not initializing the local worker yourself.

               **Warning**: if `local_worker` is provided, set`is_client` on the
               provided worker.
        """

        self.torch = torch
        self.local_worker = local_worker

        if hasattr(torch, "torch_hooked"):
            logging.warning("Torch was already hooked... skipping hooking process")
            self.local_worker = syft.local_worker
            return
        else:
            torch.torch_hooked = True

        # Add all the torch attributes in the syft.torch attr
        syft.torch = TorchAttributes(torch)

        if self.local_worker is None:
            """
            Every TorchHook instance should have a local worker which is responsible for
            interfacing with other workers. The worker interface is what allows the Torch
            specific code in TorchHook to be agnostic to the means by which workers communicate
            (such as peer-to-peer, sockets, through local ports, or all within the same process)
            """
            self.local_worker = workers.VirtualWorker(
                hook=self, is_client_worker=is_client, id="me"
            )
        else:
            self.local_worker.hook = self

        self.to_overload = {}

        self._hook_native_tensor(torch.Tensor, SyftTensor)

        # Make local_worker globaly available, so each created tensor can reference the same
        # local_worker.
        syft.local_worker = self.local_worker

    def _hook_native_tensor(self, tensor_type, syft_type):
        """
        Overloads given native tensor type (torch Tensor) with given syft tensor type
        (e.g. syft.SyftTensor) to add PySyft Tensor Functionality. The idea is to only
        overload methods specified within the syft_type and move the original methods
        to "native_{method_name}".

        Example:
            Overloading a native torch tensor method within a syft tensor

            def reshape(self, *args, **kwargs):
                #executing custom syft code
                print("Overloading torch.Tensor.reshape")

                #calling the native torch methoc
                return getattr(self, "native_reshape")(*args, **kwargs)

        Args:
            tensor_type: a torch tensor to be hooked
            syft_type: a syft tensor which provides methods to be overloaded
        """

        # Reinitialize init method of Torch tensor with Syft init
        self._add_registration_to___init__(tensor_type, torch_tensor=True)

        # Returns a list of methods to be overloaded, stored in the dict to_overload
        # with tensor_type as a key
        self.to_overload[tensor_type] = self._identify_methods_to_overload(tensor_type, syft_type)

        self._rename_native_methods(tensor_type)
        self._add_methods_to_tensor(tensor_type, syft_type)

    def _add_registration_to___init__(hook_self, tensor_type, torch_tensor=False):
        """
        Overloads tensor_type.__init__ or Variable.__init__ of Torch tensors
        to add PySyft tensor functionality.

        Args:
            tensor_type: a torch tensor
            torch_tensor (boolean): currently unsure for what it's used
        """

        if "native___init__" not in dir(tensor_type):
            tensor_type.native___init__ = tensor_type.__init__

        def new___init__(cls, *args, owner=None, id=None, register=True, **kwargs):
            if not torch_tensor:
                cls.native___init__(*args, **kwargs)

            if owner is None:
                owner = hook_self.local_worker
            if id is None:
                id = int(10e10 * random.random())

            cls.id = id
            cls.owner = owner

        tensor_type.__init__ = new___init__

    def _identify_methods_to_overload(self, tensor_type, syft_type):
        """
        This identifies methods within the syft_type tensor which hold the same name as
        the native torch method. Those methods are added to the list of methods
        to be overloaded. Methods within the exclude list are exlcuded.

        Args:
            tensor_type: torch tensor
            syft_type: syft tensor

        Returns:
            List of method names to be overloaded
        """

        to_overload = []

        for attr in dir(syft_type):
            # Conditions for overloading the method
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

    def _rename_native_methods(self, tensor_type):
        """
        This method will rename methods of tensor_type that will be
        overloaded. The new name will we f"native_{method_name}".

        Args:
            tensor_type: Torch tensor
        """

        for attr in self.to_overload[tensor_type]:

            lit = getattr(tensor_type, attr)

            # if we haven't already overloaded this function
            if f"native_{attr}" not in dir(tensor_type):
                setattr(tensor_type, f"native_{attr}", lit)

            setattr(tensor_type, attr, None)

    def _add_methods_to_tensor(self, tensor_type, syft_type):
        """
        Add methods from the syft_type class to the torch tensor_type.
        This adds all methods except the excluded ones defined in the
        syft_type.

            Args:
                tensor_type: Torch tensor
                syft_type: syft tensor
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

        for attr in dir(syft_type):
            if attr not in exclude:
                # Add to the native tensor this method
                setattr(tensor_type, attr, getattr(syft_type, attr))
