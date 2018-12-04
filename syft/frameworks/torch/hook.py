import random
import inspect
import re
import logging
import types
import syft
from syft import workers
from syft.frameworks.torch.tensors import PointerTensor, AbstractTensor, TorchTensor
from syft.frameworks.torch.torch_attributes import TorchAttributes


class TorchHook:
    r""" A Hook which Overrides Methods on PyTorch Tensors -

     The purpose of this class is to:

         * extend torch methods to allow for the moving of tensors
           from one worker to another
         * override torch methods to execute commands on one worker
           that are called on tensors controlled by the local worker.

     This class is typically the first thing you will initialize when
     using PySyft with PyTorch because it is responsible for augmenting
     PyTorch with PySyft's added functionality (such as remote execution).

     :Parameters:

         * **local_worker (**:class:`.workers.BaseWorker` **, optional)**
           you can optionally provide a local worker as a parameter which
           TorchHook will assume to be the worker owned by the local machine.
           If you leave it empty, TorchClient will automatically initialize
           a :class:`.workers.VirtualWorker` under the assumption you're
           looking to do local experimentation/development.

         * **is_client (bool, optional)** whether or not the TorchHook is
           being initialized as an end-user client. This can impact whether
           or not variables are deleted when they fall out of scope. If you set
           this incorrectly on a end user client, Tensors and Variables will
           never be deleted. If you set this incorrectly on a remote machine
           (not a client), tensors will not get saved. It's really only
           important if you're not initializing the local worker yourself. (Default: True)

         * **verbose (bool, optional)** whether or not to print operations
           as they occur. (Default: True)

         * **queue_size (int, optional)** max length of the list storing messages
           to be sent. (Default: 0)

     :Example:

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

    def __init__(self, torch, local_worker=None, is_client=True, verbose=True):
        """
        Init the hook and define all the attribute pertaining to the torch hook in a
        special TorchAttibute class, that will be added in the syft.torch attributes.
        Hence, this parameters are now conveyed by the syft module.

        :param torch: the torch module provided by the user, which will be hooked
        :param local_worker: the local_worker that will handle the computation. Can be created
        if not provided.
        :param is_client: defines if the node/worker is the client that control computations.
        :param verbose:
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

        self.to_auto_overload = {}

        self._hook_native_tensors(torch.Tensor)

        self._hook_syft_tensors(torch.Tensor)

        self._hook_torch_module()

        syft.torch.eval_torch_modules()

        # Add the local_worker to syft so that it can be found if the hook is called several times
        syft.local_worker = self.local_worker

    def _hook_native_tensors(self, tensor_type):

        """Overloads given native tensor type (Torch Tensor) to add PySyft Tensor Functionality
           parameters: tensor_type: A Torch tensor
        """

        # Reinitialize init method of Torch tensor with Syft init
        self._add_registration_to___init__(tensor_type, torch_tensor=True)

        # Overload Torch tensor properties with Syft properties
        self._hook_properties(tensor_type)

        # Returns a list of methods to be overloaded, stored in the dict to_auto_overload
        # with tensor_type as a key
        self.to_auto_overload[tensor_type] = self._which_methods_should_we_auto_overload(
            tensor_type
        )

        # Rename native functions
        self._rename_native_functions(tensor_type)

        self._assign_methods_to_use_child(tensor_type)

        # Overload auto overloaded with Torch methods
        self._add_methods_from__torch_tensor(tensor_type)

    def _hook_syft_tensors(self, tensor_type):

        """Overloads Torch Tensors with all Syft tensor types
           parameters: tensor_type: A Torch tensor
        """

        self._hook_abstract_tensor(tensor_type)

        self._hook_pointer_tensor(tensor_type)

    def _add_registration_to___init__(hook_self, tensor_type, torch_tensor=False):

        """Overloads tensor_type.__init__ or Variable.__init__ of Torch tensors
           to add PySyft tensor functionality
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

            # if register:
            #     owner.register_object(cls, id=id)

        tensor_type.__init__ = new___init__

    @staticmethod
    def _hook_properties(tensor_type):
        """Overloads tensor_type properties
           Parameters: tensor_type: Torch tensor
        """

        @property
        def location(self):
            return self.child.location

        tensor_type.location = location

        @property
        def id_at_location(self):
            return self.child.id_at_location

        tensor_type.id_at_location = id_at_location

        # @property
        # def owner(self):
        #     return self.child.owner
        #
        # tensor_type.owner = owner

    def _which_methods_should_we_auto_overload(self, tensor_type):
        """Creates list of Torch methods to auto overload except methods included in exclusion list
           Parameters: Torch Tensor type
           Return: List of methods to be overloaded
        """

        to_overload = []

        for attr in dir(tensor_type):

            # Conditions for overloading the method
            if attr in syft.torch.exclude:
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

    def _rename_native_functions(self, tensor_type):
        """Renames functions that are that not auto overloaded as native functions
           Parameters: tensor_type: Torch tensor
        """

        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)

            # if we haven't already overloaded this function
            if f"native_{attr}" not in dir(tensor_type):
                setattr(tensor_type, f"native_{attr}", lit)

            setattr(tensor_type, attr, None)

    def _assign_methods_to_use_child(self, tensor_type):
        """Assigns methods to use as child for auto overloaded functions.
           Parameters: tensor_type:Torch Tensor
        """

        # Iterate through auto overloaded tensor methods
        for attr in self.to_auto_overload[tensor_type]:
            new_attr = self._get_overloaded_method(attr)
            setattr(tensor_type, attr, new_attr)

    @staticmethod
    def _add_methods_from__torch_tensor(tensor_type):
        """Add methods from the TorchTensor class to the native torch tensor.
           The class TorchTensor is a proxy to avoid extending directly the
           torch tensor class.
           Parameters: tensor_type: Torch Tensor
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
        for attr in dir(TorchTensor):
            if attr not in exclude:
                # Add to the native tensor this method
                setattr(tensor_type, attr, getattr(TorchTensor, attr))

    def _hook_abstract_tensor(hook_self, tensor_type):
        """Overloads AbstractTensor
           Parameters: tensor_type:Torch Tensor
        """
        hook_self._add_registration_to___init__(AbstractTensor)

        for attr in hook_self.to_auto_overload[tensor_type]:
            new_attr = hook_self._get_overloaded_method(attr)
            setattr(AbstractTensor, attr, new_attr)

    def _hook_pointer_tensor(self, tensor_type):
        """Overloads PointerTensor
           Parameters: tensor_type:Torch Tensor
        """
        # TODO: isn't it redundant iwth the upper one ?
        for attr in self.to_auto_overload[tensor_type]:
            new_attr = self._get_overloaded_method(attr)
            setattr(PointerTensor, attr, new_attr)

    def _hook_torch_module(self):
        """Overloads functions in the main torch module.

        The way this is accomplished is by first moving all existing module functions in the torch
        module to native_<function_name_here>. Thus, the real :func:`torch.cat` will become
        :func:`torch.native_cat` and :func:`torch.cat` will have our hooking code.
        """

        for module_name, torch_module in syft.torch.torch_modules.items():
            module_funcs = dir(torch_module)
            torch_module = syft.torch.torch_modules[module_name]
            for attr in module_funcs:
                # Some functions we want to ignore (not override). Such functions have been hard
                # coded into the attribute self.torch_exclude
                if attr in syft.torch.torch_exclude:
                    continue

                # if we haven't already overloaded this function
                if f"native_{attr}" in dir(torch_module):
                    continue

                # if we haven't already overloaded this function (redundancy allowed)
                if "native_" in attr:
                    continue

                # Where the overloading happens
                lit = getattr(torch_module, attr)
                if type(lit) in [types.FunctionType, types.BuiltinFunctionType]:
                    new_attr = self._get_overloaded_function(f"{module_name}.{attr}")
                    setattr(torch_module, f"native_{attr}", lit)
                    setattr(torch_module, attr, new_attr)

    def _get_overloaded_method(hook_self, attr):
        """Wrapper overloading partial objects of methods in the torch module.

        Compiles command, checks for Tensors and Variables in the
        args/kwargs, determines locations of all Tensors and Variables
        involved in computation, and handles the computation
        accordingly.
        """

        def _execute_method_call(self, *args, **kwargs):
            worker = hook_self.local_worker
            return worker._execute_call(attr, self, *args, **kwargs)

        return _execute_method_call

    def _get_overloaded_function(hook_self, attr):
        """Wrapper overloading partial objects of functions in the torch
        module.

        Compiles command, checks for Tensors and Variables in the
        args/kwargs, determines locations of all Tensors and Variables
        involved in computation, and handles the computation
        accordingly.
        """

        def _execute_function_call(*args, **kwargs):
            worker = hook_self.local_worker
            return worker._execute_call(attr, None, *args, **kwargs)

        return _execute_function_call
