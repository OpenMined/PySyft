import copy
from functools import wraps
import logging
from math import inf
import torch
import weakref

import syft
from syft import dependency_check
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.hook.hook import FrameworkHook
from syft.generic.frameworks.remote import Remote
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.frameworks.torch.tensors.interpreters.native import TorchTensor
from syft.frameworks.torch.tensors.interpreters.hook import HookedTensor
from syft.frameworks.torch.tensors.interpreters.paillier import PaillierTensor
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters.private import PrivateTensor
from syft.execution.placeholder import PlaceHolder
from syft.frameworks.torch.torch_attributes import TorchAttributes
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.abstract.tensor import _apply_args
from syft.workers.base import BaseWorker
from syft.workers.virtual import VirtualWorker
from syft.execution.plan import Plan


class TorchHook(FrameworkHook):
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
        self,
        torch,
        local_worker: BaseWorker = None,
        is_client: bool = True,
        verbose: bool = False,
        seed=None,
    ):
        """
        Initializes the hook.

        Initialize the hook and define all the attributes pertaining to the
        torch hook in a special TorchAttibute class, that will be added in the
        syft.torch attributes. Hence, this parameters are now conveyed by the
        syft module.
        """
        # Save the provided torch module as an attribute of the hook
        self.torch = torch
        self.framework = self.torch
        if seed is not None:
            syft.ID_PROVIDER.seed(seed)
        self.verbose = verbose

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
        syft.framework = syft.torch

        """
        In Syft there is a syft.framework value that can contain only one framework.
        Ideally it should contain a list of supported frameworks.

        We do this because in Plans there is method to reduce the number of actions
        that are traced (and then sent).
        The actions that are not returning a result, changing a placeholder, inplace
        or changing the global state are eliminated from the traced list
        """
        if dependency_check.crypten_available:
            import crypten
            from syft.frameworks.crypten.crypten_attributes import CryptenAttributes

            syft.crypten = CryptenAttributes(crypten, self)

        # Hook some torch methods such that tensors could be created directy at workers
        self._hook_worker_methods()

        if self.local_worker is None:
            # Every TorchHook instance should have a local worker which is
            # responsible for interfacing with other workers. The worker
            # interface is what allows the Torch specific code in TorchHook to
            # be agnostic to the means by which workers communicate (such as
            # peer-to-peer, sockets, through local ports, or all within the
            # same process)
            self.local_worker = VirtualWorker(
                hook=self, is_client_worker=is_client, id="me", verbose=verbose
            )
        else:
            self.local_worker.hook = self

        self._syft_workers = {self.local_worker}

        self.to_auto_overload = {}

        self.args_hook_for_overloaded_attr = {}

        self._hook_native_tensor(torch.Tensor, TorchTensor)

        if dependency_check.crypten_available:
            from syft.frameworks.crypten.hook.hook import crypten_to_auto_overload

            for crypten_class, method_names in crypten_to_auto_overload.items():
                self.to_auto_overload[crypten_class] = method_names
                self._hook_syft_placeholder_methods(crypten_class, PlaceHolder)

        # Add all hooked tensor methods to pointer but change behaviour to have the cmd sent
        self._hook_pointer_tensor_methods(self.torch.Tensor)

        # Add all hooked tensor methods to AdditiveSharingTensor tensor but change behaviour
        # to all shares (when it makes sense, otherwise the method is overwritten in the
        # AdditiveSharingTensor class)
        self._hook_additive_shared_tensor_methods()

        # Add all hooked tensor methods to multi_pointer to change behavior to have the cmd
        # sent to all child pointers.
        self._hook_multi_pointer_tensor_methods(self.torch.Tensor)

        # Add all hooked tensor methods to Logging tensor but change behaviour to just forward
        # the cmd to the next child (behaviour can be changed in the SyftTensor class file)
        self._hook_syft_tensor_methods(LoggingTensor)

        # Add all hooked tensor methods to Paillier tensor but change behaviour to just forward
        # the cmd to the next child (behaviour can be changed in the SyftTensor class file)
        self._hook_syft_tensor_methods(PaillierTensor)

        # Add all hooked tensor methods to FixedPrecisionTensor tensor but change behaviour
        # to just forward the cmd to the next child (behaviour can be changed in the
        # SyftTensor class file)
        self._hook_syft_tensor_methods(FixedPrecisionTensor)

        # Add all hooked tensor methods to AutogradTensor tensor but change behaviour
        # to just forward the cmd to the next child (behaviour can be changed in the
        # SyftTensor class file)
        self._hook_syft_tensor_methods(AutogradTensor)

        # Add all hooked tensor methods to PrivateTensor tensor but change behaviour
        # to just forward the cmd to the next child (behaviour can be changed in the
        # SyftTensor class file)
        self._hook_private_tensor_methods(PrivateTensor)

        # Add all hooked tensor methods to PlaceHolder tensor but change behaviour
        # to just forward the cmd to the next child (behaviour can be changed in the
        # SyftTensor class file)
        self._hook_syft_placeholder_methods(self.torch.Tensor, PlaceHolder)

        # Add all hooked tensor methods to AdditiveSharingTensor tensor but change behaviour
        # to just forward the cmd to the next child (behaviour can be changed in the
        # SyftTensor class file)
        self._hook_syft_tensor_methods(AdditiveSharingTensor)

        # Add all hooked tensor methods to NumpyTensor tensor
        self._hook_syft_tensor_methods(HookedTensor)

        # Add all built-in 'str' methods to String
        self._hook_string_methods(owner=self.local_worker)

        # Add all string methods to StringPointer
        # This method call should strictly come after the
        # call to self._hook_string_methods()
        self._hook_string_pointer_methods()

        # Hook the tensor constructor function
        self._hook_tensor()

        # Hook the Parameter methods to store tensor chains in parameters
        self._hook_parameters()

        # Hook torch functions from modules like torch.add OR
        # torch.nn.functional (containing relu, etc.)
        self._hook_torch_module()

        # Hook torch.nn (containing Linear and Convolution layers)
        self._hook_module()

        # Hook torch.optim (containing optim.SGD, Adam, etc)
        self._hook_optim()

        # Hook the Crypten module
        if dependency_check.crypten_available:
            from syft.frameworks.crypten.hook.hook import hook_crypten, hook_crypten_module

            hook_crypten()
            hook_crypten_module()

        # Add the local_worker to syft so that it can be found if the hook is
        # called several times
        syft.local_worker = self.local_worker
        syft.hook = self

    def create_shape(cls, shape_dims):
        return torch.Size(shape_dims)

    def create_wrapper(cls, wrapper_type):
        # Note this overrides FrameworkHook.create_wrapper, so it must conform to
        # that classmethod's signature
        if wrapper_type is None or wrapper_type == torch.Tensor:
            return torch.Tensor()
        elif isinstance(wrapper_type, torch.dtype):
            return torch.tensor([], dtype=wrapper_type)
        else:
            raise ValueError(
                "Wrapper type should be None, torch.Tensor, or a torch.dtype like torch.long"
            )

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
        tensor_type = self.torch.Tensor
        super()._hook_syft_tensor_methods(tensor_type, syft_type)

    def _hook_private_tensor_methods(self, syft_type: type):
        tensor_type = self.torch.Tensor
        super()._hook_private_tensor_methods(tensor_type, syft_type)

    def _hook_worker_methods(self):
        class Torch(object):
            name = "torch"

            def __init__(self, worker, *args, **kwargs):
                self.worker = weakref.ref(worker)

        Remote.register_framework(Torch)

        for attr in syft.torch.worker_methods:
            new_method = self._get_hooked_base_worker_method(attr)
            setattr(Torch, attr, new_method)

    def _get_hooked_base_worker_method(hook_self, attr):
        @wraps(attr)
        def overloaded_attr(self_torch, *args, **kwargs):
            ptr = hook_self.local_worker.send_command(
                recipient=self_torch.worker(),
                cmd_name=f"{'torch'}.{attr}",
                args_=args,
                kwargs_=kwargs,
            )

            return ptr.wrap()

        return overloaded_attr

    def _hook_additive_shared_tensor_methods(self):
        """
        Add hooked version of all methods of the torch Tensor to the
        Additive Shared tensor: instead of performing the native tensor
        method, it will be forwarded to each share when it is relevant
        """

        tensor_type = self.torch.Tensor
        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(AdditiveSharingTensor):
                new_method = self._get_hooked_additive_shared_method(attr)
                setattr(AdditiveSharingTensor, attr, new_method)

    def _hook_parameters(self):
        """
        This method overrides the torch Parameter class such that
        it works correctly with our overridden tensor types. The
        native torch Parameter class kept deleting all of our
        attributes on our custom tensors, so we wrote our own.
        """

        # Hook __new__ to handle when non-pure torch tensors are given as data attribute

        def hooked__new__(cls, data=None, requires_grad=True):
            if data is None:
                data = torch.Tensor()

            # If data is not a pure torch tensor you need to store the chain in a
            # specific place otherwise it will get deleted
            if not isinstance(data, torch.Tensor) or hasattr(data, "child"):
                p = torch.Tensor._make_subclass(cls, torch.Tensor(), requires_grad)
                if isinstance(data, torch.Tensor):  # so it's a wrapper: remove it
                    p.child = data.child
                else:
                    p.child = data
            else:
                p = torch.Tensor._make_subclass(cls, data, requires_grad)

            return p

        torch.nn.Parameter.__new__ = hooked__new__

        # Hook __repr__ to handle chain repr when needed

        torch.nn.Parameter.native_param___repr__ = torch.nn.Parameter.__repr__

        def hooked__repr__(self):
            if hasattr(self, "child"):
                return "Parameter containing:\n" + self.child.__repr__()
            else:
                return self.native_param___repr__()

        # torch.nn.Parameter.__repr__ = hooked__repr__

        def get_data(self):
            if hasattr(self, "child"):
                to_return = self.child.attr("data")
            else:
                to_return = self.native_data

                # good to ensure that the ID stays consistent
                # not 100% this is required but it's at least
                # good practice
                try:
                    to_return.id = self.data_id
                except AttributeError:
                    self.data_id = to_return.id

            return to_return

        def set_data(self, new_data):
            # If data is not a pure torch tensor you need to store the chain in a
            # specific place otherwise it will get deleted
            if not isinstance(new_data, torch.Tensor) or hasattr(new_data, "child"):
                self.child = new_data  # .wrap()
            else:
                if hasattr(self, "child"):
                    del self.child

                with torch.no_grad():
                    self.native_data = new_data
            return self

        torch.nn.Parameter.data = property(fget=get_data, fset=set_data)

        # Hook .grad to handle chain assignment when needed

        torch.nn.Parameter.native_param_grad = torch.nn.Parameter.grad

        @property
        def grad(self):

            if hasattr(self, "child"):
                to_return = self.child.attr("grad")
                if to_return is not None and isinstance(to_return.child, PointerTensor):
                    if to_return.child.is_none():
                        to_return = None

            else:
                to_return = self.native_param_grad

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
                if self.native_param_grad is not None:
                    with torch.no_grad():
                        self.native_param_grad = new_grad
                elif new_grad is not None:
                    self.native_param_grad = new_grad
            return self

        torch.nn.Parameter.grad = grad

    def _hook_torch_module(self):
        """Overloads functions in the main torch modules.
        The way this is accomplished is by first moving all existing module
        functions in the torch module to native_<function_name_here>.

        Example:
            the real :func:`torch.cat` will become :func:`torch.native_cat`
            and :func:`torch.cat` will have our hooking code.
        """
        torch_modules = syft.torch.torch_modules

        for module_name, torch_module in torch_modules.items():
            for func in dir(torch_module):

                # Some functions we want to ignore (not override). Such functions have been hard
                # coded into the torch_attribute exclude (see TorchAttribute class)
                if func in syft.torch.exclude:
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
                if "native_" in func or f"native_{func}" in dir(torch_module):
                    continue

                self._perform_function_overloading(module_name, torch_module, func)

    def _get_hooked_additive_shared_method(hook_self, attr):
        """
        Hook a method to send it multiple remote workers

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        def dispatch(args_, k):
            return map(lambda x: x[k] if isinstance(x, dict) else x, args_)

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

    def _hook_tensor(hook_self):
        """Hooks the function torch.tensor()
        We need to do this seperately from hooking the class because internally
        torch does not pick up the change to add the args
        Args:
            hook_self: the hook itself
        """

        if "native_tensor" not in dir(hook_self.torch):
            hook_self.torch.native_tensor = hook_self.torch.tensor

        def new_tensor(*args, owner=None, id=None, register=True, **kwargs):
            current_tensor = hook_self.torch.native_tensor(*args, **kwargs)
            _apply_args(hook_self, current_tensor, owner, id)
            if register:
                current_tensor.owner.register_obj(current_tensor)

            return current_tensor

        hook_self.torch.tensor = new_tensor

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
            # FIXME it now overwritten in native.py to use torch.eq, because
            # of pb between == & __eq__ See #2030
            # "__eq__",
            "__gt__",
            "__ge__",
            "__lt__",
            "__le__",
        ]
        cls._transfer_methods_to_framework_class(tensor_type, syft_type, exclude)

    def _hook_module(self):
        """Overloading torch.nn.Module with PySyft functionality, the primary module
        responsible for core ML functionality such as Neural network layers and
        loss functions.

        It is important to note that all the operations are actually in-place.
        """
        self.element_iter_dict = {}

        def register_element_iterator(name, func):
            """register an internal element buffer iterator"""
            if name in self.element_iter_dict.keys():
                return
            self.element_iter_dict[name] = func

        def tensor_iterator(nn_self):
            """adding relavant iterators for the tensor elements"""
            iterators = [
                "parameters",
                "buffers",
            ]  # all the element iterators from nn module should be listed here,
            return [getattr(nn_self, iter) for iter in iterators]

        def module_is_missing_grad(model):
            """Checks if all the parameters in the model have been assigned a gradient"""
            for p in model.parameters():
                if p.grad is None:
                    return True
            return False

        def create_grad_objects(model):
            """Assigns gradient to model parameters if not assigned"""
            for p in model.parameters():
                if p.requires_grad:  # check if the object requires a grad object
                    o = p.sum()
                    o.backward()
                    if p.grad is not None:
                        p.grad -= p.grad

        def module_send_(nn_self, *dest, force_send=False, **kwargs):
            """Overloads torch.nn instances so that they could be sent to other workers"""

            if module_is_missing_grad(nn_self):
                create_grad_objects(nn_self)

            for element_iter in tensor_iterator(nn_self):
                for p in element_iter():
                    p.send_(*dest, **kwargs)

            if isinstance(nn_self.forward, Plan):
                nn_self.forward.send(*dest, force=force_send)

            return nn_self

        self.torch.nn.Module.send = module_send_
        self.torch.nn.Module.send_ = module_send_

        def module_move_(nn_self, destination):

            for element_iter in tensor_iterator(nn_self):
                for p in element_iter():
                    p.move_(destination)

        self.torch.nn.Module.move = module_move_

        # def module_end_get_(nn_self):
        #     """Overloads send to remote for torch.nn.Module."""
        #     if module_is_missing_grad(nn_self):
        #         create_grad_objects(nn_self)
        #
        #     for p in nn_self.parameters():
        #         p.end_get()
        #
        #     return nn_self
        #
        # self.torch.nn.Module.end_get = module_end_get_
        #
        # def module_move_(nn_self, dest):
        #     return nn_self.send(dest).end_get()
        #
        # self.torch.nn.Module.move = module_move_

        def module_get_(nn_self):
            """
            overloads torch.nn instances with get method so that parameters
            could be sent back to owner
            """
            for element_iter in tensor_iterator(nn_self):
                for p in element_iter():
                    p.get_()

            if isinstance(nn_self.forward, Plan):
                nn_self.forward.get()

            return nn_self

        self.torch.nn.Module.get_ = module_get_
        self.torch.nn.Module.get = module_get_

        def module_encrypt_(nn_self, **kwargs):
            """Overloads fix_precision for torch.nn.Module."""
            if module_is_missing_grad(nn_self):
                create_grad_objects(nn_self)

            for element_iter in tensor_iterator(nn_self):
                for p in element_iter():
                    p.encrypt(inplace=True, **kwargs)

            return nn_self

        self.torch.nn.Module.encrypt_ = module_encrypt_
        self.torch.nn.Module.encrypt = module_encrypt_

        def module_decrypt_(nn_self):
            """Overloads fix_precision for torch.nn.Module."""
            if module_is_missing_grad(nn_self):
                create_grad_objects(nn_self)

            for element_iter in tensor_iterator(nn_self):
                for p in element_iter():
                    p.decrypt(inplace=True)

            return nn_self

        self.torch.nn.Module.decrypt_ = module_decrypt_
        self.torch.nn.Module.decrypt = module_decrypt_

        def module_share_(nn_self, *args, **kwargs):
            """Overloads fix_precision for torch.nn.Module."""
            if module_is_missing_grad(nn_self):
                create_grad_objects(nn_self)

            for element_iter in tensor_iterator(nn_self):
                for p in element_iter():
                    p.share_(*args, **kwargs)

            return nn_self

        self.torch.nn.Module.share_ = module_share_
        self.torch.nn.Module.share = module_share_

        def module_fix_precision_(nn_self, *args, **kwargs):
            """Overloads fix_precision for torch.nn.Module."""
            if module_is_missing_grad(nn_self):
                create_grad_objects(nn_self)

            for element_iter in tensor_iterator(nn_self):
                for p in element_iter():
                    p.fix_precision_(*args, **kwargs)

            return nn_self

        self.torch.nn.Module.fix_precision_ = module_fix_precision_
        self.torch.nn.Module.fix_precision = module_fix_precision_
        self.torch.nn.Module.fix_prec = module_fix_precision_

        def module_float_precision_(nn_self):
            """Overloads float_precision for torch.nn.Module, convert fix_precision
            parameters to normal float parameters"""
            # TODO: add .data and .grad to syft tensors
            # if module_is_missing_grad(nn_self):
            #    create_grad_objects(nn_self)

            for element_iter in tensor_iterator(nn_self):
                for p in element_iter():
                    p.float_precision_()

            return nn_self

        self.torch.nn.Module.float_precision_ = module_float_precision_
        self.torch.nn.Module.float_precision = module_float_precision_
        self.torch.nn.Module.float_prec = module_float_precision_

        def module_copy(nn_self):
            """Returns a copy of a torch.nn.Module"""
            return copy.deepcopy(nn_self)

        self.torch.nn.Module.copy = module_copy

        @property
        def owner(nn_self):
            for p in nn_self.parameters():
                return p.owner

        self.torch.nn.Module.owner = owner

        @property
        def location(nn_self):
            try:
                for p in nn_self.parameters():
                    return p.location
            except AttributeError:
                raise AttributeError(
                    "Module has no attribute location, did you already send it to some location?"
                )

        self.torch.nn.Module.location = location

        # Make sure PySyft uses the PyTorch version
        self.torch.nn.modules.rnn._rnn_impls["LSTM"] = self.torch.lstm

        # Add support for GRUs
        self.torch.nn.modules.rnn._rnn_impls["GRU"] = self.torch.gru

        # Override _VF.LSTM_Cell and _VF.GRU_Cell with torch.LSTM_Cell and torch.GRU_Cell
        # With the pytorch-based version
        self.torch.nn.modules.rnn._VF = self.torch

    def _hook_optim(self):
        """Overloading torch.optim.Optimizer with PySyft functionality. Optimizer
        hyper-parameters should indeed be converted to fixed precision to interact
        with fixed precision or additive shared tensors.

        It is important to note that all the operations are actually in-place.
        """

        def optim_fix_precision_(optim_self, *args, **kwargs):
            """Overloads fix_precision for torch.optim.Optimizer"""

            for param_group in optim_self.param_groups:
                for key, param in param_group.items():
                    if isinstance(param, (float, int, bool)) and param != 0 and key != "params":
                        param_group[key] = torch.tensor(param).fix_precision(*args, **kwargs).child

            return optim_self

        self.torch.optim.Optimizer.fix_precision = optim_fix_precision_

        def optim_float_precision_(optim_self):
            """Overloads float_precision for torch.optim.Optimizer, convert fix_precision
            hyper-parameters to normal float values"""

            for param_group in optim_self.param_groups:
                for key, param in param_group.items():
                    if isinstance(param, FixedPrecisionTensor) and key != "params":
                        param_group[key] = param.float_precision().item()

            return optim_self

        self.torch.optim.Optimizer.float_precision = optim_float_precision_

        # Modification of torch/nn/utils/clip_grad.py. The plain PyTorch method was not compatible
        # with PySyft remote tensors, so this method adds support for gradient clipping of remote
        # tensors, and keeps functionalities from PyTorch to clip local PyTorch tensors.
        def clip_grad_norm_remote_(parameters, max_norm, norm_type=2):
            """Clips gradient norm of an iterable of parameters stored over a remote model

            The norm is computed over all gradients together, as if they were
            concatenated into a single vector. Gradients are modified in-place.

            Arguments:
                - parameters (Iterable[Tensor] or Tensor): an iterable of PySyft remote
                Tensors or PyTorch tensor will have gradients normalized or a single
                PySyfy / PyTorch tensor.
                - max_norm (float or int): max norm of the gradients
                - worker: The worker where the parameters are hosted and where the gradient clipping
                will be performed
                - norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                    infinity norm.

            Returns:
                Total norm of the parameters (viewed as a single vector).
            """

            def param_is_pointer_tensor(param):
                """
                A list of parameters is remote if all params contained in the list are
                remote (i.e., the child of each param is a pointer tensor).
                This method checks if a single param is indeed remote, so whether
                the child of a parameter is a pointer tensor
                """
                return hasattr(param, "child") and isinstance(param.child, PointerTensor)

            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]

            parameters = list(filter(lambda p: p.grad is not None, parameters))
            max_norm = float(max_norm)
            norm_type = float(norm_type)
            if norm_type == inf:
                total_norm = max(p.grad.data.abs().max() for p in parameters)
            else:
                # all parameters are remote
                if all(param_is_pointer_tensor(param) for param in parameters):
                    total_norm = torch.zeros(1)
                    # Let's send the total norm over to the remote where the remote tensor is
                    total_norm = total_norm.send(parameters[0].location)
                else:
                    total_norm = 0
                for p in parameters:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type

                total_norm = total_norm ** (1.0 / norm_type)
            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in parameters:
                    p.grad.data.mul_(clip_coef)
            return total_norm

        self.torch.nn.utils.clip_grad_norm_ = clip_grad_norm_remote_

    def set_verbose(self, flag):
        for workers in self._syft_workers:
            workers.verbose = flag
