"""Hooks which override deep learning interfaces with remote execution functionality."""

import torch
import inspect
import re
import json
import types
import functools
import importlib
from . import workers
from . import utils


class BaseHook(object):
    r""" A abstract interface for deep learning framework hooks."""

    def __init__(self):
        ""
    def __enter__(self):
        ""
    def __exit__(self):
        ""


class TorchHook(BaseHook):
    r""" A Hook which Overrides Methods on PyTorch Variables & Tensors -
    **Currently compatible with PyTorch 0.3.1**

    The purpose of this class is to:

        * extend torch methods to allow for the moving of tensors
          and variables from one worker to another
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
          as they occur. (Defalt: True)

    :Example:

    >>> from syft.core.hooks import TorchHook
    >>> from syft.core.hooks import torch
    >>> hook = TorchHook()
    Hooking into Torch...
    Overloading Complete.
    >>> x = torch.FloatTensor([1,2,3,4,5])
    >>> x
     1
     2
     3
     4
     5
    [torch.FloatTensor of size 5]
    """

    def __init__(self, local_worker=None, is_client=True, verbose=True):
        super().__init__()

        self.local_worker = local_worker
        if(self.local_worker is None):

            # Every TorchHook instance should have a local worker which is responsible for
            # interfacing with other workers. The worker interface is what allows the Torch
            # specific code in TorchHook to be agnostic to the means by which workers communicate
            # (such as peer-to-peer, sockets, through local ports, or all within the same process)

            if(hasattr(torch, 'local_worker')):
                self.local_worker = torch.local_worker
                if(verbose):
                    print("Torch seems to already have a local_worker object... \
                          using that one instead...")
            else:
                self.local_worker = workers.VirtualWorker(hook=self, is_client_worker=is_client)

        torch.local_worker = self.local_worker

        # this is a list of all module functions in the torch module
        self.torch_funcs = dir(torch)

        # this is the list of torch tensor types that we will override for remote execution
        self.tensor_types = [torch.FloatTensor,
                             torch.DoubleTensor,
                             torch.HalfTensor,
                             torch.ByteTensor,
                             torch.CharTensor,
                             torch.ShortTensor,
                             torch.IntTensor,
                             torch.LongTensor]

        # this is the list of torch tensor VARIABLE types that we will override for remote execution
        # Variables are simply tensors that are differentiable (support gradients)
        # Parameters are Variables that are also weights in a neural model
        self.var_types = [torch.autograd.variable.Variable, torch.nn.Parameter]

        # a list of all classes in which we will override their methods for remote execution
        self.tensorvar_types = self.tensor_types + [torch.autograd.variable.Variable]
        self.tensorvar_types_strs = [x.__name__ for x in self.tensorvar_types]
        self.tensorvar_methods = list(
            set(
                [method
                    for tensorvar in self.tensorvar_types
                    for method in dir(tensorvar)]
                )
            )

        # Methods that caused infinite recursion during testing
        # TODO: May want to handle the ones in "exclude" manually at
        #       some point
        self.exclude = (['ndimension', 'nelement', 'size', 'numel',
                         'type', 'tolist', 'dim', '__iter__', 'select'])

        # This one wasn't in dir(Variable) -- probably a C++ thing
        self.var_exclude = ['__getattr__']

        # Torch functions we don't want to override
        self.torch_exclude = ['save', 'load', 'typename']

        # Safety checks for serializing and deserializing torch objects
        # Desperately needs stress testing before going out in the wild
        self.map_tensor_type = {
            'torch.FloatTensor': torch.FloatTensor,
            'torch.DoubleTensor': torch.DoubleTensor,
            'torch.HalfTensor': torch.HalfTensor,
            'torch.ByteTensor': torch.ByteTensor,
            'torch.CharTensor': torch.CharTensor,
            'torch.ShortTensor': torch.ShortTensor,
            'torch.IntTensor': torch.IntTensor,
            'torch.LongTensor': torch.LongTensor
        }
        self.map_var_type = {
            'torch.autograd.variable.Variable': torch.autograd.variable.Variable,
            'torch.nn.parameter.Parameter': torch.nn.parameter.Parameter
        }
        self.map_torch_type = dict(self.map_tensor_type, **self.map_var_type)

        if(not hasattr(torch, 'hooked')):
            if(verbose):
                print('Hooking into Torch...')
            self._hook_torch_module()
            for t_type in self.tensor_types:
                self._hook_tensor(t_type)
            self._hook_variable()
            torch.hooked = True
            if(verbose):
                print('Overloading complete.')
        else:
            if(verbose):
                print("WARNING: Torch seems to be already overloaded... skipping...")

    def __enter__(self):
        pass

    def __exit__(self):
        importlib.reload(torch)

    def types_guard(self, torch_type_str):
        """types_guard(torch_type_str) -> torch.Tensor or torch.autograd.Variable

        This method converts strings into a type reference. This prevents
        deserialized JSON from being able to instantiate objects of arbitrary
        type which would be a security concern.

        :Parameters:

        * **torch_type_str (string)** A string representing the type of object that is
          to be returned.

        * **out (a torch type)** The type the string refersto (if it's present in the
          acceptible list self.map_torch_type)

        :Example:

        >>> from syft.core.hooks import TorchHook
        >>> hook = TorchHook()
        Hooking into Torch...
        Overloading Complete.
        >>> torch_type = hook.types_guard('torch.FloatTensor')
        >>> x = torch_type([1,2,3,4,5])
        >>> x
         1
         2
         3
         4
         5
        [torch.FloatTensor of size 5]
        """
        try:
            return self.map_torch_type[torch_type_str]
        except KeyError:
            raise TypeError(
                "Tried to receive a non-Torch object of type {}.".format(
                    torch_type_str))

    def tensor_contents_guard(self, contents):
        """tensor_contents_guard(contents) -> contents
        TODO: check to make sure the incoming list isn't dangerous to use for
               constructing a tensor (likely non-trivial). Accepts the list of JSON objects
               and returns the list of JSON ojects. Should throw and exception if there's a
               security concern.
        """
        return contents

    @staticmethod
    def get_owners(tensorvars):
        """get_owners(tensorvars) -> (bool, list(BaseWorker))
        A static utility method which returns owners given a list of tensors.

        :Parameters:

        * **tensorvars (list)** A list of :class:`torch.Tensor`
          or :class:`torch.autograd.Variable` objects which
          you want to know the owners of. Can pass in an empty
          list without breaking (but will also return an empty list
          of owners)

        * **out (bool, list(BaseWorker))** a tuple where the
          first value is a boolean indicating whether there are multiple
          owners. The second object is a list of owners, where
          each item in the list can be either an id of the owner or
          a :class:`.workers.BaseWorker` pointer.

        :Example:

        >>> from syft.core.hooks import TorchHook
        >>> from syft.core.hooks import torch
        >>> from syft.core.workers import VirtualWorker
        >>> from torch.autograd import Variable as Var
        >>> hook = TorchHook()
        Hooking into Torch...
        Overloading Complete.
        >>> local = hook.local_worker
        >>> remote = VirtualWorker(id=1, hook=hook)
        >>> local.add_worker(remote)
        >>> x = torch.FloatTensor([1,2,3,4,5])
        >>> y = torch.FloatTensor([2,4,6,8,10]).send(remote)
        >>> z = Var(x)
        >>> (is_multiple_owners, owners) = hook.get_owners([x,y,z])
        >>> print(is_multiple_owners)
        True
        >>> print(owners)
        [0, <syft.core.workers.VirtualWorker at 0x1109ef550>]
        """

        # Note from Andrew: This feels like a strange method since it
        # returns the superset of owners across all
        # tensors. I'm surprised that the logic consuming this
        # method doesn't have bugs.

        owners = list(set([owner for tensorvar in tensorvars for owner in tensorvar.owners]))
        multiple_owners = len(owners) > 1
        return multiple_owners, owners

    def _hook_torch_module(self):
        """
        Overloads functions in the main torch module.

        The way this is accomplished is by first moving all existing module functions in the torch
        module to old_<function_name_here>. Thus, the real :func:`torch.cat` will become
        :func:`torch.old_cat` and :func:`torch.cat` will have our hooking code. Generically,
        this hooking code checks to see if the tensor is on the current worker (aka, we can read it)
        or on a remote opne (and we only have a pointer). If the data is local, then the method
        will simply execute :func:`torch.old_cat`, but if the data for a tensor is remote, then
        it will instead send a message to the remote machine instructing it to perform an arbitrary
        command (:func:`torch.old_cat` on the remote machine).
        """

        for attr in self.torch_funcs:

            # Some functions we want to ignore (not override). Such functions have been hard coded
            # into the attribute self.torch_exclude
            if attr in self.torch_exclude:
                continue

            # if we haven't already overloaded this function
            if 'old_{}'.format(attr) in dir(torch):
                continue

            # if we haven't already overloaded this function (redundancy allowed)
            if 'old_' in attr:
                continue

            # Where the overloading happens
            lit = getattr(torch, attr)
            if (type(lit) in [types.FunctionType, types.BuiltinFunctionType]):

                passer = self._pass_func_args(lit)
                new_attr = self._overload_function(passer)
                setattr(torch, 'old_{}'.format(attr), lit)
                setattr(torch, attr, new_attr)

    @staticmethod
    def _pass_func_args(func):
        """Wrapper gathering partial object from function call."""

        @functools.wraps(func)
        def pass_args(*args, **kwargs):
            # Return a new partial object which when called will behave like func called with the
            # positional arguments args and keyword arguments keywords. If more arguments are
            # supplied to the call, they are appended to args. If additional keyword arguments
            # are supplied, they extend and override keywords.

            # The partial() is used for partial function application which “freezes” some
            # portion of a function’s arguments and/or keywords resulting in a new object
            # with a simplified signature.
            return functools.partial(func, *args, **kwargs)
        return pass_args

    def _overload_inner(hook_self, part, has_self=True):
        """
        Performs the actual method/function overloading.
        Note that this is the method/function agnostic
        piece and is only called from within _overload_method
        and _overload_function
        """

        # Step 1: Compiles Command
        command = hook_self._compile_command(part, has_self=has_self)

        # Step 2: checks for Tensors and Variables in the args/kwargs
        tensorvars = hook_self._get_tensorvars(command)

        # Step 3: Checks to see if the tensor is local (on this machine) or is a pointer
        # to a remote one (on a different machine)
        has_remote = hook_self._check_remote(tensorvars)

        # Checks to see if the tensor has multiple owners (not yet fully supported func)
        multiple_owners, owners = hook_self.get_owners(tensorvars)

        # If one of the tensor arguments is remote
        # if the tensor only has one owner (remote)
        if has_remote and not multiple_owners:

            command = hook_self._replace_in_command(command)

            for worker in owners:

                _pr = hook_self.local_worker.process_response
                response = hook_self.local_worker.request_response(recipient=worker,
                                                                   message=command,
                                                                   response_handler=_pr)

                registration, torch_type, var_data, var_grad = response

                if registration is None:
                    return var_data, has_remote, multiple_owners
                # only returns last pointer, since tensors will
                # be identical across machines for right now
                pointer = hook_self._assemble_result_pointer(worker,
                                                             registration,
                                                             torch_type,
                                                             var_data,
                                                             var_grad)
                return pointer, has_remote, multiple_owners

        elif(has_remote and multiple_owners):
            raise NotImplementedError("""MPC not yet implemented:
                Torch objects need to be on the same machine in
                order to compute with them.""")

        return (None, has_remote, multiple_owners)

    def _overload_method(hook_self, method, isfunc=False):
        """
        Wrapper overloading partialmethod objects of Torch object
        methods.  Compiles command, checks for Tensors and Variables in
        the args/kwargs, determines locations of all Tensors and
        Variables involved in computation, and handles the computation
        accordingly.
        """
        @functools.wraps(method)
        def send_to_workers(self, *args, **kwargs):
            """
            This method is responsible for sending a command executed
            \on a client to a worker to be performed.
            """
            part = method(self, *args, **kwargs)
            if self.is_pointer:
                return hook_self._overload_inner(part, has_self=True)[0]
            else:
                result = part.func(self, *args, **kwargs)
                if (type(result) in hook_self.tensorvar_types and (not hasattr(result, 'owner'))):
                    result = hook_self.local_worker.register_object(hook_self.local_worker, result,
                                                                    is_pointer=False)
                return result
        return send_to_workers

    def _overload_function(hook_self, func):

        """
        Wrapper overloading partial objects of functions in the torch
        module.  Compiles command, checks for Tensors and Variables in
        the args/kwargs, determines locations of all Tensors and
        Variables involved in computation, and handles the computation
        accordingly.
        """
        @functools.wraps(func)
        def send_to_workers(*args, **kwargs):
            part = func(*args, **kwargs)

            pointer, has_remote, multiple_owners = hook_self._overload_inner(part, has_self=False)

            if not (has_remote and not multiple_owners):
                result = part.func(*args, **kwargs)
                if type(result) in hook_self.tensorvar_types:
                    result = hook_self.local_worker.register_object(hook_self.local_worker,
                                                                    result, is_pointer=False)
                return result

        return send_to_workers

    @staticmethod
    def _compile_command(partial_func, has_self):
        """
        Assembles a JSON-serializable message from a partial function.

        Args:
        partial_func: a functools.partial or functools.partialmethod
            object wrapped around a torch command, its args, and its
            kwargs.
        has_self: a flag for whether or not the function is a method.
        """
        func = partial_func.func
        args = partial_func.args
        kwargs = partial_func.keywords
        command = {}
        command['has_self'] = has_self
        if has_self:
            command['self'] = args[0]
            args = args[1:]
        command['command'] = func.__name__
        command['args'] = args
        command['kwargs'] = kwargs
        command['arg_types'] = [type(x).__name__ for x in args]
        command['kwarg_types'] = [type(kwargs[x]).__name__ for x in kwargs]
        return command

    def _get_tensorvars(self, command):
        """Returns all Tensors and Variables in the args/kwargs of the command"""

        args = command['args']
        kwargs = command['kwargs']
        arg_types = command['arg_types']
        kwarg_types = command['kwarg_types']
        tensorvar_args = [args[i]
                          for i in range(len(args)) if arg_types[i] in self.tensorvar_types_strs]
        tensorvar_kwvals = [kwargs[i][1] for i in range(len(kwargs))
                            if kwarg_types[i] in self.tensorvar_types_strs]
        if command['has_self']:
            tensorvar_args.insert(0, command['self'])
        return tensorvar_args + tensorvar_kwvals

    @staticmethod
    def _check_remote(tensorvars):
        """Checks to see if the tensor is local (on this machine) or is a pointer
        to a remote one (on a different machine)"""

        return any([tensorvar.is_pointer for tensorvar in tensorvars])

    @staticmethod
    def _replace_in_command(command_msg):
        command_msg['args'] = utils.map_tuple(
            None, command_msg['args'], TorchHook._replace_tensorvar)
        command_msg['kwargs'] = utils.map_dict(
            None, command_msg['kwargs'], TorchHook._replace_tensorvar)
        try:
            command_msg['self'] = TorchHook._replace_tensorvar(command_msg['self'])
        except KeyError:
            pass
        return command_msg

    @staticmethod
    def _replace_tensorvar(x):
        """This method takes a tensor/var/param and replaces it with a
        string containing it's ID and special flag for recognizing that
        it's a tensor type arg instead of a string.

        This method also works for an iterable of tensors (e.g. `torch.cat([x1, x2, x3])`)
        """
        if hasattr(torch, 'old_is_tensor'):
            check = torch.old_is_tensor
        else:
            check = torch.is_tensor
        try:
            _is_param = isinstance(x, torch.nn.Parameter)
            if check(x) or isinstance(x, torch.autograd.Variable) or _is_param:
                return '_fl.{}'.format(x.id)
            else:
                [TorchHook._replace_tensorvar(i) for i in x]
        except (AttributeError, TypeError):
            return x

    def _assemble_result_pointer(self, worker, registration, torch_type, var_data, var_grad):
        """
        Assembles a pointer to a remote Torch object. Pointers feel like
        real Torch objects, but they're zero-dimensional until their
        contents are retrieved from their owners.

        Args
        registration (dict): registration attributes for the pointer
        torch_type: the torch class to construct the pointer from
        """
        # TODO: extend to iterables of tensor pointers
        try:
            torch_type = self.map_torch_type[torch_type]
        except KeyError:
            raise TypeError(
                "Tried to receive a non-Torch object of type {}.".format(
                    torch_type))

        if var_data is not None:
            data = self._assemble_result_pointer(worker, **var_data)
            data = self.local_worker.register_object(worker, data, **var_data['registration'])
        elif torch_type in self.var_types:
            data = torch.Tensor(0)
        else:
            data = 0
        result = torch_type(data)
        if var_grad is not None:
            # grad = self.assemble_result_pointer(**var_grad)
            self.local_worker.register_object(worker, result.grad, **var_grad['registration'])
        return self.local_worker.register_object(self.local_worker, result, **registration)

    def _hook_tensor(self, tensor_type):
        """Overloading a given tensor_type"""
        # Overload 'special' methods here
        self._hook_tensor___init__(tensor_type)
        # self.hook_tensor___del__(tensor_type)
        self._hook_tensor___new__(tensor_type)
        self._hook_tensor___repr__(tensor_type)

        for attr in dir(tensor_type):
            # if we haven't already overloaded this function
            if 'old_{}'.format(attr) not in dir(tensor_type):
                # Conditions for inclusion/exclusion
                if attr in self.exclude:
                    continue
                lit = getattr(tensor_type, attr)
                is_base = attr in dir(object)
                is_desc = inspect.ismethoddescriptor(lit)
                is_func = isinstance(lit, types.FunctionType)
                try:
                    is_service_func = 'HookService' in lit.__qualname__
                except:
                    is_service_func = False
                is_old = re.match('old*', attr) is not None

                # Where the overloading happens
                if ((is_desc or (is_func and not is_service_func)) and not is_base and not is_old):
                    passer = self._pass_method_args(lit)
                    new_attr = self._overload_method(passer)
                    setattr(tensor_type, 'old_{}'.format(attr), lit)
                    setattr(tensor_type, attr, new_attr)

        # Add in our own Grid-specific methods
        self._hook_tensor_send_(tensor_type)
        self._hook_get_(tensor_type)
        self._hook_tensor__ser(tensor_type)

    # Special Tensor method HookService
    def _hook_tensor___init__(hook_self, tensor_type):
        """Overload tensor_type.__init__"""

        def new___init__(self, *args):
            super(tensor_type, self).__init__()
            self = hook_self.local_worker.register_object(hook_self.local_worker,
                                                          self, is_pointer=False)

        tensor_type.__init__ = new___init__

    def _hook_tensor___del__(hook_self, tensor_type):
        def new____del__(self, *args):
            print("deleting tensor")

        tensor_type.__del__ = new____del__

    def _hook_tensor___new__(hook_self, tensor_type):
        """Overload tensor_type.__new__"""

        if('old___new__' not in dir(tensor_type)):
            tensor_type.old___new__ = tensor_type.__new__

            def new___new__(cls, *args, **kwargs):
                result = cls.old___new__(cls, *args,  **kwargs)
                result = hook_self.local_worker.register_object(
                         hook_self.local_worker, result, is_pointer=False)
                return result

            tensor_type.__new__ = new___new__

    def _hook_tensor___repr__(hook_self, tensor_type):
        """Overload tensor_type.__repr__"""
        if('old__repr__' not in dir(tensor_type)):
            tensor_type.old__repr__ = tensor_type.__repr__

            def new___repr__(self):
                _id_in_owners = hook_self.local_worker.id in self.owners
                if (hook_self.local_worker in self.owners or _id_in_owners):
                    return self.old__repr__()
                else:
                    return "[{}.{} - Locations:{}]".format(
                        tensor_type.__module__,
                        tensor_type.__name__,
                        self.owners)

            tensor_type.__repr__ = new___repr__

    @staticmethod
    def _pass_method_args(method):
        """Wrapper gathering partialmethod object from method call."""
        @functools.wraps(method)
        def pass_args(*args, **kwargs):
            return functools.partialmethod(method, *args, **kwargs)
        return pass_args

    # Grid-specific method hooking
    def _hook_tensor_send_(hook_self, tensor_type):
        def send_(self, workers):
            """
            Sends a Tensor object to a (sequence of) Grid workers.

            Args:
            workers: string (or sequence) containing IPFS address(es)
                of worker node(s).
            """

            # makes singleton, if needed
            workers = hook_self.local_worker._check_workers(self, workers)
            self = hook_self.local_worker.register_object(hook_self.local_worker, obj=self,
                                                          id=self.id, owners=workers)
            for worker in workers:
                # TODO: sync or async? likely won't be worth doing async,
                #       but should check (low priority)
                hook_self.local_worker.send_obj(self, worker)
            self = hook_self.local_worker.register_object(hook_self.local_worker,
                                                          self.old_set_(tensor_type(0)),
                                                          id=self.id, owners=workers,
                                                          is_pointer=True)
            return self
        setattr(tensor_type, 'send_', send_)
        setattr(tensor_type, 'send', send_)

    def _hook_get_(hook_self, torch_type):
        def get_(self, reduce=lambda x: x[0]):
            """
            Gets a Torch object from its current owners.

            Args:
            reduce: (EXPERIMENTAL) How to reduce tensors that come from
                multiple workers
            """
            # TODO: fully generalize this to multiple workers; consider
            #       adding arguments for other tensor ids, e.g. mapping workers
            #       to tensors, and a reduce function (for example, would allow
            #       for built-in gradient averaging when Variable.get is done)
            #       (low priority)
            try:
                assert len(self.owners) == 1
            except AssertionError:
                raise NotImplementedError('Only able to get_ tensors belonging \
                                            to a single worker right now.')
            if hook_self.local_worker.id in self.owners:
                return self

            _out = hook_self.local_worker.request_obj(obj_id=self.id,
                                                      sender=self.owners[0])
            x, request_obj_cleanup_method = _out

            hook_self.local_worker.register_object(hook_self.local_worker, x, id=x.id)

            # if self == tensor
            _id = hook_self.local_worker.id  # for brevity
            if(type(self) != torch.autograd.variable.Variable):
                _os = self.old_set_(x.type(self.type()))
                self = hook_self.local_worker.register_object(hook_self.local_worker,
                                                              _os,
                                                              id=self.id, owners=[_id])

            else:

                _os = self.old_set_(x.type(self.data.type()))  # for brevity
                self = hook_self.local_worker.register_object(hook_self.local_worker,
                                                              _os,
                                                              id=self.id, owners=[_id])

                self.data = hook_self.local_worker.register_object(hook_self.local_worker,
                                                                   x.data,
                                                                   id=x.data.id,
                                                                   owners=[_id])
                if(x.grad is not None):
                    self.grad = hook_self.local_worker.register_object(hook_self.local_worker,
                                                                       x.grad,
                                                                       id=x.grad.id,
                                                                       owners=[_id])

            """for some reason, when retuning obj from request_obj
            method (above), the gradient gets re-initialized without
            being re-registered and as a consequence does not have an
            id, causing the last register_object above to fail
            because x.grad.id does not exist. As a result, we've needed
            to register objects temporarily which seems to
            fix it. Super strange bug which took multiple days to figure
            out. The true cause is still unknown but this
            workaround seems to work well for now. Anyway, we don't need
            the temporary objects past this point.
            request_obj_cleanup_method()"""
            return self

        setattr(torch_type, 'get_', get_)

        # TODO: make this a non-inline version
        setattr(torch_type, 'get', get_)

    def _hook_tensor__ser(hook_self, tensor_type):

        def ser(self, include_data=True):
            """Serializes a {} object to JSON.""".format(tensor_type)
            tensor_msg = {}
            tensor_msg['torch_type'] = self.type()
            if include_data:
                tensor_msg['data'] = self.tolist()
            tensor_msg['id'] = self.id
            if(type(self.owners[0]) is int):
                tensor_msg['owners'] = self.owners
            else:
                tensor_msg['owners'] = list(map(lambda x: x.id, self.owners))
            tensor_msg['is_pointer'] = not include_data

            return json.dumps(tensor_msg)

        def deser(self, obj_msg):

            # this could be a significant failure point, security-wise
            data = hook_self.tensor_contents_guard(obj_msg['data'])
            v = self(data)
            return v

        tensor_type.ser = ser
        tensor_type.deser = deser

    def _hook_variable(self):
        # Overload 'special' methods here
        self._hook_var___new__()
        self._hook_var_contents()

        for attr in dir(torch.autograd.variable.Variable):

            # Conditions for inclusion/exclusion
            if attr in self.exclude + self.var_exclude:
                continue
            lit = getattr(torch.autograd.variable.Variable, attr)
            is_base = attr in dir(object)
            is_desc = inspect.ismethoddescriptor(lit)
            # is_func = isinstance(type(lit), types.FunctionType)
            is_func = isinstance(lit, types.FunctionType)
            try:
                is_service_func = 'HookService' in lit.__qualname__
            except:
                is_service_func = False
            is_old = re.match('old*', attr) is not None

            # Where the overloading happens
            if ((is_desc or (is_func and not is_service_func)) and not is_base and not is_old):
                passer = self._pass_method_args(lit)
                new_attr = self._overload_method(passer)
                setattr(torch.autograd.variable.Variable,
                        'old_{}'.format(attr), lit)
                setattr(torch.autograd.variable.Variable, attr, new_attr)

        self._hook_var_send_()
        self._hook_get_(torch.autograd.variable.Variable)
        self._hook_var_ser()

    # Special Variable method hooks
    def _hook_var___new__(hook_self):
        """Overload Variable.__new__"""

        torch.autograd.variable.Variable.old___new__ = torch.autograd.variable.Variable.__new__

        def new___new__(cls, *args, **kwargs):
            result = cls.old___new__(cls, *args,  **kwargs)
            result = hook_self.local_worker.register_object(hook_self.local_worker,
                                                            result, is_pointer=False)
            return result

        torch.autograd.variable.Variable.__new__ = new___new__

    def _hook_var_contents(hook_self):
        """Overload Variable.data and Variable.grad properties."""
        torch.autograd.variable.Variable.old_data = torch.autograd.variable.Variable.data
        torch.autograd.variable.Variable.old_grad = torch.autograd.variable.Variable.grad

        hook_self._hook_new_data()
        hook_self._hook_new_grad()

    def _hook_new_data(hook_self):

        @property
        def new_data(self):
            if not hasattr(self, 'data_registered'):

                if(hasattr(self.old_data, 'id')):
                    obj_id = self.old_data.id
                else:
                    obj_id = None

                if(not hasattr(self, 'owners')):
                    hook_self.local_worker.register_object(hook_self.local_worker,
                                                           obj=self,
                                                           owners=[hook_self.local_worker.id],
                                                           is_pointer=False)

                self.old_data = hook_self.local_worker.register_object(hook_self.local_worker,
                                                                       obj=self.old_data,
                                                                       owners=self.owners,
                                                                       id=obj_id,
                                                                       is_pointer=self.is_pointer)
                self.data_registered = True

            return self.old_data

        @new_data.setter
        def new_data(self, new):
            self.old_data = new

        torch.autograd.variable.Variable.data = new_data

    def _hook_new_grad(hook_self):

        @property
        def new_grad(self):
            if not hasattr(self, 'grad_registered'):

                if self.old_grad is not None:

                    if(hasattr(self.old_grad, 'id')):
                        grad_id = self.old_grad.id
                    else:
                        grad_id = None

                    if(not hasattr(self, 'owners')):
                        hook_self.local_worker.register_object(hook_self.local_worker,
                                                               obj=self,
                                                               owners=[hook_self.local_worker.id],
                                                               is_pointer=False)

                    _ip = self.is_pointer
                    self.old_grad = hook_self.local_worker.register_object(hook_self.local_worker,
                                                                           obj=self.old_grad,
                                                                           owners=self.owners,
                                                                           id=grad_id,
                                                                           is_pointer=_ip)
                    self.grad_registered = True

            return self.old_grad

        @new_grad.setter
        def new_grad(self, new):
            self.old_grad = new

        torch.autograd.variable.Variable.grad = new_grad

    def _hook_var_send_(hook_self):
        def send_(self, workers):
            """
            Sends a Variable object to a (sequence of) Grid workers.

            Args:
            workers: string (or sequence) containing IPFS address(es)
                of worker node(s).
            """

            # makes singleton if needed
            workers = hook_self.local_worker._check_workers(self, workers)
            self = hook_self.local_worker.register_object(hook_self.local_worker,
                                                          obj=self,
                                                          id=self.id,
                                                          owners=workers)
            for worker in workers:
                # TODO: sync or async? likely won't be worth doing async,
                #       but should check (low priority)
                hook_self.local_worker.send_obj(self, worker)

            hook_self.local_worker.register_object(hook_self.local_worker, obj=self, id=self.id,
                                                   owners=self.owners, is_pointer=True)

            return hook_self._var_to_pointer(self, hook_self)

        setattr(torch.autograd.variable.Variable, 'send_', send_)

    def _hook_var_ser(hook_self):
        def ser(self, include_data=True):
            var_msg = {}
            var_msg['torch_type'] = re.search("<class '(.*)'>", str(self.__class__)).group(1)
            var_msg['requires_grad'] = self.requires_grad
            var_msg['volatile'] = self.volatile
            var_msg['data'] = self.data.ser(include_data)
            if self.grad is not None:
                var_msg['grad'] = self.grad.ser(include_data)
            else:
                var_msg['grad'] = None
            var_msg['id'] = self.id
            if(type(self.owners[0]) is int):
                var_msg['owners'] = self.owners
            else:
                var_msg['owners'] = list(map(lambda x: x.id, self.owners))
            var_msg['is_pointer'] = not include_data
            return json.dumps(var_msg)

        def deser(self, obj_msg):

            if 'data' in obj_msg.keys():
                data_msg = json.loads(obj_msg['data'])
                tensor_type = hook_self.types_guard(data_msg['torch_type'])
                data_obj = tensor_type.deser(tensor_type, data_msg)
                # data_obj = hook_self.build_tensor(data_msg, tensor_type)
                data = hook_self.local_worker.handle_register(data_obj, data_msg)

            if 'grad' in obj_msg.keys():
                if obj_msg['grad'] is not None:
                    grad_msg = json.loads(obj_msg['grad'])
                    var_type = hook_self.types_guard(grad_msg['torch_type'])
                    grad_obj = hook_self._build_var(grad_msg, var_type)
                    grad = hook_self.local_worker.handle_register(grad_obj, grad_msg,
                                                                  force_attach_to_worker=False,
                                                                  temporary=True)

                else:
                    grad = None
            var = self(data, volatile=obj_msg['volatile'], requires_grad=obj_msg['requires_grad'])
            # var.grad = grad
            if(grad is not None):
                setattr(var, 'grad', grad)
            else:
                var.grad = None

            # this returns grad because garbage collection seems to do something really strange
            # if grad isn't returned here. It re-initializes the gradient somehow but in a way
            # where it's not registered (which is bad)
            return var

        torch.autograd.variable.Variable.ser = ser
        torch.autograd.variable.Variable.deser = deser

    def _var_to_pointer(self, var, hook_self):
        if var.grad is not None:
            self._var_to_pointer(var.grad, hook_self)

        var.data.old_set_(var.data.__class__(0))
        self.local_worker.register_object(hook_self.local_worker,
                                          obj=var.data,
                                          id=var.data.id,
                                          owners=var.owners,
                                          is_pointer=True)
        return var

    def _build_var(self, obj_msg, torch_type):

        if 'data' in obj_msg.keys():
            data_msg = json.loads(obj_msg['data'])
            tensor_type = self.types_guard(data_msg['torch_type'])
            data_obj = tensor_type.deser(tensor_type, data_msg)
            # data_obj = self.build_tensor(data_msg, tensor_type)
            data = self.local_worker.handle_register(data_obj, data_msg, temporary=True)

        if 'grad' in obj_msg.keys():
            if obj_msg['grad'] is not None:
                grad_msg = json.loads(obj_msg['grad'])
                var_type = self.types_guard(grad_msg['torch_type'])
                grad_obj = self._build_var(grad_msg, var_type)
                grad = self.local_worker.handle_register(grad_obj, grad_msg, temporary=True)
            else:
                grad = None
        var = torch_type(data, volatile=obj_msg['volatile'],
                         requires_grad=obj_msg['requires_grad'])
        var.grad = grad
        return var


class TensorflowHook(BaseHook):
    r""" TODO: Hook Tensorflow"""


class KerasHook(BaseHook):
    r""" TODO: Hook Keras"""
