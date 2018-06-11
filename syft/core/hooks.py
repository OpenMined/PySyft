"""Hooks which override deep learning interfaces with remote execution functionality."""

import torch, inspect, random, re, json, types, functools
from . import workers

# from types import FunctionType, BuiltinFunctionType
# from functools import wraps, partial, partialmethod
# from .local_worker import BaseWorker, LocalWorker


class TorchHook(object):
    r""" A Hook which Overrides Methods on PyTorch Variables & Tensors

    The purpose of this class is to:

        * extend torch methods to allow for the moving of tensors
          and variables from one worker to another
        * override torch methods to execute commands on one worker
          that are called on tensors controlled by the local worker.

    This class is typically the first thing you will initialize when using PySyft with PyTorch because it
    is responsible for augmenting PyTorch with PySyft's added functionality (such as remote execution).

    :Parameters:
        
        * **local_worker (**:class:`.local_workers.BaseWorker` **)** you can optionally provide a local worker as
          a parameter which TorchHook will assume to be the worker owned by the local machine. If you leave it
          empty, TorchClient will automatically initialize a :class:`.local_workers.VirtualWorker` under the 
          assumption you're looking to do local experimentation/development.
        
        * **is_client (bool)** whether or not the TorchHook is being initialized as an end-user client.
          This can impact whether or not variables are deleted when they fall out of scope. If you set
          this incorrectly on a end user client, Tensors and Variables will never be deleted. If you set this
          incorrectly on a remote machine (not a client), tensors will not get saved. It's really only
          important if you're not initializing the local worker yourself.
    
    :Example:
    >>> from syft.core.hooks import TorchHook
    >>> from syft.core.hooks import torch
    >>> hook = TorchHook(is_client=True)
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


    def __init__(self, local_worker=None, is_client=True):
        

        self.local_worker = local_worker
        if(self.local_worker is None):

            # Every TorchHook instance should have a local worker which is responsible for 
            # interfacing with other workers. The worker interface is what allows the Torch
            # specific code in TorchHook to be agnostic to the means by which workers communicate
            # (such as peer-to-peer, sockets, through local ports, or all within the same process)
            self.local_worker = workers.VirtualWorker(hook=self, is_client_worker=is_client)

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

        # Perform overloading
        print('Hooking into Torch...')
        self._hook_torch_module()
        for t_type in self.tensor_types:
            self._hook_tensor(t_type)
        # self.hook_variable()
        print('Overloading complete.')

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
            #Return a new partial object which when called will behave like func called with the 
            # positional arguments args and keyword arguments keywords. If more arguments are 
            # supplied to the call, they are appended to args. If additional keyword arguments 
            # are supplied, they extend and override keywords. 

            #The partial() is used for partial function application which “freezes” some 
            # portion of a function’s arguments and/or keywords resulting in a new object 
            # with a simplified signature.
            return functools.partial(func, *args, **kwargs)
        return pass_args

    def _overload_method(service_self, method):
        """
        Wrapper overloading partialmethod objects of Torch object
        methods.  Compiles command, checks for Tensors and Variables in
        the args/kwargs, determines locations of all Tensors and
        Variables involved in computation, and handles the computation
        accordingly.
        """
        @functools.wraps(method)
        def send_to_workers(self, *args, **kwargs):
            part = method(self, *args, **kwargs)
            if self.is_pointer:
                command = service_self.compile_command(part, has_self=True)
                tensorvars = get_tensorvars(service_self, command)
                has_remote = check_remote(tensorvars)
                multiple_owners, owners = get_owners(tensorvars)
                if has_remote and not multiple_owners:
                    for worker in owners:
                        command = replace_in_command(command)
                        registration, torch_type, var_data, var_grad = service_self.send_command(
                            command, worker)
                        if registration is None:
                            return var_data
                        # only returns last pointer, since tensors will
                        # be identical across machines for right now
                        pointer = service_self.assemble_result_pointer(worker,
                                                                       registration,
                                                                       torch_type,
                                                                       var_data,
                                                                       var_grad)
                else:
                    raise NotImplementedError("""MPC not yet implemented:
                        Torch objects need to be on the same machine in
                        order to compute with them.""")
                return pointer
            else:
                result = part.func(self, *args, **kwargs)
                if (type(result) in service_self.tensorvar_types and not hasattr(result, 'owner')):
                    result = service_self.register_object_(service_self.local_worker, result,
                                                           is_pointer=False)
                return result
        return send_to_workers

    def _overload_function(self, func):
        return self._overload_method(func)
        # """
        # Wrapper overloading partial objects of functions in the torch
        # module.  Compiles command, checks for Tensors and Variables in
        # the args/kwargs, determines locations of all Tensors and
        # Variables involved in computation, and handles the computation
        # accordingly.
        # """
        # @functools.wraps(func)
        # def send_to_workers(*args, **kwargs):
        #     part = func(*args, **kwargs)

        #     # Step 1: Compiles Command
        #     command = self._compile_command(part, has_self=False)

        #     # Step 2: checks for Tensors and Variables in the args/kwargs
        #     tensorvars = self._get_tensorvars(self, command)

        #     # Step 3: Checks to see if the tensor is local (on this machine) or is a pointer 
        #     # to a remote one (on a different machine)
        #     has_remote = self._check_remote(tensorvars)

        #     # If one of the tensor arguments is remote
        #     if has_remote:

        #         # Checks to see if the tensor has multiple owners (not yet fully supported func)
        #         multiple_owners, owners = self._get_owners(tensorvars)
        #         if multiple_owners:
        #             raise NotImplementedError("""MPC not yet implemented:
        #             Torch objects need to be on the same machine in order
        #             to compute with them.""")
        #         else:

        #             # if the tensor only has one owner (remote)
        #             command = _replace_in_command(command)

        #             for worker in owners:
        #                 # TODO: extend to iterables of pointers
        #                 registration, torch_type, var_data, var_grad = self._send_command(
        #                     command, worker)
        #                 if registration is None:
        #                     return var_data
        #                 pointer = self._assemble_result_pointer(
        #                     registration, torch_type, var_data, var_grad)
        #             return pointer
        #     else:
        #         result = part.func(*args, **kwargs)
        #         if type(result) in self.tensorvar_types:
        #             result = self._register_object(self.local_worker, result, is_pointer=False)
        #         return result

        # return send_to_workers

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

    @staticmethod
    def _get_tensorvars(command):
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
    def _get_owners(tensorvars):
        """Returns owners given a list of tensors
        Note - Andrew: This feels like a strange method."""

        owners = list(set([owner for tensorvar in tensorvars for owner in tensorvar.owners]))
        multiple_owners = len(owners) > 1
        return multiple_owners, owners

    @staticmethod
    def _send_command(command, recipient):
        """Send Torch command to recipient."""
        # TODO: Fix the case when response contains only a numeric
        response = self.local_worker.request_response(recipient=recipient,
                                                message=command,
                                                response_handler=self.process_response)

        registration, torch_type, var_data, var_grad = response
        return registration, torch_type, var_data, var_grad

    @staticmethod
    def _replace_in_command(command_msg):
        command_msg['args'] = map_tuple(
            None, command_msg['args'], replace_tensorvar)
        command_msg['kwargs'] = map_dict(
            None, command_msg['kwargs'], replace_tensorvar)
        try:
            command_msg['self'] = replace_tensorvar(command_msg['self'])
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
            if check(x) or isinstance(x, torch.autograd.Variable) or isinstance(x, torch.nn.Parameter):
                return '_fl.{}'.format(x.id)
            else:
                [replace_tensorvar(i) for i in x]
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
            torch_type = map_torch_type[torch_type]
        except KeyError:
            raise TypeError(
                "Tried to receive a non-Torch object of type {}.".format(
                    torch_type))

        if var_data is not None:
            data = self._assemble_result_pointer(worker, **var_data)
            data = self._register_object(worker, data, **var_data['registration'])
        elif torch_type in self.var_types:
            data = torch.Tensor(0)
        else:
            data = 0
        result = torch_type(data)
        if var_grad is not None:
            # grad = self.assemble_result_pointer(**var_grad)
            self._register_object(worker, result.grad, **var_grad['registration'])
        return self._register_object(self.local_worker, result, **registration)

    def _hook_tensor(self, tensor_type):
        """Overloading a given tensor_type"""
        # Overload 'special' methods here
        self.hook_tensor___init__(tensor_type)
        # self.hook_tensor___del__(tensor_type)
        self.hook_tensor___new__(tensor_type)
        self.hook_tensor___repr__(tensor_type)

        for attr in dir(tensor_type):
            # if we haven't already overloaded this function
            if 'old_{}'.format(attr) not in dir(tensor_type):
                # Conditions for inclusion/exclusion
                if attr in self.exclude:
                    continue
                lit = getattr(tensor_type, attr)
                is_base = attr in dir(object)
                is_desc = inspect.ismethoddescriptor(lit)
                is_func = type(lit) == types.FunctionType
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
    def hook_tensor___init__(service_self, tensor_type):
        """Overload tensor_type.__init__"""

        def new___init__(self, *args):
            super(tensor_type, self).__init__()
            self = service_self._register_object(service_self.local_worker, self, is_pointer=False)

        tensor_type.__init__ = new___init__

    def hook_tensor___del__(service_self, tensor_type):
        def new____del__(self, *args):
            print("deleting tensor")

        tensor_type.__del__ = new____del__

    def hook_tensor___new__(service_self, tensor_type):
        """Overload tensor_type.__new__"""

        if('old___new__' not in dir(tensor_type)):
            tensor_type.old___new__ = tensor_type.__new__

            def new___new__(cls, *args, **kwargs):
                result = cls.old___new__(cls, *args,  **kwargs)
                result = service_self._register_object(
                         service_self.local_worker, result, is_pointer=False)
                return result

            tensor_type.__new__ = new___new__

    def hook_tensor___repr__(service_self, tensor_type):
        """Overload tensor_type.__repr__"""
        if('old__repr__' not in dir(tensor_type)):
            tensor_type.old__repr__ = tensor_type.__repr__

            def new___repr__(self):
                if service_self.local_worker in self.owners:
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
    def _hook_tensor_send_(service_self, tensor_type):
        def send_(self, workers):
            """
            Sends a Tensor object to a (sequence of) Grid workers.

            Args:
            workers: string (or sequence) containing IPFS address(es)
                of worker node(s).
            """
            workers = check_workers(self, workers)  # makes singleton, if needed
            self = service_self.register_object_(service_self.local_worker, obj=self,
                                                 id=self.id, owners=workers)
            for worker in workers:
                # TODO: sync or async? likely won't be worth doing async,
                #       but should check (low priority)
                service_self.send_obj(self, worker)
            self = service_self.register_object_(service_self.local_worker, self.old_set_(tensor_type(0)),
                                                 id=self.id, owners=workers, is_pointer=True)
            return self

    def _hook_get_(service_self, torch_type):
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
            if service_self.local_worker.id in self.owners:
                return self

            x = service_self.request_obj(self, self.owners[0])
            service_self.register_object_(service_self.local_worker, x, id=x.id)

            try:
                self = service_self.register_object_(service_self.local_worker,
                                                     self.old_set_(x.type(self.type())),
                                                     id=self.id, owners=[service_self.local_worker.id])
            except TypeError:
                self = service_self.register_object_(service_self.local_worker,
                                                     self.old_set_(x.type(self.data.type())),
                                                     id=self.id, owners=[service_self.local_worker.id])
            try:
                self.data = service_self.register_object_(service_self.local_worker, x.data, id=x.data.id,
                                                          owners=[service_self.local_worker.id])
                try:
                    self.grad = service_self.register_object_(service_self.local_worker,
                                                              x.grad,
                                                              id=x.grad.id,
                                                              owners=[service_self.local_worker.id])
                except AttributeError:
                    pass
            except RuntimeError:
                pass

            return self

        setattr(torch_type, 'get_', get_)

    def _hook_tensor__ser(service_self, tensor_type):
        def _ser(self, include_data=True):
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

    def _register_object(self, worker, obj, force_attach_to_worker=False, **kwargs):
        """
        Registers an object with the current worker node.
        Selects an id for the object, assigns a list of owners,
        and establishes whether it's a pointer or not.

        Args:
            obj: a Torch instance, e.g. Tensor or Variable
        Default kwargs:
            id: random integer between 0 and 1e10
            owners: list containing local worker's IPFS id
            is_pointer: False
        """
        # TODO: Assign default id more intelligently (low priority)
        #       Consider popping id from long list of unique integers
        keys = kwargs.keys()

        obj.id = (kwargs['id']
                  if 'id' in keys
                  else random.randint(0, 1e10))

        obj.owners = (kwargs['owners']
                      if 'owners' in keys
                      else [worker.id])

        # check to see if we can resolve owner id to pointer
        owner_pointers = list()
        for owner in obj.owners:
            if owner in self.local_worker._known_workers.keys():
                owner_pointers.append(self.local_worker._known_workers[owner])
            else:
                owner_pointers.append(owner)
        obj.owners = owner_pointers

        obj.is_pointer = (kwargs['is_pointer']
                          if 'is_pointer' in keys
                          else False)

        mal_points_away = obj.is_pointer and worker.id in obj.owners
        # print("Mal Points Away:" + str(mal_points_away))
        # print("self.local_worker.id in obj.owners == " + str(self.local_worker.id in obj.owners))
        # The following was meant to assure that we didn't try to
        # register objects we didn't have. We end up needing to register
        # objects with non-local owners on the worker side before sending
        # things off, so it's been relaxed.  Consider using a 'strict'
        # kwarg for strict checking of this stuff
        mal_points_here = False
        # mal_points_here = not obj.is_pointer and self.local_worker.id not in obj.owners
        if mal_points_away or mal_points_here:
            raise RuntimeError(
                'Invalid registry: is_pointer is {} but owners is {}'.format(
                    obj.is_pointer, obj.owners))

        worker.set_obj(obj.id, obj, force=force_attach_to_worker)

        return obj

    # # Registration and communication handlers
    # def send_obj(self, obj, recipient):
    #     """Send Torch object to recipient."""
    #     self.local_worker.send_obj(obj=obj, recipient=recipient)

    # def request_obj(self, obj, sender):
    #     """Request Torch object from sender."""
    #     try:
    #         return self.local_worker.request_obj(obj_id=obj.id, sender=sender)
    #     except AttributeError:
    #         return self.local_worker.request_obj(obj_id=obj, sender=sender)





    # def process_response(self, response):
    #     """Processes a worker's response from a command."""
    #     # TODO: Extend to responses that are iterables.
    #     # TODO: Fix the case when response contains only a numeric
    #     # print(response)
    #     response = json.loads(response)
    #     try:
    #         return (response['registration'], response['torch_type'],
    #                 response['var_data'], response['var_grad'])
    #     except:
    #         return response

    



#         setattr(tensor_type, 'send_', send_)

#     def hook_var_send_(service_self):
#         def send_(self, workers):
#             """
#             Sends a Variable object to a (sequence of) Grid workers.

#             Args:
#             workers: string (or sequence) containing IPFS address(es)
#                 of worker node(s).
#             """
#             workers = check_workers(self, workers)  # makes singleton, if needed
#             self = service_self.register_object_(service_self.local_worker,
#                                                  obj=self,
#                                                  id=self.id,
#                                                  owners=workers)
#             for worker in workers:
#                 # TODO: sync or async? likely won't be worth doing async,
#                 #       but should check (low priority)
#                 service_self.send_obj(self, worker)
#             service_self.register_object_(service_self.local_worker, obj=self, id=self.id,
#                                           owners=self.owners, is_pointer=True)

#             return service_self.var_to_pointer(self, service_self)

#         setattr(torch.autograd.variable.Variable, 'send_', send_)

#     def var_to_pointer(self, var, service_self):
#         if var.grad is not None:
#             self.var_to_pointer(var.grad)

#         var.data.old_set_(var.data.__class__(0))
#         self.register_object_(service_self.local_worker,
#                               obj=var.data,
#                               id=var.data.id,
#                               owners=var.owners,
#                               is_pointer=True)
#         return var














#     # Special Variable method hooks
#     def hook_var___new__(service_self):
#         """Overload Variable.__new__"""

#         torch.autograd.variable.Variable.old___new__ = torch.autograd.variable.Variable.__new__

#         def new___new__(cls, *args, **kwargs):
#             result = cls.old___new__(cls, *args,  **kwargs)
#             result = service_self.register_object_(service_self.local_worker, result, is_pointer=False)
#             return result

#         torch.autograd.variable.Variable.__new__ = new___new__





#     def hook_variable(self):
#         # Overload 'special' methods here
#         self.hook_var___new__()
#         hook_var_contents(self)

#         for attr in dir(torch.autograd.variable.Variable):

#             # Conditions for inclusion/exclusion
#             if attr in self.exclude + self.var_exclude:
#                 continue
#             lit = getattr(torch.autograd.variable.Variable, attr)
#             is_base = attr in dir(object)
#             is_desc = inspect.ismethoddescriptor(lit)
#             is_func = type(lit) == FunctionType
#             try:
#                 is_service_func = 'HookService' in lit.__qualname__
#             except:
#                 is_service_func = False
#             is_old = re.match('old*', attr) is not None

#             # Where the overloading happens
#             if ((is_desc or (is_func and not is_service_func)) and not is_base and not is_old):
#                 passer = self.pass_method_args(lit)
#                 new_attr = self.overload_method(passer)
#                 setattr(torch.autograd.variable.Variable,
#                         'old_{}'.format(attr), lit)
#                 setattr(torch.autograd.variable.Variable, attr, new_attr)

#         self.hook_var_send_()
#         self.hook_get_(torch.autograd.variable.Variable)
#         hook_var__ser(self)

#     @classmethod
#     def build_tensor(cls, obj_msg, torch_type):
#         # this could be a significant failure point, security-wise
#         if 'data' in obj_msg.keys():
#             data = obj_msg['data']
#             data = tensor_contents_guard(data)
#             v = torch_type(data)
#         else:
#             v = torch.old_zeros(0).type(torch_type)
#         return v

#     def build_var(self, obj_msg, torch_type):

#         if 'data' in obj_msg.keys():
#             data_msg = json.loads(obj_msg['data'])
#             tensor_type = types_guard(data_msg['torch_type'])
#             data_obj = self.build_tensor(data_msg, tensor_type)
#             data = self.handle_register(data_obj, data_msg)

#         if 'grad' in obj_msg.keys():
#             if obj_msg['grad'] is not None:
#                 grad_msg = json.loads(obj_msg['grad'])
#                 var_type = types_guard(grad_msg['torch_type'])
#                 grad_obj = self.build_var(grad_msg, var_type)
#                 grad = self.handle_register(grad_obj, grad_msg)
#             else:
#                 grad = None
#         var = torch_type(data, volatile=obj_msg['volatile'],
#                          requires_grad=obj_msg['requires_grad'])
#         var.grad = grad
#         return var

#     def handle_register(self, torch_object, obj_msg):
#         try:
#             # TorchClient case
#             # delete registration from init; it's got the wrong id
#             self.local_worker.rm_obj(torch_object.id)
#         except (AttributeError, KeyError):
#             # Worker case: v was never formally registered
#             pass
#         torch_object = self.register_object_(self.local_worker,
#                                              obj=torch_object,
#                                              id=obj_msg['id'],
#                                              owners=[self.local_worker.id])
#         return torch_object




# # Helpers for HookService and TorchService
# def check_workers(self, workers):
#     if type(workers) is str:
#         workers = [workers]
#     if issubclass(type(workers), workers.BaseWorker):
#         workers = [workers]
#     elif not hasattr(workers, '__iter__'):
#         raise TypeError(
#             """Can only send {} to a string worker ID or an iterable of
#             string worker IDs, not {}""".format(self.__name__, workers)
#             )
#     return workers















# # Client needs to identify a tensor before sending commands that use it
# def id_tensorvar(x):
#     pat = re.compile('_fl.(.*)')
#     try:
#         if isinstance(x, str):
#             return int(pat.search(x).group(1))
#         else:
#             return [id_tensorvar(i) for i in x]
#     except AttributeError:
#         return x


# # Safety checks for serializing and deserializing torch objects
# # Desperately needs stress testing before going out in the wild
# map_tensor_type = {
#     'torch.FloatTensor': torch.FloatTensor,
#     'torch.DoubleTensor': torch.DoubleTensor,
#     'torch.HalfTensor': torch.HalfTensor,
#     'torch.ByteTensor': torch.ByteTensor,
#     'torch.CharTensor': torch.CharTensor,
#     'torch.ShortTensor': torch.ShortTensor,
#     'torch.IntTensor': torch.IntTensor,
#     'torch.LongTensor': torch.LongTensor
# }
# map_var_type = {
#     'torch.autograd.variable.Variable': torch.autograd.variable.Variable,
#     'torch.nn.parameter.Parameter': torch.nn.parameter.Parameter
# }
# map_torch_type = dict(map_tensor_type, **map_var_type)


# def types_guard(_torch_type):
#     try:
#         return map_torch_type[_torch_type]
#     except KeyError:
#         raise TypeError(
#             "Tried to receive a non-Torch object of type {}.".format(
#                 _torch_type))


# def tensor_contents_guard(contents):
#     # TODO: check to make sure the incoming list isn't dangerous to use for
#     #       constructing a tensor (likely non-trivial)
#     return contents


# def command_guard(command, allowed):
#     if command not in allowed:
#         raise RuntimeError(
#             'Command "{}" is not a supported Torch operation.'.format(command))
#     return command


# # Worker needs to retrieve tensor by ID before computing with it
# def retrieve_tensor(self, x):
#     try:
#         return self.get_obj(id_tensorvar(x))
#     except TypeError:
#         try:
#             return [self.get_obj(i) for i in id_tensorvar(x)]
#         except TypeError:
#             return x
#     except KeyError:
#         return x


# def map_tuple(service, args, func):
#     if service:
#         return tuple(func(service, x) for x in args)
#     else:
#         return tuple(func(x) for x in args)


# def map_dict(service, kwargs, func):
#     if service:
#         return {key: func(service, val) for key, val in kwargs.items()}
#     else:
#         return {key: func(val) for key, val in kwargs.items()}




#     def _deser(self, obj_msg):

#         # this could be a significant failure point, security-wise
#         data = tensor_contents_guard(obj_msg['data'])
#         v = self(data)
#         return v

#     tensor_type._ser = _ser
#     tensor_type._deser = _deser


# def hook_var__ser(service_self):
#     def _ser(self, include_data=True):
#         var_msg = {}
#         var_msg['torch_type'] = re.search("<class '(.*)'>", str(self.__class__)).group(1)
#         var_msg['requires_grad'] = self.requires_grad
#         var_msg['volatile'] = self.volatile
#         var_msg['data'] = self.data._ser(include_data)
#         if self.grad is not None:
#             var_msg['grad'] = self.grad._ser(include_data)
#         else:
#             var_msg['grad'] = None
#         var_msg['id'] = self.id
#         if(type(self.owners[0]) is int):
#             var_msg['owners'] = self.owners
#         else:
#             var_msg['owners'] = list(map(lambda x: x.id, self.owners))
#         var_msg['is_pointer'] = not include_data
#         return json.dumps(var_msg)

#     def _deser(self, obj_msg):

#         if 'data' in obj_msg.keys():
#             data_msg = json.loads(obj_msg['data'])
#             tensor_type = types_guard(data_msg['torch_type'])
#             data_obj = service_self.build_tensor(data_msg, tensor_type)
#             data = service_self.handle_register(data_obj, data_msg)

#         if 'grad' in obj_msg.keys():
#             if obj_msg['grad'] is not None:
#                 grad_msg = json.loads(obj_msg['grad'])
#                 var_type = types_guard(grad_msg['torch_type'])
#                 grad_obj = service_self.build_var(grad_msg, var_type)
#                 grad = service_self.handle_register(grad_obj, grad_msg)
#             else:
#                 grad = None
#         var = self(data, volatile=obj_msg['volatile'], requires_grad=obj_msg['requires_grad'])
#         var.grad = grad
#         return var

#     torch.autograd.variable.Variable._ser = _ser
#     torch.autograd.variable.Variable._deser = _deser


# def hook_var_contents(service_self):
#     """Overload Variable.data and Variable.grad properties."""
#     torch.autograd.variable.Variable.old_data = torch.autograd.variable.Variable.data
#     torch.autograd.variable.Variable.old_grad = torch.autograd.variable.Variable.grad

#     hook_new_data(service_self)
#     hook_new_grad(service_self)


# def hook_new_data(service_self):

#     @property
#     def new_data(self):
#         try:
#             self.data_registered
#         except AttributeError:
#             try:
#                 self.old_data = service_self.register_object_(service_self.local_worker,
#                                                               obj=self.old_data,
#                                                               id=self.old_data.id,
#                                                               owners=self.owners,
#                                                               is_pointer=self.is_pointer)
#                 self.data_registered = True
#             except AttributeError:
#                 try:
#                     self.old_data = service_self.register_object_(service_self.local_worker,
#                                                                   obj=self.old_data,
#                                                                   owners=self.owners,
#                                                                   is_pointer=self.is_pointer)
#                     self.data_registered = True
#                 except AttributeError:
#                     service_self.register_object_(service_self.local_worker,
#                                                   obj=self,
#                                                   owners=[service_self.local_worker.id],
#                                                   is_pointer=False)
#                     self.old_data = service_self.register_object_(service_self.local_worker,
#                                                                   obj=self.old_data,
#                                                                   owners=self.owners,
#                                                                   is_pointer=self.is_pointer)
#                     self.data_registered = True
#         return self.old_data

#     @new_data.setter
#     def new_data(self, new):
#         self.old_data = new

#     torch.autograd.variable.Variable.data = new_data


# def hook_new_grad(service_self):

#     @property
#     def new_grad(self):
#         try:
#             self.grad_registered
#         except AttributeError:
#             if self.old_grad is not None:
#                 try:
#                     self.old_grad = service_self.register_object_(service_self.local_worker,
#                                                                   obj=self.old_grad,
#                                                                   owners=self.owners,
#                                                                   id=self.old_grad.id,
#                                                                   is_pointer=self.is_pointer)
#                     self.grad_registered = True
#                 except AttributeError:
#                     try:
#                         self.old_grad = service_self.register_object_(service_self.local_worker,
#                                                                       obj=self.old_grad,
#                                                                       owners=self.owners,
#                                                                       is_pointer=self.is_pointer)
#                         self.grad_registered = True
#                     except AttributeError:
#                         service_self.register_object_(service_self.local_worker,
#                                                       obj=self,
#                                                       owners=[service_self.local_worker.id],
#                                                       is_pointer=False)
#                         self.old_grad = service_self.register_object_(service_self.local_worker,
#                                                                       obj=self.old_grad,
#                                                                       owners=self.owners,
#                                                                       is_pointer=self.is_pointer)
#                         self.grad_registered = True

#         return self.old_grad

#     @new_grad.setter
#     def new_grad(self, new):
#         self.old_grad = new

#     torch.autograd.variable.Variable.grad = new_grad
