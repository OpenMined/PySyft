import torch
# from ...lib import utils, torch_utils as tu
from . import utils as tu
from ..worker import LocalWorker

from functools import wraps, partial, partialmethod
import inspect
import random
import re
from types import FunctionType, BuiltinFunctionType
import json


class TorchHook(object):
    def __init__(self, worker=None):
        # super().__init__()

        self.worker = worker
        if(self.worker is None):
            self.worker = LocalWorker(hook=self)

        self.known_workers = {}
        self.known_workers[self.worker.id] = self.worker

        self.torch_funcs = dir(torch)

        # Torch-specific
        self.tensor_types = [torch.FloatTensor,
                             torch.DoubleTensor,
                             torch.HalfTensor,
                             torch.ByteTensor,
                             torch.CharTensor,
                             torch.ShortTensor,
                             torch.IntTensor,
                             torch.LongTensor]
        self.var_types = [torch.autograd.Variable, torch.nn.Parameter]
        self.tensorvar_types = self.tensor_types + [torch.autograd.Variable]
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
        self.hook_torch_module()
        for t_type in self.tensor_types:
            self.hook_tensor(t_type)
        self.hook_variable()
        print('Overloading complete.')

    def add_worker(self, id, worker):
        if(id not in self.known_workers.keys()):
            self.known_workers[id] = worker

    def register_object_(self, worker, obj, **kwargs):
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
            if owner in self.known_workers.keys():
                owner_pointers.append(self.known_workers[owner])
            else:
                owner_pointers.append(owner)
        obj.owners = owner_pointers

        obj.is_pointer = (kwargs['is_pointer']
                          if 'is_pointer' in keys
                          else False)

        mal_points_away = obj.is_pointer and worker.id in obj.owners
        # print("Mal Points Away:" + str(mal_points_away))
        # print("self.worker.id in obj.owners == " + str(self.worker.id in obj.owners))
        # The following was meant to assure that we didn't try to
        # register objects we didn't have. We end up needing to register
        # objects with non-local owners on the worker side before sending
        # things off, so it's been relaxed.  Consider using a 'strict'
        # kwarg for strict checking of this stuff
        mal_points_here = False
        # mal_points_here = not obj.is_pointer and self.worker.id not in obj.owners
        if mal_points_away or mal_points_here:
            raise RuntimeError(
                'Invalid registry: is_pointer is {} but owners is {}'.format(
                    obj.is_pointer, obj.owners))
        worker.objects[obj.id] = obj

        return obj

    # Registration and communication handlers
    def send_obj(self, obj, recipient):
        """Send Torch object to recipient."""
        self.worker.send_obj(obj=obj, recipient=recipient)

    def request_obj(self, obj, sender):
        """Request Torch object from sender."""
        try:
            return self.worker.request_obj(obj_id=obj.id, sender=sender)
        except AttributeError:
            return self.worker.request_obj(obj_id=obj, sender=sender)

    def send_command(self, command, recipient):
        """Send Torch command to recipient."""
        # TODO: Fix the case when response contains only a numeric
        response = self.worker.request_response(recipient=recipient,
                                                message=command,
                                                response_handler=self.process_response)

        registration, torch_type, var_data, var_grad = response
        return registration, torch_type, var_data, var_grad

    def assemble_result_pointer(self, worker, registration, torch_type, var_data, var_grad):
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
            torch_type = tu.map_torch_type[torch_type]
        except KeyError:
            raise TypeError(
                "Tried to receive a non-Torch object of type {}.".format(
                    torch_type))

        if var_data is not None:
            data = self.assemble_result_pointer(worker, **var_data)
            data = self.register_object_(worker, data, **var_data['registration'])
        elif torch_type in self.var_types:
            data = torch.Tensor(0)
        else:
            data = 0
        result = torch_type(data)
        if var_grad is not None:
            # grad = self.assemble_result_pointer(**var_grad)
            self.register_object_(worker, result.grad, **var_grad['registration'])
        return self.register_object_(self.worker, result, **registration)

    def process_response(self, response):
        """Processes a worker's response from a command."""
        # TODO: Extend to responses that are iterables.
        # TODO: Fix the case when response contains only a numeric
        # print(response)
        response = json.loads(response)
        try:
            return (response['registration'], response['torch_type'],
                    response['var_data'], response['var_grad'])
        except:
            return response

    @staticmethod
    def compile_command(partial_func, has_self):
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

    # Grid-specific method hooking
    def hook_tensor_send_(service_self, tensor_type):
        def send_(self, workers):
            """
            Sends a Tensor object to a (sequence of) Grid workers.

            Args:
            workers: string (or sequence) containing IPFS address(es)
                of worker node(s).
            """
            workers = tu.check_workers(self, workers)  # makes singleton, if needed
            self = service_self.register_object_(service_self.worker, obj=self,
                                                 id=self.id, owners=workers)
            for worker in workers:
                # TODO: sync or async? likely won't be worth doing async,
                #       but should check (low priority)
                service_self.send_obj(self, worker)
            self = service_self.register_object_(service_self.worker, self.old_set_(tensor_type(0)),
                                                 id=self.id, owners=workers, is_pointer=True)
            return self

        setattr(tensor_type, 'send_', send_)

    def hook_var_send_(service_self):
        def send_(self, workers):
            """
            Sends a Variable object to a (sequence of) Grid workers.

            Args:
            workers: string (or sequence) containing IPFS address(es)
                of worker node(s).
            """
            workers = tu.check_workers(self, workers)  # makes singleton, if needed
            self = service_self.register_object_(service_self.worker,
                                                 obj=self,
                                                 id=self.id,
                                                 owners=workers)
            for worker in workers:
                # TODO: sync or async? likely won't be worth doing async,
                #       but should check (low priority)
                service_self.send_obj(self, worker)
            service_self.register_object_(service_self.worker, obj=self, id=self.id,
                                          owners=self.owners, is_pointer=True)

            return service_self.var_to_pointer(self, service_self)

        setattr(torch.autograd.variable.Variable, 'send_', send_)

    def var_to_pointer(self, var, service_self):
        if var.grad is not None:
            self.var_to_pointer(var.grad)

        var.data.old_set_(var.data.__class__(0))
        self.register_object_(service_self.worker,
                              obj=var.data,
                              id=var.data.id,
                              owners=var.owners,
                              is_pointer=True)
        return var

    def hook_get_(service_self, torch_type):
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
            if service_self.worker.id in self.owners:
                return self

            x = service_self.request_obj(self, self.owners[0])
            service_self.register_object_(service_self.worker, x, id=x.id)

            try:
                self = service_self.register_object_(service_self.worker,
                                                     self.old_set_(x.type(self.type())),
                                                     id=self.id, owners=[service_self.worker.id])
            except TypeError:
                self = service_self.register_object_(service_self.worker,
                                                     self.old_set_(x.type(self.data.type())),
                                                     id=self.id, owners=[service_self.worker.id])
            try:
                self.data = service_self.register_object_(service_self.worker, x.data, id=x.data.id,
                                                          owners=[service_self.worker.id])
                try:
                    self.grad = service_self.register_object_(service_self.worker,
                                                              x.grad,
                                                              id=x.grad.id,
                                                              owners=[service_self.worker.id])
                except AttributeError:
                    pass
            except RuntimeError:
                pass

            return self

        setattr(torch_type, 'get_', get_)

    # General hooking wrappers
    @staticmethod
    def pass_func_args(func):
        """Wrapper gathering partial object from function call."""
        @wraps(func)
        def pass_args(*args, **kwargs):
            return partial(func, *args, **kwargs)
        return pass_args

    def overload_function(self, func):
        """
        Wrapper overloading partial objects of functions in the torch
        module.  Compiles command, checks for Tensors and Variables in
        the args/kwargs, determines locations of all Tensors and
        Variables involved in computation, and handles the computation
        accordingly.
        """
        @wraps(func)
        def send_to_workers(*args, **kwargs):
            part = func(*args, **kwargs)
            command = self.compile_command(part, has_self=False)
            tensorvars = tu.get_tensorvars(self, command)
            has_remote = tu.check_remote(tensorvars)
            if has_remote:
                multiple_owners, owners = tu.get_owners(tensorvars)
                if multiple_owners:
                    raise NotImplementedError("""MPC not yet implemented:
                    Torch objects need to be on the same machine in order
                    to compute with them.""")
                else:
                    command = tu.replace_in_command(command)
                    for worker in owners:
                        # TODO: extend to iterables of pointers
                        registration, torch_type, var_data, var_grad = self.send_command(
                            command, worker)
                        if registration is None:
                            return var_data
                        pointer = self.assemble_result_pointer(
                            registration, torch_type, var_data, var_grad)
                    return pointer
            else:
                result = part.func(*args, **kwargs)
                if type(result) in self.tensorvar_types:
                    result = self.register_object_(self.worker, result, is_pointer=False)
                return result

        return send_to_workers

    @staticmethod
    def pass_method_args(method):
        """Wrapper gathering partialmethod object from method call."""
        @wraps(method)
        def pass_args(*args, **kwargs):
            return partialmethod(method, *args, **kwargs)
        return pass_args

    def overload_method(service_self, method):
        """
        Wrapper overloading partialmethod objects of Torch object
        methods.  Compiles command, checks for Tensors and Variables in
        the args/kwargs, determines locations of all Tensors and
        Variables involved in computation, and handles the computation
        accordingly.
        """
        @wraps(method)
        def send_to_workers(self, *args, **kwargs):
            part = method(self, *args, **kwargs)
            if self.is_pointer:
                command = service_self.compile_command(part, has_self=True)
                tensorvars = tu.get_tensorvars(service_self, command)
                has_remote = tu.check_remote(tensorvars)
                multiple_owners, owners = tu.get_owners(tensorvars)
                if has_remote and not multiple_owners:
                    for worker in owners:
                        command = tu.replace_in_command(command)
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
                    result = service_self.register_object_(service_self.worker, result,
                                                           is_pointer=False)
                return result
        return send_to_workers

    # Special Tensor method HookService
    def hook_tensor___init__(service_self, tensor_type):
        """Overload tensor_type.__init__"""

        def new___init__(self, *args):
            super(tensor_type, self).__init__()
            self = service_self.register_object_(service_self.worker, self, is_pointer=False)

        tensor_type.__init__ = new___init__

    def hook_tensor___new__(service_self, tensor_type):
        """Overload tensor_type.__new__"""

        if('old___new__' not in dir(tensor_type)):
            tensor_type.old___new__ = tensor_type.__new__

            def new___new__(cls, *args, **kwargs):
                result = cls.old___new__(cls, *args,  **kwargs)
                result = service_self.register_object_(
                         service_self.worker, result, is_pointer=False)
                return result

            tensor_type.__new__ = new___new__

    def hook_tensor___repr__(service_self, tensor_type):
        """Overload tensor_type.__repr__"""
        if('old__repr__' not in dir(tensor_type)):
            tensor_type.old__repr__ = tensor_type.__repr__

            def new___repr__(self):
                if service_self.worker in self.owners:
                    return self.old__repr__()
                else:
                    return "[{}.{} - Locations:{}]".format(
                        tensor_type.__module__,
                        tensor_type.__name__,
                        self.owners)

            tensor_type.__repr__ = new___repr__

    # Special Variable method hooks
    def hook_var___new__(service_self):
        """Overload Variable.__new__"""

        torch.autograd.variable.Variable.old___new__ = torch.autograd.variable.Variable.__new__

        def new___new__(cls, *args, **kwargs):
            result = cls.old___new__(cls, *args,  **kwargs)
            result = service_self.register_object_(service_self.worker, result, is_pointer=False)
            return result

        torch.autograd.variable.Variable.__new__ = new___new__

    # Overloading Torch objects
    def hook_torch_module(self):
        """Overload functions in the main torch module"""
        for attr in self.torch_funcs:

            # Conditions for inclusion/exclusion
            if attr in self.torch_exclude:
                continue

            # if we haven't already overloaded this function
            if 'old_{}'.format(attr) in dir(torch):
                continue

            if 'old_' in attr:
                continue

            # Where the overloading happens
            lit = getattr(torch, attr)
            if (type(lit) in [FunctionType, BuiltinFunctionType]):

                passer = self.pass_func_args(lit)
                new_attr = self.overload_function(passer)
                setattr(torch, 'old_{}'.format(attr), lit)
                setattr(torch, attr, new_attr)

    def hook_tensor(self, tensor_type):
        """Overloading a given tensor_type"""
        # Overload 'special' methods here
        self.hook_tensor___init__(tensor_type)
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
                is_func = type(lit) == FunctionType
                try:
                    is_service_func = 'HookService' in lit.__qualname__
                except:
                    is_service_func = False
                is_old = re.match('old*', attr) is not None

                # Where the overloading happens
                if ((is_desc or (is_func and not is_service_func)) and not is_base and not is_old):
                    passer = self.pass_method_args(lit)
                    new_attr = self.overload_method(passer)
                    setattr(tensor_type, 'old_{}'.format(attr), lit)
                    setattr(tensor_type, attr, new_attr)

        # Add in our own Grid-specific methods
        self.hook_tensor_send_(tensor_type)
        self.hook_get_(tensor_type)
        tu.hook_tensor__ser(self, tensor_type)

    def hook_variable(self):
        # Overload 'special' methods here
        self.hook_var___new__()
        tu.hook_var_contents(self)

        for attr in dir(torch.autograd.variable.Variable):

            # Conditions for inclusion/exclusion
            if attr in self.exclude + self.var_exclude:
                continue
            lit = getattr(torch.autograd.variable.Variable, attr)
            is_base = attr in dir(object)
            is_desc = inspect.ismethoddescriptor(lit)
            is_func = type(lit) == FunctionType
            try:
                is_service_func = 'HookService' in lit.__qualname__
            except:
                is_service_func = False
            is_old = re.match('old*', attr) is not None

            # Where the overloading happens
            if ((is_desc or (is_func and not is_service_func)) and not is_base and not is_old):
                passer = self.pass_method_args(lit)
                new_attr = self.overload_method(passer)
                setattr(torch.autograd.variable.Variable,
                        'old_{}'.format(attr), lit)
                setattr(torch.autograd.variable.Variable, attr, new_attr)

        self.hook_var_send_()
        self.hook_get_(torch.autograd.variable.Variable)
        tu.hook_var__ser(self)

    @classmethod
    def build_tensor(cls, obj_msg, torch_type):
        # this could be a significant failure point, security-wise
        if 'data' in obj_msg.keys():
            data = obj_msg['data']
            data = tu.tensor_contents_guard(data)
            v = torch_type(data)
        else:
            v = torch.old_zeros(0).type(torch_type)
        return v

    def build_var(self, obj_msg, torch_type):

        if 'data' in obj_msg.keys():
            data_msg = json.loads(obj_msg['data'])
            tensor_type = tu.types_guard(data_msg['torch_type'])
            data_obj = self.build_tensor(data_msg, tensor_type)
            data = self.handle_register(data_obj, data_msg)

        if 'grad' in obj_msg.keys():
            if obj_msg['grad'] is not None:
                grad_msg = json.loads(obj_msg['grad'])
                var_type = tu.types_guard(grad_msg['torch_type'])
                grad_obj = self.build_var(grad_msg, var_type)
                grad = self.handle_register(grad_obj, grad_msg)
            else:
                grad = None
        var = torch_type(data, volatile=obj_msg['volatile'],
                         requires_grad=obj_msg['requires_grad'])
        var.grad = grad
        return var

    def handle_register(self, torch_object, obj_msg):
        try:
            # TorchClient case
            # delete registration from init; it's got the wrong id
            del self.worker.objects[torch_object.id]
        except (AttributeError, KeyError):
            # Worker case: v was never formally registered
            pass
        torch_object = self.register_object_(self.worker,
                                             obj=torch_object,
                                             id=obj_msg['id'],
                                             owners=[self.worker.id])
        return torch_object
