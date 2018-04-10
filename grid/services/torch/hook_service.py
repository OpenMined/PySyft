import torch
from ..base import BaseService
from ... import channels
from ...lib import utils, torch_utils as tu

from collections import OrderedDict
from functools import wraps, partial, partialmethod
import inspect
import random
import re
from types import *


class HookService(BaseService):
    def __init__(self, worker):
        super().__init__(worker)

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


    ## Registration and communication handlers
    def send_obj(self, obj, recipient):
        """Send Torch object to recipient."""
        self.worker.publish(
            channels.torch_listen_for_obj_callback(recipient),
            message=obj._ser())


    def request_obj(self, obj, sender):
        """Request Torch object from sender."""
        return self.worker.request_response(
            channel=channels.torch_listen_for_obj_req_callback(sender),
            message=obj.id,
            response_handler=self.worker.services['torch_service'].receive_obj_break)


    def send_command(self, command, recipient):
        """Send Torch command to recipient."""
        # TODO: Fix the case when response contains only a numeric
        registration, torch_type, var_data, var_grad = self.worker.request_response(
            channels.torch_listen_for_command_callback(recipient),
            message=command,
            response_handler=self.process_response)
        return registration, torch_type, var_data, var_grad


    def assemble_result_pointer(self, registration, torch_type, var_data, var_grad):
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
            data = self.assemble_result_pointer(**var_data)
            self.register_object_(data, **var_data['registration'])
        else:
            data = 0
        result = torch_type(data)
        if var_grad is not None:
            grad = self.assemble_result_pointer(**var_grad)
            self.register_object_(result.grad, **var_grad['registration'])
        return self.register_object_(result, **registration)


    def process_response(self, response):
        """Processes a worker's response from a command."""
        # TODO: Extend to responses that are iterables.
        # TODO: Fix the case when response contains only a numeric
        response = utils.unpack(response)
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


    ## Grid-specific method hooking
    def hook_tensor_send_(service_self, tensor_type):
        def send_(self, workers):
            """
            Sends a Tensor object to a (sequence of) Grid workers.

            Args:
            workers: string (or sequence) containing IPFS address(es)
                of worker node(s).
            """
            workers = tu.check_workers(self, workers) # makes singleton, if needed
            self = service_self.register_object_(self, id=self.id, owners=workers)
            for worker in workers:
                # TODO: sync or async? likely won't be worth doing async,
                #       but should check (low priority)
                service_self.send_obj(self, worker)
            self = service_self.register_object_(self.old_set_(tensor_type(0)),
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
            workers = tu.check_workers(self, workers) # makes singleton, if needed
            self = service_self.register_object_(self, id=self.id, owners=workers)
            for worker in workers:
                # TODO: sync or async? likely won't be worth doing async,
                #       but should check (low priority)
                service_self.send_obj(self, worker)
            service_self.register_object_(self, id=self.id,
                owners=self.owners, is_pointer=True)

            return service_self.var_to_pointer(self)

        setattr(torch.autograd.variable.Variable, 'send_', send_)


    def var_to_pointer(self, var):
        if var.grad is not None:
            self.var_to_pointer(var.grad)

        var.data.old_set_(var.data.__class__(0))
        self.register_object_(var.data, id=var.data.id, owners=var.owners,
            is_pointer=True)

        return var


    def hook_get_(service_self, torch_type):
        def get_(self, reduce=lambda x:x[0]):
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
            if service_self.worker.id in self.owners:
                return self
            collected = []
            for worker in self.owners:
                x = service_self.request_obj(self, worker)
                collected.append(service_self.register_object_(x, id=x.id))  
            return service_self.register_object_(self.old_set_(reduce(collected)), id=self.id)
        setattr(torch_type, 'get_', get_)


    ## General hooking wrappers
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
            command = self.compile_command(part, has_self = False)
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
                        pointer = self.assemble_result_pointer(
                            registration, torch_type, var_data, var_grad)
                    return pointer
            else:
                result = part.func(*args, **kwargs)
                if type(result) in self.tensorvar_types:
                    result = self.register_object_(result, is_pointer=False)
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
                        # only returns last pointer, since tensors will
                        # be identical across machines for right now
                        pointer = service_self.assemble_result_pointer(
                            registration, torch_type, var_data, var_grad)
                else:
                    raise NotImplementedError("""MPC not yet implemented:
                        Torch objects need to be on the same machine in
                        order to compute with them.""")
                return pointer
            else:
                result = part.func(self, *args, **kwargs)
                if (type(result) in service_self.tensorvar_types and 
                    not hasattr(result, 'owner')):
                    result = service_self.register_object_(result,
                        is_pointer=False)
                return result
        return send_to_workers


    ## Special Tensor method HookService
    def hook_tensor___init__(service_self, tensor_type):
        """Overload tensor_type.__init__"""

        def new___init__(self, *args):
            super(tensor_type, self).__init__()
            self = service_self.register_object_(self, is_pointer=False)

        tensor_type.__init__ = new___init__
    

    def hook_tensor___new__(service_self, tensor_type):
        """Overload tensor_type.__new__"""

        if('old___new__' not in dir(tensor_type)):
            tensor_type.old___new__ = tensor_type.__new__
            def new___new__(cls, *args, **kwargs):
                result = cls.old___new__(cls, *args,  **kwargs)
                result = service_self.register_object_(result, is_pointer=False)
                return result
            
            tensor_type.__new__ = new___new__


    def hook_tensor___repr__(service_self, tensor_type):
        """Overload tensor_type.__repr__"""
        if('old__repr__' not in dir(tensor_type)):
            tensor_type.old__repr__ = tensor_type.__repr__
            def new___repr__(self):
                if service_self.worker.id in self.owners:
                    return self.old__repr__()
                else:
                    return "[{}.{} - Locations:{}]".format(
                        tensor_type.__module__,
                        tensor_type.__name__,
                        self.owners)

            tensor_type.__repr__ = new___repr__


    ## Special Variable method hooks
    def hook_var___new__(service_self):
        """Overload Variable.__new__"""
    
        torch.autograd.variable.Variable.old___new__ = torch.autograd.variable.Variable.__new__
        def new___new__(cls, *args, **kwargs):
            result = cls.old___new__(cls, *args,  **kwargs)
            result = service_self.register_object_(result, is_pointer=False)
            return result
        
        torch.autograd.variable.Variable.__new__ = new___new__


    def hook_var_contents(service_self):
        """Overload Variable.data and Variable.grad properties."""
        torch.autograd.variable.Variable.old_data = torch.autograd.variable.Variable.data
        torch.autograd.variable.Variable.old_grad = torch.autograd.variable.Variable.grad
        @property
        def new_data(self):
            try:
                self.data_registered
            except AttributeError:
                self.old_data = service_self.register_object_(
                    self.old_data, owners=self.owners,
                    is_pointer=self.is_pointer)
                self.data_registered = True
            return self.old_data

        @new_data.setter
        def new_data(self, new):
            self.old_data = new
        
        @property
        def new_grad(self):
            try:
                self.grad_registered
            except AttributeError:
                if self.old_grad is not None:
                    self.old_grad = service_self.register_object(
                        self.old_grad, owners=self.owners,
                        is_pointer=self.is_pointer)
                    self.grad_registered = True
            return self.old_grad

        @new_grad.setter
        def new_grad(self, new):
            self.old_grad = new
        
        torch.autograd.variable.Variable.data = new_data
        torch.autograd.variable.Variable.grad = new_grad


    ## Overloading Torch objects
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
                is_func = type(lit)==FunctionType
                try:
                    is_service_func = 'HookService' in lit.__qualname__
                except:
                    is_service_func = False
                is_old = re.match('old*', attr) is not None

                # Where the overloading happens
                if ((is_desc or (is_func and not is_service_func)) 
                    and not is_base and not is_old):
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
        self.hook_var_contents()

        for attr in dir(torch.autograd.variable.Variable):

            # Conditions for inclusion/exclusion
            if attr in self.exclude + self.var_exclude:
                continue
            lit = getattr(torch.autograd.variable.Variable, attr)
            is_base = attr in dir(object)
            is_desc = inspect.ismethoddescriptor(lit)
            is_func = type(lit)==FunctionType
            try:
                is_service_func = 'HookService' in lit.__qualname__
            except:
                is_service_func = False
            is_old = re.match('old*', attr) is not None

            # Where the overloading happens
            if ((is_desc or (is_func and not is_service_func)) 
                and not is_base and not is_old):
                passer = self.pass_method_args(lit)
                new_attr = self.overload_method(passer)
                setattr(torch.autograd.variable.Variable, 
                    'old_{}'.format(attr), lit)
                setattr(torch.autograd.variable.Variable, attr, new_attr)

        self.hook_var_send_()
        self.hook_get_(torch.autograd.variable.Variable)
        tu.hook_var__ser(self)
