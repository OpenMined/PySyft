import torch
import random
import inspect
import re
import logging
import types
import syft as sy
from ... import workers
from ... import utils
from . import utils as torch_utils
from .tensor import _SyftTensor, _LocalTensor, _PointerTensor, _GeneralizedPointerTensor,_FixedPrecisionTensor, _TorchTensor
from .tensor import _TorchVariable


class TorchHook(object):
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

         * **queue_size (int, optional)** max length of the list storing messages
           to be sent. (Default: 0)
 
     :Example:
 
     >>> import syft as sy
     >>> hook = sy.TorchHook()
     Hooking into Torch...
     Overloading Complete.
     >>> x = sy.FloatTensor([-2,-1,0,1,2,3])
     >>> x
      -2
      -1
      0
      1
      2
      3
     [syft.core.frameworks.torch.tensor.FloatTensor of size 6]
     """
    def __init__(self, local_worker=None, is_client=True, verbose=True, queue_size=0):
        self.local_worker = local_worker

        if not hasattr(torch, 'torch_hooked'):
            torch.torch_hooked = 0
        else:
            torch.torch_hooked += 1

        # Methods that caused infinite recursion during testing
        # TODO: May want to handle the ones in "exclude" manually at
        #       some point
        self.exclude = (['ndimension', 'nelement', 'size', 'numel',
                         'type', 'tolist', 'dim', '__iter__', 'select',
                         '__getattr__', '_get_type'])

        self.to_auto_overload = {}

        if torch.torch_hooked > 0:
            logging.warn("Torch was already hooked... skipping hooking process")
            self.local_worker = torch.local_worker
        else:

            if self.local_worker is None:
                # Every TorchHook instance should have a local worker which is responsible for
                # interfacing with other workers. The worker interface is what allows the Torch
                # specific code in TorchHook to be agnostic to the means by which workers communicate
                # (such as peer-to-peer, sockets, through local ports, or all within the same process)
                self.local_worker = workers.VirtualWorker(hook=self, is_client_worker=is_client,
                                                          queue_size=queue_size)
            else:
                # if the local_worker already exists, then it MUST not know about the hook which is
                # just being created. Thus, we must inform it.
                self.local_worker.hook = self

            for typ in torch.tensorvar_types:
                self._hook_native_tensors_and_variables(typ)
                self._hook_syft_tensor_types(typ)

            self._hook_torch_module()
            self._hook_backward()
            self._hook_module()

            torch.local_worker = self.local_worker

    def _hook_native_tensors_and_variables(self, tensor_type):
        """Overloads given tensor_type (native)"""
        # Overload 'special' methods here

        self._add_registration_to___init__(tensor_type, register_child_instead=True)

        self._hook_properties(tensor_type)

        self.to_auto_overload[tensor_type] = self._which_methods_should_we_auto_overload(
            tensor_type)

        self._rename_native_functions(tensor_type)

        self._assign_methods_to_use_child(tensor_type)

        self._add_methods_from__TorchObject(tensor_type)

    def _hook_syft_tensor_types(self, tensor_type):
        """Overloads syft tensor_types"""
        self._hook_LocalTensor(tensor_type)

        self._hook_SyftTensor(tensor_type)

        self._hook_PointerTensor(tensor_type)
        self._hook_GeneralizedPointerTensor(tensor_type)


    def _add_registration_to___init__(hook_self, tensorvar_type, register_child_instead=False):
        """Overloads tensor_type.__new__ or Variable.__new__"""

        # TODO: This is added because of the following contradiction: instanciate x = FloatTensor()
        # and ask x.__module__, you'll get `sy.core.frameworks.torch.tensor.FloatTensor`
        # but now ask for dir(sy.core.frameworks.torch.tensor) you'll find no FloatTensor attribute
        # and x.float() will raise an exception because of this.
        # Why is x.__module__ == 'sy.core...'? How can we do this with elegance?
        if tensorvar_type.__module__ != sy._SyftTensor.__module__:
            setattr(sy.core.frameworks.torch.tensor, tensorvar_type.__name__, tensorvar_type)

        if 'native___init__' not in dir(tensorvar_type):
            tensorvar_type.native___init__ = tensorvar_type.__init__

        def new___init__(cls, *args, **kwargs):

            if 'owner' in kwargs and kwargs['owner'] is not None:
                owner = kwargs['owner']
            else:
                owner = hook_self.local_worker

            if 'id' in kwargs:
                id = kwargs['id']
            else:
                id = None

            if register_child_instead:
                cls.native___init__()
                _ = cls.child
                _ = "ignore pep8"
            else:
                cls.native___init__(*args, **kwargs)
                if id is None:
                    id = random.randint(0, 1e10)
                cls.id = id
                cls.owner = owner
                if 'skip_register' in kwargs and kwargs['skip_register']:
                    pass
                else:
                    owner.register_object(cls, id=id)

        tensorvar_type.__init__ = new___init__

    def _hook_properties(hook_self, tensor_type):
        """Overloads tensor_type properties"""
        @property
        def child(self):
            try:
                if hasattr(self, '_child') and self._child is not None:
                    return self._child
                else:
                    self._child = _LocalTensor(child=self,
                                               parent=self,
                                               torch_type='syft.' + type(self).__name__)
                    return self._child
            except TypeError:
                # for some reason, hasattr(self, '_child') returns a TypeError saying
                # "TypeError: 'NoneType' object is not callable". It's supposed to only
                # return False and I can't get to the bottom of it. So, for now, I'm
                # going to break a personal rule and use try/catch for logic, but
                # this is merely supposed to evaluate whether self has ._child as an
                # attribute. Note this only seems to happen when self is a
                # torch.autograd.Variable

                self._child = _LocalTensor(child=self,
                                           parent=self,
                                           torch_type='syft.' + type(self).__name__)
                return self._child

        @child.setter
        def child(self, value):
            self._child = value

        tensor_type.child = child

        @property
        def id(self):
            return self.child.id

        # TODO: this should not be possible, but it should also be possible to define a FloatTensor
        # with a specific id. This is in theory possible, but it doesnt seem to work in practice
        @id.setter
        def id(self, new_id):
            self.child.id = new_id
            return self

        tensor_type.id = id

        @property
        def location(self):
            return self.child.location

        tensor_type.location = location

        @property
        def id_at_location(self):
            return self.child.id_at_location

        tensor_type.id_at_location = id_at_location

        @property
        def owner(self):
            return self.child.owner

        tensor_type.owner = owner

    def _which_methods_should_we_auto_overload(self, tensor_type=torch.FloatTensor):
        """Creates list of methods to auto overload"""
        to_overload = list()

        for attr in dir(tensor_type):

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
            is_old = re.match('native*', attr) is not None

            if (is_desc or (is_func and not is_service_func)) and not is_base and not is_old:
                to_overload.append(attr)

        return to_overload

    def _rename_native_functions(self, tensor_type):
        """Renames functions that are auto overloaded"""
        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)

            # if we haven't already overloaded this function
            if 'native_{}'.format(attr) not in dir(tensor_type):
                setattr(tensor_type, 'native_{}'.format(attr), lit)

            setattr(tensor_type, attr, None)

    def _assign_methods_to_use_child(self, tensor_type):
        """Assigns methods to use as child for auto overloaded functions"""
        for attr in self.to_auto_overload[tensor_type]:

            def forward_method_to_child(self, *args, **kwargs):

                child_args = torch_utils.get_child_in_args(*args, **kwargs)

                response = getattr(self.child, attr)(*child_args, **kwargs)

                return response

            new_attr = self._get_overloaded_method(attr)

            # if we haven't already overloaded this method
            if attr not in dir(tensor_type) or getattr(tensor_type, attr) is None:
                setattr(tensor_type, attr, new_attr)

    def _add_methods_from__TorchObject(self, tensor_type):
        """Add methods to auto overloaded functions"""
        exclude = ['__class__',
                   '__delattr__',
                   '__dir__',
                   '__doc__',
                   '__dict__',
                   '__eq__',
                   '__format__',
                   '__ge__',
                   '__getattribute__',
                   '__gt__',
                   '__hash__',
                   '__init__',
                   '__init_subclass__',
                   '__le__',
                   '__lt__',
                   '__weakref__',
                   '__ne__',
                   '__new__',
                   '__reduce__',
                   '__reduce_ex__',
                   '__setattr__',
                   '__sizeof__',
                   '__subclasshook__',
                   '_get_type']

        if issubclass(tensor_type, torch._TensorBase):
            parent_syft_obj = _TorchTensor
        else:
            parent_syft_obj = _TorchVariable

        for attr in dir(parent_syft_obj):
            if attr not in exclude:
                if attr in dir(tensor_type) and "native_" + str(attr) not in dir(tensor_type):
                    setattr(tensor_type, "native_" + str(attr), getattr(tensor_type, attr))
                setattr(tensor_type, attr, getattr(parent_syft_obj, attr))

    def _hook_LocalTensor(self, tensor_type):
        """Overloads LocalTensor"""
        # iterate through all methods and tell them to call the native function
        # on self.child
        for attr in self.to_auto_overload[tensor_type]:

            def forward_method_to_child(self, *args, **kwargs):

                child_args = torch_utils.get_child_in_args(*args, **kwargs)
                if attr == 'zero_':
                    response = getattr(self.child, 'native_' + attr)()
                else:
                    response = getattr(self.child, 'native_' + attr)(*child_args, **kwargs)

                syft_node = type(self)(child=response.child, parent=None,
                                       torch_type=type(response).__name__)

                # Insert the new node just before the wrapper
                # syft_node.child = response.child
                response.child.parent = syft_node
                response.child = syft_node
                syft_node.parent = response

                return response

            new_attr = self._get_overloaded_method(attr)

            # if we haven't already overloaded this method
            if attr not in dir(_LocalTensor) or getattr(_LocalTensor, attr) is None:
                setattr(_LocalTensor, attr, new_attr)

    def _hook_SyftTensor(hook_self, tensor_type):
        """Overloads SyftTensor"""
        hook_self._add_registration_to___init__(_SyftTensor)

        for attr in hook_self.to_auto_overload[tensor_type]:

            def forward_method_to_child(self, *args, **kwargs):

                child_args = torch_utils.get_child_in_args(*args, **kwargs)
                response = getattr(self.child, attr)(*child_args, **kwargs)

                syft_node = type(self)(child=response.child)

                # Insert the new node just before the wrapper
                # syft_node.child = response.child
                response.child.parent = syft_node
                response.child = syft_node
                syft_node.parent = response

                return response

            new_attr = hook_self._get_overloaded_method(attr)

            # if we haven't already overloaded this method
            if attr not in dir(_SyftTensor) or getattr(_SyftTensor, attr) is None:
                # call child method
                setattr(_SyftTensor, attr, new_attr)

    def _hook_PointerTensor(self, tensor_type):
        """Overloads PointerTensor"""
        for attr in self.to_auto_overload[tensor_type]:
            # # if we haven't already overloaded this method
            # if attr not in dir(_PointerTensor) or getattr(_PointerTensor, attr) is None:

            setattr(_PointerTensor, attr, self._get_overloaded_method(attr))
    def _hook_GeneralizedPointerTensor(self, tensor_type):

        for attr in self.to_auto_overload[tensor_type]:
            # # if we haven't already overloaded this method
            # if attr not in dir(_GeneralizedPointerTensor) or getattr(_GeneralizedPointerTensor, attr) is None:

            setattr(_GeneralizedPointerTensor, attr, self._get_overloaded_method(attr))


    def _get_overloaded_method(hook_self, attr):
        """
        Wrapper overloading partial objects of methods in the torch
        module.  Compiles command, checks for Tensors and Variables in
        the args/kwargs, determines locations of all Tensors and
        Variables involved in computation, and handles the computation
        accordingly.
        """
        def _execute_method_call(self, *args, **kwargs):
            return hook_self._execute_call(attr, self, *args, **kwargs)

        return _execute_method_call

    def _hook_torch_module(self):
        """
        Overloads functions in the main torch module.

        The way this is accomplished is by first moving all existing module functions in the torch
        module to native_<function_name_here>. Thus, the real :func:`torch.cat` will become
        :func:`torch.native_cat` and :func:`torch.cat` will have our hooking code.
        """

        for module_name, module_funcs in torch.torch_modules.items():
            torch_module = eval(module_name)
            for attr in module_funcs:
                # Some functions we want to ignore (not override). Such functions have been hard
                # coded into the attribute self.torch_exclude
                if attr in torch.torch_exclude:
                    continue

                # if we haven't already overloaded this function
                if 'native_{}'.format(attr) in dir(torch_module):
                    continue

                # if we haven't already overloaded this function (redundancy allowed)
                if 'native_' in attr:
                    continue

                # Where the overloading happens
                lit = getattr(torch_module, attr)
                if type(lit) in [types.FunctionType, types.BuiltinFunctionType]:
                    new_attr = self._get_overloaded_function(module_name + '.' + attr)
                    setattr(torch_module, 'native_{}'.format(attr), lit)
                    setattr(torch_module, attr, new_attr)

    def _get_overloaded_function(hook_self, attr):
        """
        Wrapper overloading partial objects of functions in the torch
        module.  Compiles command, checks for Tensors and Variables in
        the args/kwargs, determines locations of all Tensors and
        Variables involved in computation, and handles the computation
        accordingly.
        """

        def _execute_function_call(*args, **kwargs):
            return hook_self._execute_call(attr, None, *args, **kwargs)

        return _execute_function_call

    def _execute_call(hook_self, attr, self, *args, **kwargs):
        """
        Forward the call to the local_worker
        """
        return hook_self.local_worker._execute_call(attr, self, *args, **kwargs)

    def _hook_backward(hook_self):
        """
        Overloads backward method used to compute gradients 
        of all the variables that are part of the computational 
        graph which produced self. Because native backward breaks 
        things (especially the .id attribute of the gradient), 
        we store the id of all variables we can access 
        (only the leaf variables of the graph) and we reconstruct our 
        variables correctly after backward was performed by basically 
        restoring the grad "envelope" (including its id)
        """
        sy.Variable.native_native_backward = sy.Variable.native_backward

        def new_backward(self, *args, **kwargs):
            worker = self.owner
            # Retrieve all the variable ids involved in the computation graph
            variable_ids = torch_utils.get_connected_variables(self)
            variable_ids = [var_id for var_id in variable_ids if var_id in worker._objects]
            # Save all the gradients (to keep the id) and reset the grads
            saved_grads = {}
            for variable_id in variable_ids:
                syft_tensor = worker.get_obj(variable_id)
                var = syft_tensor.parent
                assert var.id == variable_id
                saved_grads[variable_id] = var.grad
                var.grad = None

            # Performs the backward
            self.native_native_backward(*args, **kwargs)

            # Put back the original grad envelop and insert the new grad value in it
            for variable_id in variable_ids:
                syft_tensor = worker.get_obj(variable_id)
                # retrieve the var to fix
                var = syft_tensor.parent
                # retrieve the old grad, and insert it (to keep the chain) [first the envelope, then the data]
                saved_grad = saved_grads[variable_id]
                if saved_grad is not None:
                    # store the computed gradient
                    computed_grad = var.grad
                    var.assign_grad_(saved_grad)
                    # Insert the value of the computed_grad
                    if computed_grad is not None:
                        var.grad.data.native_set_(computed_grad.data)
                # Make sure everyone has the right owner
                torch_utils.enforce_owner(var, worker)
                # Fix the .data and .grad attributes on the chain
                torch_utils.link_var_chain_to_data_and_grad_chains(var, var.data, var.grad)

        sy.Variable.native_backward = new_backward

    def _hook_module(self):
        """Overloading for torch.nn.Module"""        
        def module_is_missing_grad(model):
            """Overloads missing grad parameter in model"""
            missing_grad = False
            for p in model.parameters():
                if p.grad is None:
                    missing_grad = True
            return missing_grad

        def create_grad_objects(model):
            """Overloads create grad parameter for model"""
            for p in model.parameters():
                o = p.sum()
                o.backward()
                p.grad -= p.grad

        def module_send_(self, dest):
            """Overloads send to remote for torch.nn.Module"""
            if (module_is_missing_grad(self)):
                create_grad_objects(self)

            for p in self.parameters():
                p.send(dest)

        torch.nn.Module.send = module_send_

        def module_get_(self):
            """Overload get from remote for torch.nn.Module"""
            for p in self.parameters():
                p.get()

        torch.nn.Module.get = module_get_


