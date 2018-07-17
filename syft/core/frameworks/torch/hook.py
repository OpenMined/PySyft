import torch
import random
import inspect
import re
import json
import types
import functools
import syft as sy
from ... import workers
from ... import utils
from .tensor import _SyftTensor, _LocalTensor, _PointerTensor, _FixedPrecisionTensor, _TorchTensor
from .tensor import _TorchVariable
class TorchHook(object):
    
    def __init__(self, local_worker=None, is_client=True, verbose=True, queue_size=0):
        
        self.local_worker = local_worker
        if (self.local_worker is None):

            # Every TorchHook instance should have a local worker which is responsible for
            # interfacing with other workers. The worker interface is what allows the Torch
            # specific code in TorchHook to be agnostic to the means by which workers communicate
            # (such as peer-to-peer, sockets, through local ports, or all within the same process)

            if (hasattr(torch, 'local_worker')):
                self.local_worker = torch.local_worker
                if (verbose):
                    print("Torch seems to already have a local_worker object... \
                          using that one instead...")
            else:
                self.local_worker = workers.VirtualWorker(
                    hook=self, is_client_worker=is_client, queue_size=queue_size)
        else:
            # if the local_worker already exists, then it MUST not know about the hook which is
            # just being created. Thus, we must inform it.
            self.local_worker.hook = self

        # Methods that caused infinite recursion during testing
        # TODO: May want to handle the ones in "exclude" manually at
        #       some point
        self.exclude = (['ndimension', 'nelement', 'size', 'numel',
                         'type', 'tolist', 'dim', '__iter__', 'select',
                         '__getattr__'])

        self.to_auto_overload = {}

        for typ in torch.tensorvar_types:
            self._hook_native_tensors_and_variables(typ)
            self._hook_syft_tensor_types(typ)


    def _hook_native_tensors_and_variables(self, tensor_type):
        """Overloading a given tensor_type"""
        # Overload 'special' methods here

        self._add_registration_to___init__(tensor_type, register_child_instead=True)

        self._hook_properties(tensor_type)

        self.to_auto_overload[tensor_type] = self._which_methods_should_we_auto_overload(tensor_type)

        self._rename_native_functions(tensor_type)

        self._assign_methods_to_use_child(tensor_type)

        self._add_methods_from__TorchObject(tensor_type)

    def _hook_syft_tensor_types(self, tensor_type):

        self._hook_LocalTensor(tensor_type)

        self._hook_SyftTensor(tensor_type)

        self._hook_PointerTensor(tensor_type)

    def _add_registration_to___init__(hook_self, tensorvar_type, register_child_instead=False):
        """Overloads tensor_type.__new__ or Variale.__new__"""

        if ('native___init__' not in dir(tensorvar_type)):
            tensorvar_type.native___init__ = tensorvar_type.__init__

        def new___init__(cls, *args, **kwargs):

            if('owner' in kwargs and kwargs['owner'] is not None):
                owner = kwargs['owner']
            else:
                owner = hook_self.local_worker

            if('id' in kwargs):
                id = kwargs['id']
            else:
                id = None


            if(register_child_instead):
                cls.native___init__()
                _ = cls.child
            else:
                cls.native___init__(*args, **kwargs)
                if('skip_register' in kwargs and kwargs['skip_register']):
                    if(id is None):
                        cls.id = random.randint(0, 1e10)
                    else:
                        cls.id = id

                    cls.owner = owner
                else:
                    owner.register_object(cls, owner=owner, id=id)

        tensorvar_type.__init__ = new___init__

    def _hook_properties(hook_self, tensor_type):
        @property
        def child(self):

            try:
                if (hasattr(self, '_child') and self._child is not None):
                    return self._child
                else:
                    self._child = _LocalTensor(child=self,
                                               parent=self,
                                               torch_type='syft.'+type(self).__name__)
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
                                               torch_type='syft.'+type(self).__name__)
                return self._child


        @child.setter
        def child(self, value):
            self._child = value

        tensor_type.child = child
        
        @property
        def id(self):
            return self.child.id
        
        tensor_type.id = id
        
        @property
        def owner(self):
            return self.child.owner
        
        tensor_type.owner = owner
    
    def _which_methods_should_we_auto_overload(self, tensor_type=torch.FloatTensor):
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

            if ((is_desc or (is_func and not is_service_func)) and not is_base and not is_old):
                to_overload.append(attr)

        return to_overload

    def _rename_native_functions(self, tensor_type):
        
        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)
                
            # if we haven't already overloaded this function
            if 'native_{}'.format(attr) not in dir(tensor_type):
                setattr(tensor_type, 'native_{}'.format(attr), lit)
            
            setattr(tensor_type, attr, None)

    def _assign_methods_to_use_child(self, tensor_type):

        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)
            passer = utils.pass_method_args(lit)
            new_attr = self._forward_call_to_child(passer, attr)
            
            # if we haven't already overloaded this method
            if attr not in dir(tensor_type) or getattr(tensor_type, attr) is None:
            
                setattr(tensor_type, attr, new_attr)

    def _add_methods_from__TorchObject(self, tensor_type):

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
                   '__subclasshook__']

        if issubclass(sy.Tensor, tensor_type):
            parent_syft_obj = _TorchTensor
        else:
            parent_syft_obj = _TorchVariable

        for attr in dir(parent_syft_obj):
            if attr not in exclude:
                if(attr in dir(tensor_type) and "native_"+str(attr) not in dir(tensor_type)):
                    setattr(tensor_type, "native_"+str(attr), getattr(tensor_type, attr))
                setattr(tensor_type, attr, getattr(parent_syft_obj, attr))
                
    def _hook_LocalTensor(self, tensor_type):
        
        # iterate through all methods and tell them to call the native function
        # on self.child
        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)
            passer = utils.pass_method_args(lit)
            new_attr = self._forward_call_to_child(passer, attr, call_native=True)
            
            # if we haven't already overloaded this method
            if attr not in dir(_LocalTensor) or getattr(_LocalTensor, attr) is None:
                setattr(_LocalTensor, attr, new_attr)
    
    def _hook_SyftTensor(self, tensor_type):
        
        self._add_registration_to___init__(_SyftTensor)
        
        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)
            passer = utils.pass_method_args(lit)
            new_attr = self._forward_call_to_child(passer, attr)
            
            # if we haven't already overloaded this method
            if attr not in dir(_SyftTensor) or getattr(_SyftTensor, attr) is None:
                setattr(_SyftTensor, attr, new_attr)

    def _hook_PointerTensor(self, tensor_type):

        for attr in self.to_auto_overload[tensor_type]:
            
            # # if we haven't already overloaded this method
            # if attr not in dir(_PointerTensor) or getattr(_PointerTensor, attr) is None:

            setattr(_PointerTensor, attr, self._forward_call_to_remote(attr))

    def _forward_call_to_remote(hook_self, attr):

        def _execute_remote_call(self, *args, **kwargs):
            command, tensorvars = self.compile_command(attr,
                          args,
                          kwargs,
                          True)

            response = self.owner.send_torch_command(recipient=self.location,
                                         message=command)
            return sy.deser(response, owner=self.owner).wrap()

        return _execute_remote_call

    def _forward_call_to_child(hook_self, method, attr, call_native=False):
        """
        Wrapper overloading partialmethod objects of Torch object
        methods.  Compiles command, checks for Tensors and Variables in
        the args/kwargs, determines locations of all Tensors and
        Variables involved in computation, and handles the computation
        accordingly.
        """

        def method_router(self, *args, **kwargs):
            """
            This is a routing function. If self is a local
            tensor (data stored locally), then it executes
            the call locally. If self is a remote tensor, it
            executes a call to a remote worker.
            """

            results = list()
            if(call_native):
                result = getattr(self.child, "native_"+attr)(*args, **kwargs)
            else:
                result = getattr(self.child, attr)(*args, **kwargs)
        
            if(type(result) in torch.tensorvar_types and (not hasattr(result, 'owner'))):
                hook_self.local_worker.register_object(result.child, owner=self.owner)

            return result


        return method_router
