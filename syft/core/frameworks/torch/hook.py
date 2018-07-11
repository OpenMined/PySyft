import torch
import inspect
import re
import json
import types
import functools
from ... import workers
from ... import utils
from .tensor import _SyftTensor, _LocalTensor, _PointerTensor, _FixedPrecisionTensor, _TorchTensor

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
                         'type', 'tolist', 'dim', '__iter__', 'select'])
        
        self.to_auto_overload = {}

        for typ in torch.tensor_types:

            self._hook_native_tensor(typ)

            # instantiating it seems to impact the propagation of the overrides
            # some of them (particularly __new__) don't seem to show up until
            # after a tensor has been initialized
#             x = typ([0])

        for typ in torch.tensor_types:

            self._hook_syft_tensor_types(typ)

            # instantiating it seems to impact the propagation of the overrides
            # some of them (particularly __new__) don't seem to show up until
            # after a tensor has been initialized            
#             x = typ([0])
    
    def _hook_native_tensor(self, tensor_type):
        """Overloading a given tensor_type"""
        # Overload 'special' methods here
        
        self._add_registration_to___init__(tensor_type, register_children_instead=True)
        
        self._hook_properties(tensor_type)
        
        self.to_auto_overload[tensor_type] = self._which_methods_should_we_auto_overload(tensor_type)

        self._rename_native_functions(tensor_type)
        
        self._assign_methods_to_use_children(tensor_type)
        
        self._add_methods_from__TorchTensor(tensor_type)

    def _hook_syft_tensor_types(self, tensor_type):

        self._hook_LocalTensor(tensor_type)
        
        self._hook_SyftTensor(tensor_type)
        
    def _add_registration_to___init__(hook_self, tensorvar_type, register_children_instead=False):
        """Overloads tensor_type.__new__ or Variale.__new__"""

        if ('native___init__' not in dir(tensorvar_type)):
            tensorvar_type.native___init__ = tensorvar_type.__init__

        def new___init__(cls, *args, **kwargs):
            if(register_children_instead):
                cls.native___init__()
                _ = cls.children
            else:
                cls.native___init__(*args, **kwargs)                
                hook_self.local_worker.register_object(cls, owners=[hook_self.local_worker])
#                 return result

        tensorvar_type.__init__ = new___init__

    def _hook_properties(hook_self, tensor_type):
        @property
        def children(self):
            if (hasattr(self, '_children') and self._children is not None):
                if(isinstance(self._children, list)):
                    return self._children
                else:
                    return [self._children]
            else:
                self._children = [_LocalTensor(self)]
                return self._children

        @children.setter
        def children(self, value):
            if(isinstance(value, list)):
                self._children = value
            else:
                self._children = [value]

        tensor_type.children = children
        
        @property
        def id(self):
            return self.children[0].id
        
        tensor_type.id = id
        
        @property
        def owners(self):
            return set(reduce(lambda x,y:x+y,list(child.owners for child in self.children)))
        
        tensor_type.owners = owners
    
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

    def _assign_methods_to_use_children(self, tensor_type):

        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)
            passer = utils.pass_method_args(lit)
            new_attr = self._forward_call_to_children(passer, attr)
            
            # if we haven't already overloaded this method
            if attr not in dir(tensor_type) or getattr(tensor_type, attr) is None:
            
                setattr(tensor_type, attr, new_attr)

    def _add_methods_from__TorchTensor(self, tensor_type):

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
        for attr in dir(_TorchTensor):
            if attr not in exclude:
                if(attr in dir(tensor_type) and "native_"+str(attr) not in dir(tensor_type)):
                    setattr(tensor_type, "native_"+str(attr), getattr(tensor_type, attr))
                setattr(tensor_type, attr, getattr(_TorchTensor, attr))
    # def _hook_send_(hook_self, tensorvar_type):
    #     def send_(self, workers):
    #         """
    #         Sends a Tensor or Variable object to a (sequence of) Grid workers.

    #         Args:
    #         workers: string (or sequence) containing IPFS address(es)
    #             of worker node(s).
    #         """

    #         # makes singleton, if needed
    #         workers = hook_self.local_worker._check_workers(self, workers)

    #         for worker in workers:
    #             hook_self.local_worker.send_obj(self,
    #                                             worker)


    #     setattr(tensorvar_type, 'send_', send_)
    #     setattr(tensorvar_type, 'send', send_)
                
    def _hook_LocalTensor(self, tensor_type):
        
        # iterate through all methods and tell them to call the native function
        # on self.child
        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)
            passer = utils.pass_method_args(lit)
            new_attr = self._forward_call_to_children(passer, attr, call_native=True)
            
            # if we haven't already overloaded this method
            if attr not in dir(_LocalTensor) or getattr(_LocalTensor, attr) is None:
                setattr(_LocalTensor, attr, new_attr)
    
    def _hook_SyftTensor(self, tensor_type):
        
        self._add_registration_to___init__(_SyftTensor)        
        
        for attr in self.to_auto_overload[tensor_type]:

            lit = getattr(tensor_type, attr)
            passer = utils.pass_method_args(lit)
            new_attr = self._forward_call_to_children(passer, attr)
            
            # if we haven't already overloaded this method
            if attr not in dir(_SyftTensor) or getattr(_SyftTensor, attr) is None:
                setattr(_SyftTensor, attr, new_attr)
    
    def _forward_call_to_children(hook_self, method, attr, call_native=False):
        """
        Wrapper overloading partialmethod objects of Torch object
        methods.  Compiles command, checks for Tensors and Variables in
        the args/kwargs, determines locations of all Tensors and
        Variables involved in computation, and handles the computation
        accordingly.
        """

        @functools.wraps(method)
        def method_router(self, *args, **kwargs):
            """
            This is a routing function. If self is a local
            tensor (data stored locally), then it executes
            the call locally. If self is a remote tensor, it
            executes a call to a remote worker.
            """
            results = list()
            if(call_native):
                for child in self.children:
                    results.append(getattr(child, "native_"+attr)(*args, **kwargs))
            else:
                for child in self.children:
                    results.append(getattr(child, attr)(*args, **kwargs))

            for result in results:
                if(type(result) in torch.tensorvar_types and (not hasattr(result, 'owner'))):
                    _children = list()
                    for _child in result.children:
                        _children.append(hook_self.local_worker.register_object(_child))
                    result.children = _children
            if(isinstance(results, list) and len(results) == 1):
                return results[0]
            else:
                return results


        return method_router
