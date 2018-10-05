import json
import msgpack
import time
import re
import torch
import random
import syft as sy
from . import utils as torch_utils
from .. import encode
from ... import utils
import logging
import numpy as np
from syft.spdz import spdz
from syft.mpc.securenn import relu, relu_deriv


class _SyftTensor(object):
    """
    Super class for all Syft tensors, that contains all the specific syft functions
    """

    def __init__(self, child=None, parent=None, torch_type=None, owner=None, id=None, skip_register=False):
        if torch_utils.is_syft_tensor(child):
            if torch_type is None:
                torch_type = child.torch_type
            if owner is None:
                owner = child.owner

        self.id = id
        # self.old_ids = None - this will only get initialized if self.set_id() is called, but i'm referencing it
        # in this comment so that people know it can exist. It's a set()

        self.child = child
        self.parent = parent
        self.torch_type = torch_type

        if self.child is not None:
            try:
                self.child.parent = self
            except AttributeError:  # for non-torch tensor child (can occur in __repr__)
                pass

        if owner is not None:
            if not isinstance(owner, sy.core.workers.BaseWorker):
                owner = self.child.owner.get_worker(owner)
        self.owner = owner

    def __str__(self):
        return "[" + type(self).__name__ + " - id:" + str(self.id) + " owner:" + str(
            self.owner.id) + "]"

    def __repr__(self):
        return self.__str__()

    def get_shape(self):
        if torch_utils.is_tensor(self.child) or torch_utils.is_variable(self.child):
            return self.child.shape
        else:
            return self.child.get_shape()

    def share(self, *workers):
        if torch_utils.is_variable(self.child):
            response = self.child.share(*workers)
            self.child = response
            self.data.child = response.data
            self.grad.child = response.grad
            self.grad.data.child = response.grad.data
            r = self.wrap(True)
            r.data.child = self.data
            r.init_grad_()
            r.grad.child = self.grad
            r.grad.data.child = self.grad.data
            return r
        else:
            return self.wrap(True).share(*workers)

    def set_id(self, new_id):
        """
        This changes the id of a tensor.
        :param new_id: a string or integer id
        :return: returns self, for convenience.
        """
        if new_id not in self.owner._objects:
            if not hasattr(self, 'old_ids'):
                self.old_ids = set()

            self.old_ids.add(self.id)

            self.owner.register_object(self, new_id)
            return self
        else:
            raise KeyError("There is already a tensor with that ID - please choose another.")

    @property
    def parent(self):
        if hasattr(self, '_parent') and self._parent is not None:
            return self._parent
        else:
            return None  # Parents should be manually specified

    @parent.setter
    def parent(self, value):
        self._parent = value

    @classmethod
    def handle_call(cls, command, owner):
        """
        Receive a command and an owner and before sending it downward the syft chain,
        Performs different operations like:
        - command substitution
        - args substitution
        - command overloading with special methods or arguments
        """
        attr = command['command']
        args = command['args']
        kwargs = command['kwargs']
        has_self = command['has_self']

        # Overload methods
        if has_self and cls.is_overloaded_method(attr):
            self_ = command['self']
            result = getattr(self_, attr)(*args, **kwargs)
        # Overload functions
        elif not has_self and cls.is_overloaded_function(attr):
            overload_function = cls.overload_functions.get(attr)
            result = overload_function(*args, **kwargs)
        else:
            # replace a function attr with an existing other
            if attr in cls.replaced_functions():
                command['command'] = cls.replaced_functions(attr)

            # Or do whatever you want, but be careful not to overwrite the args!
            # (...)

            # Get the next node type and update in command tensorvar with tensorvar.child
            next_command, child_type = torch_utils.prepare_child_command(
                command, replace_tensorvar_with_child=True)

            # Forward the call to the next child
            result = child_type.handle_call(next_command, owner)

        if result is None:
            return result

        if not isinstance(result, (int, float, str, bool)):
            # Insert the new node just before the wrapper
            syft_response = cls.syft_wrap(result, owner)
        else:
            syft_response = result

        return syft_response

    def create_pointer(self, parent=None, ptr_id=None, owner=None, location=None,
                       id_at_location=None, register=False):

        if owner is None:
            owner = self.owner
        if isinstance(owner, (str, int)):
            owner = self.owner.get_worker(owner)

        local_pointer = False
        if location is None:
            location = self.owner.id
            local_pointer = True

        if id_at_location is None:
            id_at_location = self.id

        if ptr_id is not None:
            if ptr_id == id_at_location:
                raise AttributeError(
                    "The PointerTensor and the tensor being pointed to cannot have the same id.")

        else:
            # Normally if there is no id specified, we keep the same as the original pointer
            # Except if the pointer is local (we don't want to overwrite it!)
            if not local_pointer:
                ptr_id = self.id
            else:
                ptr_id = random.randint(0, 10e10)

        if hasattr(self, 'torch_type') and self.torch_type is not None:
            torch_type = self.torch_type
        else:
            torch_type = None
            logging.warning("The torch tensor's child has no torch_type. Is it well formed?")

        previous_pointer = owner.get_pointer_to(location, id_at_location)
        if previous_pointer is None:
            ptr = _PointerTensor(child=None,
                                 parent=parent,
                                 id=ptr_id,
                                 torch_type=torch_type,
                                 location=location,
                                 id_at_location=id_at_location,
                                 owner=owner,
                                 skip_register=(not register))
            if not register:
                ptr.owner.rm_obj(ptr.id)
        else:
            ptr = previous_pointer

        return ptr

    def ser(self, private, as_dict=True):
        """
        General method for serializing a Syft object. Specific tensors like _PointerTensor
        should overload this method.
        """
        data = {
            'owner': self.owner.id,
            'id': self.id,
            'torch_type': self.torch_type
        }
        if self.child is not None and not torch_utils.is_tensor(self.child):
            data['child'] = self.child.ser(private, as_dict)

        if as_dict:
            return {'__{}__'.format(self.__class__.__name__): data}
        else:
            return msgpack.packb({'__{}__'.format(self.__class__.__name__): data}, use_bin_type=True)

    @classmethod
    def deser_routing(cls, obj_type, obj, worker, acquire):
        """
        Method analysing the dict given to see which Syft Tensor should deserialized,
        and forwarding the call

        [Is this case note that the dct param is assumed to have a single key, which is
        compatible with our encode/decode process (ex: {'___PointerTensor__': {...} })]
        """
        syft_code = torch.syft_tensor_codes[obj_type]
        if syft_code == torch.syft_tensor_codes._LocalTensor:
            return sy._LocalTensor.deser(obj, worker, acquire)
        elif syft_code == torch.syft_tensor_codes._PointerTensor:
            return sy._PointerTensor.deser(obj, worker, acquire)
        else:
            syft_type = torch.guard[obj_type]
            return syft_type.deser(obj, worker, acquire)

        raise Exception("could not deserialize an object sent to router\n"+str(dct))

    @classmethod
    def deser(cls, msg_obj, worker, acquire):
        """
        General method for de-serializing a Syft object. Specific tensors like _PointerTensor
        should overload this method.
        """
        if acquire:  # We need to register the info given
            syft_obj = cls(child=None,
                           parent=None,
                           torch_type=msg_obj['torch_type'],
                           owner=worker,
                           id=msg_obj['id'],
                           skip_register=True
                           )
            if 'child' in msg_obj:
                child_type, child_obj = torch_utils.extract_type_and_obj(msg_obj['child'])
                syft_child = cls.deser_routing(child_type, child_obj, worker, acquire)
                syft_obj.child = syft_child
                syft_child.parent = syft_obj

        else:  # We point at the info which generally we can't really have
            # We make sure we are not creating a duplicate pointer
            previous_pointer = worker.get_pointer_to(msg_obj['owner'], msg_obj['id'])
            if previous_pointer is None:
                syft_obj = sy._PointerTensor(child=None,
                                             parent=None,
                                             torch_type=msg_obj['torch_type'],
                                             location=msg_obj['owner'],
                                             id_at_location=msg_obj['id'],
                                             owner=worker,
                                             id=None,
                                             skip_register=True)
            else:
                syft_obj = previous_pointer
        return syft_obj

    def on(self, wrapper):
        """
        Used to add a new node at the top of the chain, just before the tensorvar wrapper

        Example with _PlusIsMinusTensor:
        x = sy.FloatTensor([1, 2, 3])       # the chain is FloatTensor > _LocalTensor
        x = sy._PlusIsMinusTensor().on(x)   # the chain is FloatTensor > _PlusIsMinusTensor > _LocalTensor
        """

        cls = type(self)
        # Assign the newly created tensor to the good owner and torch_type
        self.torch_type = wrapper.child.torch_type
        self.owner = wrapper.child.owner

        # Insert self between wrapper and wrapper child
        torch_utils.wrap_command_with(wrapper.child, wrapper=self)
        torch_utils.wrap_command_with(self, wrapper=wrapper)

        # In case wrapper is a variable, do the same with data and grad (if necessary)
        if torch_utils.is_variable(wrapper):
            wrapper.data = cls().on(wrapper.data)
            if torch_utils.is_variable(wrapper.grad):
                wrapper.grad = cls().on(wrapper.grad)
            if wrapper.grad is None and wrapper.data.dim() > 0:
                # create an empty envelope in wrapper.grad
                wrapper.init_grad_()
                # Build the chain with _PlusIsMinusTensor
                wrapper_grad = cls().on(wrapper.grad)
                # Insert the gradient within its chain
                wrapper.grad.native_set_(wrapper_grad)

        return wrapper

    def wrap(self, skip_fix_chain_end=False):
        """
        Wrap a syft node with a torch wrapper
        """
        wrapper = torch.guard[self.torch_type]()
        self.owner.rm_obj(wrapper.child.id)
        wrapper.child = self
        self.parent = wrapper
        if not skip_fix_chain_end:
            torch_utils.fix_chain_ends(wrapper)
        return wrapper

    @classmethod
    def syft_wrap(cls, result, owner):
        """
        Wrap a torch node with a syft wrapper
        """

        if(torch_utils.is_tensor(result)):
            return result

        # Insert the new syft node just before the wrapper
        syft_wrapper = cls(child=result, owner=owner)
        result.parent = syft_wrapper

        if torch_utils.is_variable(result.torch_type):
            syft_response_data = cls(child=result.data, owner=owner)
            result.data.parent = syft_response_data
            syft_wrapper.data = syft_response_data
            # TODO: same for grad ?

        return syft_wrapper

    @classmethod
    def is_overloaded_method(cls, attr):
        """
        State if a function name corresponds to a Syft Tensor method which
        overloads a torch method
        """
        exclude = ('on', '__init__', 'native___init__', '__repr__', '__str__', 'create_pointer',
                   'ser', 'deser', 'handle_call')
        if attr in exclude:
            return False
        if hasattr(getattr(cls, attr), '__module__') \
                and getattr(cls, attr).__module__ == 'syft.core.frameworks.torch.tensor':
            return True
        return False

    @classmethod
    def is_overloaded_function(cls, attr):
        """
        State if a function name corresponds to an overloaded function by the Syft
        tensor, which declared the corresponding overloading function in
        cls.overload_functions
        """
        attr = attr.split('.')[-1]
        overloaded_functions = [
            func for func in dir(cls.overload_functions)
                 if re.match(r'__(.*)__', func) is None
                 and func != 'get'
        ]
        return attr in overloaded_functions

    @classmethod
    def replaced_functions(cls, attr=None):
        """
        If attr is none, return all the function substitution a Syft Tensor class
        wants to perform.
        Else, return the substitution corresponding to attr
        """
        if attr is None:
            return cls.substitution_table
        else:
            return cls.substitution_table[attr]

    substitution_table = {}

    class overload_functions:
        pass


class _LocalTensor(_SyftTensor):

    def __init__(self, child=None, parent=None, torch_type=None, owner=None, id=None, skip_register=False):
        super().__init__(child=child, parent=parent, torch_type=torch_type, owner=owner, id=id,
                         skip_register=skip_register)

    @classmethod
    def handle_call(cls, syft_command, owner):
        """
        Execute a forwarded command on the native tensor with native operations.
        Receive a syft command and an owner, and converts it into command with
        native torch args. Excute native operations and converts it back into
        syft response using _LocalTensors.
        """
        # start_time = time.time()
        tensor_command, torch_type = torch_utils.prepare_child_command(syft_command,
                                                                       replace_tensorvar_with_child=True)
        torch_utils.assert_has_only_torch_tensorvars(tensor_command)

        attr = tensor_command['command']
        args = tensor_command['args']
        kwargs = tensor_command['kwargs']
        has_self = tensor_command['has_self']

        if has_self:
            self = tensor_command['self']
            attr = torch._command_guard(attr, 'tensorvar_methods')
            command = getattr(self, "native_" + attr)
        else:
            attr = torch._command_guard(attr, 'torch_modules')
            elems = attr.split('.')
            elems[-1] = 'native_' + elems[-1]
            native_func_name = '.'.join(elems)
            command = eval(native_func_name)
        # torch.handle_call_timer += time.time() - start_time
        response = command(*args, **kwargs)
        # start_time = time.time()

        # TODO : control registration process
        if response is None:
            # torch.handle_call_timer += time.time() - start_time
            return response

        if owner.id != owner.hook.local_worker.id:
            if isinstance(response, (int, float, bool)):
                response = torch_type([response])
            elif isinstance(response, (np.ndarray, )):
                logging.warning("[np.ndarray] Hardcoding FloatTensor")
                response = sy.FloatTensor(response)
        else:
            if isinstance(response, (int, float, bool, np.ndarray)):
                # torch.handle_call_timer += time.time() - start_time
                return response

        # If the command is an in-place method, wrap self and return
        if has_self and utils.is_in_place_method(attr):
            # wrap the main element
            torch_utils.wrap_command_with(response, syft_command['self'])

            if torch_utils.is_variable(response):
                # Also wrap the data if it's a variable (don't use wrap_command_with: the chain is not well formed yet)
                syft_command['self'].child.data = response.data
                response.data.parent = syft_command['self'].child.data.parent
                # And wrap the grad if there is one
                if response.grad is not None:
                    if response.grad.data.dim() > 0:
                        syft_command['self'].child.grad = response.grad
                    else:
                        syft_command['self'].child.grad.native_set_()
                    response.grad.parent = syft_command['self'].child.grad.parent
                # Finally, fix the links .data and .grad
                if response.grad is None:
                    torch_utils.link_var_chain_to_data_chain(syft_command['self'], response.data.child)
                else:
                    torch_utils.link_var_chain_to_data_and_grad_chains(syft_command['self'], response.data.child, response.grad.child)

            return_response = syft_command['self']

        elif hasattr(response, 'child') and (isinstance(response.child, (_SPDZTensor, _SNNTensor, _FixedPrecisionTensor))):
            return response
        # Else, the response if not self. Iterate over the response(s) and wrap with a syft tensor
        else:

            responses = response if isinstance(response, tuple) else (response,)
            syft_responses = []
            for resp in responses:
                if resp is None:  # Don't wrap None
                    syft_responses.append(resp)
                    continue

                if isinstance(resp, (int, float, bool)):
                    # if not final worker, convert into Float Tensor, which comes with a _LocalTensor
                    if owner.id != owner.hook.local_worker.id:
                        resp = sy.zeros(1) + resp
                    else:  # Else don't wrap it
                        syft_responses.append(resp)
                        continue

                syft_response = sy._LocalTensor(child=resp, parent=resp, owner=owner,
                                                torch_type='syft.' + type(resp).__name__)

                if torch_utils.is_variable(resp):
                    if resp.grad is None:
                        torch_utils.link_var_chain_to_data_chain(syft_response, resp.data.child)
                    else:
                        torch_utils.link_var_chain_to_data_and_grad_chains(syft_response, resp.data.child, resp.grad.child)

                syft_responses.append(syft_response)

            return_response = tuple(syft_responses) if len(syft_responses) > 1 else syft_responses[0]

        # torch.handle_call_timer += time.time() - start_time
        return return_response

    def ser(self, private, as_dict=True):

        data = {
            'owner': self.owner.id,
            'id': self.id,
            'torch_type': self.torch_type
        }

        if as_dict:
            return {'___LocalTensor__': data}
        else:
            return msgpack.packb({'___LocalTensor__': data}, use_bin_type=True)

    @staticmethod
    def deser(msg_obj, worker, acquire):

        if 'owner' not in msg_obj:
            raise TypeError("sy._LocalTensor can't deserialize a non-valid sy._LocalTensor. "
                            "Do you wan to call sy.FloatTensor.deser() instead?")
        if msg_obj['owner'] == worker.id:
            logging.warning('_LocalTensor sent to itself')
        if acquire:  # We need to register the info given
            syft_obj = sy._LocalTensor(child=None,
                                       parent=None,
                                       torch_type=msg_obj['torch_type'],
                                       owner=worker,
                                       id=msg_obj['id'],
                                       skip_register=True
                                       )
        else:  # We point at the info which generally we can't really have
            # We make sure we are not creating a duplicate pointer
            previous_pointer = worker.get_pointer_to(msg_obj['owner'], msg_obj['id'])
            if previous_pointer is None:
                syft_obj = sy._PointerTensor(child=None,
                                             parent=None,
                                             torch_type=msg_obj['torch_type'],
                                             location=msg_obj['owner'],
                                             id_at_location=msg_obj['id'],
                                             owner=worker,
                                             id=None,
                                             skip_register=True)
            else:
                syft_obj = previous_pointer
        return syft_obj

    def get(self, parent, deregister_ptr=None):
        raise TypeError("Cannot call .get() on a tensor you already have.")

class _WrapTorchObjectPlusIsMinusTensor(_SyftTensor):
    """
    Example of a custom overloaded SyftTensor wherein the .child
    object is also a TorchObject (such as FloatTensor or LongTensor).
    Once implemented, you can wrap an existing tensor like.

    x = torch.LongTensor([[1,2],[3,4]])
    torch_type='syft.LongTensor'
    fpt = _WrapTorchObjectTensorPlusIsMinus(x, torch_type=tt).wrap(True)

    and then commands will automatically get executed on the child

    y = fpt + fpt

    after which y equals
     2  4
     6  8
    [syft.core.frameworks.torch.tensor.LongTensor of size 2x2]

    A production example of this tensor is _SPDZTensor

    """
    def __init__(self, child=None, owner=None, torch_type=None):
        super().__init__(child=child, owner=owner)

        self.child = child
        self.torch_type = torch_type

        # The table of command you want to replace

    def on(self, shares):
        return self.wrap(True)



    @classmethod
    def handle_call(cls, command, owner):
        """
        This is a special handle_call method which is compatible with
        .child objects that are themselves torch objects (wrappers) of
        other methods.
        :param command:
        :param owner:
        :return:
        """
        attr = command['command']
        args = command['args']
        kwargs = command['kwargs']
        self = command['self']

        if (attr == '__add__'):
            return cls.__add__(self, *args, **kwargs)
        else:
            result_child = getattr(self.child, attr)(*args, **kwargs)
            return _WrapTorchObjectPlusIsMinusTensor(result_child).wrap(True)

    def __add__(self, other):
        # gp_ stands for GeneralizedPointer
        gp_response = self.child - other.child
        response = _WrapTorchObjectPlusIsMinusTensor(gp_response).wrap(True)
        return response


class _PlusIsMinusTensor(_SyftTensor):
    """
    Example of a custom overloaded _SyftTensor where the .child
    object is NOT a torch tensor (instead the wrapper of the child
    is re-purposed to the wrapper of this tensor)

    Role:
    Converts all add operations into sub/minus ones.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # The table of command you want to replace
    substitution_table = {
        'torch.add': 'torch.add'
    }

    class overload_functions:
        """
        Put here the functions you want to overload
        Beware of recursion errors.
        """
        @staticmethod
        def add(x, y):
            return x.add(y)

        @staticmethod
        def get(attr):
            attr = attr.split('.')[-1]
            return getattr(sy._PlusIsMinusTensor.overload_functions, attr)

    # Put here all the methods you want to overload
    def add(self, arg):
        """
        Overload the add method and execute another function or method with the provided args
        """
        _response = self.sub(arg)

        return _response

    def abs(self):
        """
        Overload the abs() method and execute another function
        """
        return torch.abs(self)


class _GeneralizedPointerTensor(_SyftTensor):

    def __init__(self, pointer_tensor_dict, parent=None, torch_type=None, id=None, owner=None, skip_register=False):
         super().__init__(child=None, parent=parent, torch_type=torch_type, owner=owner, id=id,
                         skip_register=skip_register)
         pointer_dict = {}
         for worker, pointer in pointer_tensor_dict.items():
             if not isinstance(pointer, sy._PointerTensor):
                 if isinstance(pointer.child, sy._PointerTensor):
                     pointer_tensor_dict[worker] = pointer.child
                 else:
                    raise TypeError('Passed in non-pointer'+str(type(pointer))+' to GeneralizedPointerTensor')
             key = worker if isinstance(worker, (int, str)) else worker.id
             pointer_dict[key] = pointer
         self.pointer_tensor_dict = pointer_dict
         self.torch_type = torch_type

    def get_shape(self):
        return list(self.pointer_tensor_dict.values())[0].get_shape()

    def ser(self, private, as_dict=True):
        pointer_dict = {}

        for owner,pointer in self.pointer_tensor_dict.items():
            pointer_dict[owner] = pointer.ser(private=private, as_dict=True)

        data = {
            'owner': self.owner.id,
            'id': self.id,
            'pointer_tensor_dict': pointer_dict,
            'torch_type': self.torch_type
        }
        if as_dict:
            return {'___GeneralizedPointerTensor__': data}
        else:
            return msgpack.packb({'___GeneralizedPointerTensor__': data}, use_bin_type=True)

    @classmethod
    def deser(cls, msg_obj, worker, acquire):

        pointer_tensor_dict = {}
        for owner_id, pointer in msg_obj['pointer_tensor_dict'].items():
            obj = _PointerTensor.deser(pointer['___PointerTensor__'], worker, acquire)
            pointer_tensor_dict[owner_id] = obj

        result = _GeneralizedPointerTensor(pointer_tensor_dict,
                                           owner=worker,
                                           id=msg_obj['id'],
                                           torch_type=msg_obj['torch_type'])
        return result

    @classmethod
    def handle_call(cls, syft_command, owner):

        syft_commands = torch_utils.split_to_pointer_commands(syft_command)
        result_dict = {}
        torch_type = None
        var_data_type = None
        for worker_id in syft_commands.keys():
            syft_command = syft_commands[worker_id]
            result_dict[worker_id] = sy._PointerTensor.handle_call(syft_command, owner)
            if torch_type is None:
                torch_type = result_dict[worker_id].torch_type
                if torch_utils.is_variable(torch_type):
                    var_data_type = result_dict[worker_id].data.torch_type

        gpt = _GeneralizedPointerTensor(result_dict, torch_type=torch_type, owner=owner)

        if torch_utils.is_variable(torch_type):
            gpt.child = torch.guard[torch_type]()
            data_pointer_dict = {
                w: p.data
                for w, p in gpt.pointer_tensor_dict.items()
            }
            gpt.data = _GeneralizedPointerTensor(data_pointer_dict, torch_type=var_data_type, owner=owner)
            gpt.data.child = torch.guard[var_data_type]()

            grad_pointer_dict = {
                w: p.grad
                for w, p in gpt.pointer_tensor_dict.items()
            }
            gpt.grad = _GeneralizedPointerTensor(grad_pointer_dict, torch_type=torch_type, owner=owner)
            gpt.grad.child = torch.guard[torch_type]()

            grad_data_pointer_dict = {
                w: p.grad.data
                for w, p in gpt.pointer_tensor_dict.items()
            }
            gpt.grad.data = _GeneralizedPointerTensor(grad_data_pointer_dict, torch_type=var_data_type, owner=owner)
            gpt.grad.data.child = torch.guard[var_data_type]()

        else:  # else tensor
            gpt.child = torch.guard[torch_type]([])
        return gpt

    def public_add_(self, value):
        for worker, pointer in self.pointer_tensor_dict.items():
            location = pointer.location
            value.send(location)
            torch_sum = pointer.parent + value
            self.pointer_tensor_dict[worker] = torch_sum.child
            break
        return self

    def get(self, deregister_ptr=False):

        # TODO: deregister_ptr doesn't work

        res = []
        for worker, pointer in self.pointer_tensor_dict.items():
            res.append(pointer.get())
        return res

    def sum_get(self):
        shares = self.get()
        res = None
        for share in shares:
            if res is None:
                res = share
            else:
                if len(share.size()) > 0:
                    res += share
        return res

    def workers(self):
        return list(self.pointer_tensor_dict.keys())

    def on(self, wrapper):
        """
        Used to add a new _GeneralizedPointerTensor at the top of the chain, just before the tensorvar wrapper
        """
        # Assign the newly created tensor to the good owner and torch_type
        self.torch_type = wrapper.child.torch_type
        self.owner = wrapper.child.owner

        # Insert self between wrapper and wrapper child
        torch_utils.wrap_command_with(wrapper.child, wrapper=self)
        torch_utils.wrap_command_with(self, wrapper=wrapper)
        self.child = None

        # In case wrapper is a variable, do the same with data and grad (if necessary)
        if torch_utils.is_variable(wrapper):
            try:
                data_pointer_dict = {
                    w: p.data
                    for w, p in self.pointer_tensor_dict.items()
                }
                wrapper.data = _GeneralizedPointerTensor(data_pointer_dict).on(wrapper.data)
            except AttributeError:
                pass
            if torch_utils.is_variable(wrapper.grad):
                grad_pointer_dict = {
                    w: p.grad
                    for w, p in self.pointer_tensor_dict.items()
                }
                wrapper.assign_grad_(_GeneralizedPointerTensor(grad_pointer_dict).on(wrapper.grad))

                # grad_data_pointer_dict = {
                #     w: p.grad.data
                #     for w, p in self.pointer_tensor_dict.items()
                # }
                # wrapper.grad.data = _GeneralizedPointerTensor(grad_data_pointer_dict).on(wrapper.grad.data)

        return wrapper


class _PointerTensor(_SyftTensor):

    def __init__(self, child, parent, torch_type, location=None, id_at_location=None, id=None,
                 owner=None, skip_register=False):
        super().__init__(child=child, parent=parent, torch_type=torch_type, owner=owner, id=id,
                         skip_register=skip_register)
        if location is None:
            raise AttributeError("Pointer must have a location specified")
        self.location = self.owner.get_worker(location)
        self.id_at_location = id_at_location
        self.torch_type = torch_type

        self.register_pointer()

        # pointers to themselves that get registered should trigger the flat
        # if it's not getting registered the pointer is probably about to be
        # sent over the wire
        if self.location == self.owner and not skip_register:
            logging.warning("Do you really want a pointer pointing to itself? (self.location == self.owner)")

    def share(self, *workers):

        worker_ids = list()
        for worker in workers:
            if hasattr(worker, 'id'):
                worker_ids.append(worker.id)
            else:
                worker_ids.append(worker)

        cmd = {}
        cmd['command'] = "share"
        cmd['args'] = worker_ids
        cmd['kwargs'] = {}
        cmd['has_self'] = True
        cmd['self'] = self

        response = self.handle_call(cmd, self.owner)

        return response

    def register_pointer(self):
        worker = self.owner
        if isinstance(self.location, int):
            location = self.location
        else:
            location = self.location.id
        id_at_location = self.id_at_location
        # Add the remote location worker key if needed
        if location not in worker._pointers.keys():
            worker._pointers[location] = {}
        # Add the remote address
        worker._pointers[location][id_at_location] = self.id

    @classmethod
    def handle_call(cls, syft_command, owner):
        """
        _PointerTensor has an overloaded handle_call function because it converts
        the command to torch tensors and send it over the network
        """
        #start_time = time.time()
        tensor_command = torch_utils.wrap_command(syft_command)

        attr = tensor_command['command']
        args = tensor_command['args']
        kwargs = tensor_command['kwargs']
        has_self = tensor_command['has_self']
        self_ = tensor_command['self'] if has_self else None

        command, locations, owners = torch_utils.compile_command(attr,
                                                                 args,
                                                                 kwargs,
                                                                 has_self=has_self,
                                                                 self=self_)
        location = locations[0]
        owner = owners[0]

        #torch.handle_call_timer += time.time() - start_time
        # Else we send the command
        response = owner.send_torch_command(recipient=location, message=command)
        # start_time = time.time()

        torch_utils.assert_has_only_torch_tensorvars(response)

        # If the command is an in-place method, we only need to return the same wrapper to the same
        # pointer, instead jof returning the new wrapper created in response
        if has_self and utils.is_in_place_method(attr):
            return syft_command['self']

        if torch_utils.is_variable(response):
            torch_utils.link_var_chain_to_data_and_grad_chains(response, response.data, response.grad)

        # Perform the un-wrap: remove the head on all chains (also .data and .grad if any)
        response, _ = torch_utils.get_child_command(response)
        # response is now a _Pointer, with a .data attr which is a _Pointer, etc.
        # torch.handle_call_timer += time.time() - start_time
        return response

    def __str__(self):
        return "[" + type(self).__name__ + " - id:" + str(self.id) + " owner:" + str(
            self.owner.id) + " loc:" + str(self.location.id) + " id@loc:" + str(
            self.id_at_location) + "]"

    def ser(self, private, as_dict=True):
        data = {
            'owner': self.owner.id,
            'id': self.id,
            'location': self.location.id,
            'id_at_location': self.id_at_location,
            'torch_type': self.torch_type
        }
        if as_dict:
            return {'___PointerTensor__': data}
        else:
            return msgpack.packb({'___PointerTensor__': data}, use_bin_type=True)

    @classmethod
    def deser(cls, msg_obj, worker, acquire):
        # If local, we render the object or syft object
        if msg_obj['location'] == worker.id:
            syft_obj = worker.get_obj(msg_obj['id_at_location'])
        else:
            if acquire:  # If there is data transmission, data being here Pointer
                # We acquire the tensor pointer
                previous_pointer = worker.get_pointer_to(msg_obj['owner'], msg_obj['id'])
                if previous_pointer is None:
                    syft_obj = cls(child=None,
                                   parent=None,
                                   torch_type=msg_obj['torch_type'],
                                   location=msg_obj['location'],
                                   id_at_location=msg_obj['id_at_location'],
                                   owner=worker,
                                   id=msg_obj['id'],
                                   skip_register=True)
                else:
                    syft_obj = previous_pointer
            else:  # We point at the Pointer (same part as every syft tensors)
                previous_pointer = worker.get_pointer_to(msg_obj['owner'], msg_obj['id'])
                if previous_pointer is None:
                    syft_obj = sy._PointerTensor(child=None,
                                                 parent=None,
                                                 torch_type=msg_obj['torch_type'],
                                                 location=msg_obj['owner'],
                                                 id_at_location=msg_obj['id'],
                                                 owner=worker,
                                                 id=None,
                                                 skip_register=True)
                else:
                    syft_obj = previous_pointer
        return syft_obj

    def get(self, deregister_ptr=True):
        """
            Get back from a remote worker the chain this pointer is pointing at
        """
        # Remove this pointer - TODO: call deregister function instead of doing it by hand
        if deregister_ptr:
            if self.torch_type == 'syft.Variable':
                if hasattr(self.parent, 'data'):
                    self.owner.rm_obj(self.parent.data.child.id)
            self.owner.rm_obj(self.id)

        # if the pointer happens to be pointing to a local object,
        # just return that object (this is an edge case)
        if self.location == self.owner:
            return self.owner.get_obj(self.id_at_location).child

        # get SyftTensor (Local or Pointer) from remote machine
        tensorvar = self.owner.request_obj(self.id_at_location, self.location)
        torch_utils.assert_has_only_torch_tensorvars(tensorvar)

        syft_tensor = tensorvar.child
        syft_tensor.id = self.id
        if self.torch_type == 'syft.Variable':
            if hasattr(self.parent, 'data'):
                tensorvar.data.child.id = self.parent.data.child.id

        # Register the result
        self.owner.register(syft_tensor)
        if syft_tensor.torch_type == 'syft.Variable':
            if hasattr(self.parent, 'data'):
                self.owner.register(tensorvar.data.child)

        torch_utils.fix_chain_ends(tensorvar)

        return tensorvar

    def get_shape(self):
        cmd = {
            'command': 'get_shape',
            'self': self,
            'args': [],
            'kwargs': {},
            'has_self': True
        }
        shape_ptr = self.handle_call(cmd, self.owner)

        if not isinstance(shape_ptr, tuple):
            shape_ptr = (shape_ptr, )

        raw_output = self.handle_call(cmd, self.owner)

        if(isinstance(raw_output, (tuple, list))):
            dims = list()
            for each in raw_output:
                dims.append(each.get().int().tolist())
                if(len(dims[-1]) == 1):
                    dims[-1] = dims[-1][0]

            return sy.Size(dims)

        return sy.Size(raw_output.get().int().tolist())

    def decode(self):
        raise NotImplementedError("It is not possible to remotely decode a tensorvar for the moment")


class _FixedPrecisionTensor(_SyftTensor):
    """
    The FixedPrecision enables to manipulate floats over an interface which supports only integers,
    Such as _SPDZTensor.
    This is done by specifying a precision p and given a float x, multiply it with 10**p before
    rounding to an integer (hence you keep p decimals)
    """

    def __init__(self,
                 child=None,
                 owner=None,
                 torch_type=None,
                 field=2**31 - 1,
                 base=10,
                 precision_fractional=3,
                 precision_integral=1,
                 already_encoded=False,
                 ):

        if torch_type is None:
            if not already_encoded:
                torch_type = "syft." + type(child).__name__
            else:
                torch_type = 'syft.FloatTensor' # FIXME or sy.Variable

        super().__init__(child=child, owner=owner, torch_type=torch_type)

        self.field = field

        if spdz.field != self.field:
            logging.warning("spdz.field != self.field, be careful you may experience issues with "
                            "multiplication on fix precision shared tensors.")
        self.base = base
        self.precision_fractional = precision_fractional
        self.precision_integral = precision_integral
        self.precision = self.precision_fractional + self.precision_integral
        self.torch_max_value = torch.LongTensor([round(self.field / 2)])

        if already_encoded:
            self.child = child
        else:
            torch_utils.assert_has_only_torch_tensorvars(child)
            chain_tail = None
            if not isinstance(child.child, sy._LocalTensor):
                chain_tail = child.child

            if torch_utils.is_variable(child):
                var_data = child.data
                if len(var_data.size()) > 0:
                    self.encode(var_data)  # this puts in .child an encoded Tensor
                    self.child = sy.Variable(self.child)
                else:
                    self.child = sy.Variable(sy.LongTensor())
                self.child.child = chain_tail
            else:
                if len(child.size()) > 0:
                    self.encode(child)
                else:
                    self.child = sy.LongTensor()
                self.child.child = chain_tail

    def ser(self, private, as_dict=True):

        data = {
            'owner': self.owner.id,
            'id': self.id,
            'child': self.child.ser(private=private, as_dict=True),
            'torch_type': self.torch_type,
            'field': self.field,
            'base': self.base,
            'precision_fractional': self.precision_fractional,
        }
        if as_dict:
            return {'___FixedPrecisionTensor__': data}
        else:
            return msgpack.packb({'___FixedPrecisionTensor__': data}, use_bin_type=True)

    @classmethod
    def deser(cls, msg_obj, worker, acquire):
        """
        General method for de-serializing an SPDZTensor
        """

        if acquire:
            child = encode.decode(msg_obj['child'], worker, acquire, message_is_dict=True)

            obj = _FixedPrecisionTensor(child=child,
                                        owner=worker,
                                        torch_type=msg_obj['torch_type'],
                                        field=msg_obj['field'],
                                        base=msg_obj['base'],
                                        precision_fractional=msg_obj['precision_fractional'],
                                        already_encoded=True)
            return obj
        else:
            return _SyftTensor.deser(msg_obj, worker, acquire)

    def on(self, shares):
        return self.wrap(True)

    def encode(self, rational):
        owner = rational.owner
        upscaled = (rational * self.base ** self.precision_fractional).long()
        field_element = upscaled % self.field

        # Handle neg values
        gate = field_element.gt(self.torch_max_value).long()
        neg_nums = (field_element - self.field) * gate
        pos_nums = field_element * (1 - gate)
        field_element = (neg_nums + pos_nums)

        torch_utils.enforce_owner(field_element, owner)
        self.child = field_element
        return self

    def decode(self):
        save = self.child.child*1
        self.child.child = None # <-- This is doing magic things
        value = self.child % self.field
        if len(value.size()) == 0:
            # raise TypeError("Can't decode empty tensor")
            return None
        gate = value.native_gt(self.torch_max_value).long()
        neg_nums = (value - self.field) * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums).float() / (self.base ** self.precision_fractional)
        self.child.child = save.child
        return result

    @classmethod
    def handle_call(cls, command, owner):
        """
        This is a special handle_call method which is compatible with
        .child objects that are themselves torch objects (wrappers) of
        other methods.
        :param command:
        :param owner:
        :return:
        """

        attr = command['command']
        args = command['args']
        kwargs = command['kwargs']
        has_self = command['has_self']
        if has_self:
            self = command['self']

            # A) override the "share" command (which would normally call .share on the .child object
            if attr == 'share':
                response = self.share(*args, **kwargs)
                return response

            # B) override commands which take no other tensors as arguments
            elif attr == 'prod':
                response = cls.prod(self, *args, **kwargs)
                return _FixedPrecisionTensor(response).wrap(True)
            elif attr == 'sum':
                response = cls.sum(self, *args, **kwargs)
                return _FixedPrecisionTensor(response).wrap(True)
            elif attr == 'cumsum':
                response = cls.cumsum(self, *args, **kwargs)
                return _FixedPrecisionTensor(response).wrap(True)

            # C) override functions which have tensors as arguments
            elif attr in ('__add__', '__mul__', '__sub__', '__div__', '__truediv__', 'mm',
                        'matmul') and\
                    isinstance(args[0], sy._FixedPrecisionTensor):

                # Compute the precision to keep
                other = args[0]
                assert (self.base == other.base) and (self.field == other.field), \
                    'Arguments should share the same base and field'

                self_precision = self.precision_fractional
                other_precision = other.precision_fractional

                # If the precision fractional of self is different than other's raise an exception
                # You may uncomment this line out if you do care about different precisions,
                # the code will work either way.
                # if not(self_precision == other_precision):
                #     raise ArithmeticError("The tensors have different precisions")

                # Perform the computation
                torch_tensorvar = None
                if attr == '__mul__':
                    torch_tensorvar = cls.__mul__(self, other)
                elif attr in ('mm',) or attr == 'matmul':
                    torch_tensorvar = cls.mm(self, other)
                elif attr == '__add__':
                    torch_tensorvar = cls.__add__(self, *args, **kwargs)
                elif attr == '__sub__':
                    torch_tensorvar = cls.__sub__(self, *args, **kwargs)
                elif attr == '__div__' or '__truediv__':
                    torch_tensorvar = cls.__div__(self, *args, **kwargs)
                if attr not in ('mm','__mul__') :
                    response = torch_tensorvar.fix_precision(
                        already_encoded=True,
                        precision_fractional=max(self_precision, other_precision)
                    )
                    return response

            else:  # Standard procedure for methods
                # Get the next node type and update in command tensorvar with tensorvar.child
                next_command, child_type = torch_utils.prepare_child_command(
                    command, replace_tensorvar_with_child=True)

                is_var = torch_utils.is_variable(child_type.__name__)

                if is_var:
                    child_type = torch.guard[self.data.torch_type]

                # Forward the call to the next child
                torch_tensorvar = child_type.handle_call(next_command, owner)


            # Compute the precision to keep
            precision = self.precision_fractional
            if attr in ('mm', 'matmul', '__mul__'):
                other = args[0]
                torch_tensorvar, precision = self.truncate(torch_tensorvar, args[0])

            response = torch_tensorvar.fix_precision(
                already_encoded=True,
                precision_fractional=precision
            )
            return response

    def truncate(self, torch_tensorvar, other):

        def egcd(a, b):
            if a == 0:
                return (b, 0, 1)
            else:
                g, y, x = egcd(b % a, a)
                return (g, x - (b // a) * y, y)

        def modinv(a, m):
            g, x, y = egcd(a, m)
            if g != 1:
                raise Exception('modular inverse does not exist')
            else:
                return x % m

        if isinstance(other, sy._FixedPrecisionTensor):

            result_precision_fractional = max(self.precision_fractional, other.precision_fractional)
            result_precision_integral = self.precision_integral
            result_precision = result_precision_fractional + result_precision_integral

            if result_precision_fractional > 0:
                tail_node = torch_utils.find_tail_of_chain(torch_tensorvar)
                if isinstance(tail_node, sy._GeneralizedPointerTensor):

                    if(isinstance(torch_tensorvar, sy.Variable)):
                        a = torch_tensorvar.data
                    else:
                        a = torch_tensorvar

                    workers = list(tail_node.pointer_tensor_dict.keys())

                    b_ = int((self.base ** (2 * result_precision + 1)))

                    b = a + b_

                    rand_shape = torch.IntTensor(list(b.get_shape())).prod()

                    mask = torch.LongTensor(1).send(workers[0]).expand(rand_shape).contiguous().view(list(b.get_shape()))
                    mask.random_(self.base ** result_precision)

                    mask_low = torch.fmod(mask, self.base ** result_precision_fractional)
                    mpc_mask = mask.share(*workers).get()

                    b_masked = (b + mpc_mask).get()
                    b_masked_low = torch.fmod(b_masked, self.base ** result_precision_fractional)
                    b_low = b_masked_low.share(*workers) - mask_low.share(*workers).get()

                    # TODO: calculating the inverse every time is stupid slow - but i need to keep moving
                    c = (a - b_low) * modinv(self.base**result_precision_fractional, self.field)

                    if (isinstance(torch_tensorvar, sy.Variable)):
                        torch_tensorvar = (torch_tensorvar * 0)
                        torch_tensorvar.data.child.child += c.child.child
                    else:
                        torch_tensorvar = c

                else:
                    torch_tensorvar = torch_tensorvar / self.base ** result_precision_fractional

            return torch_tensorvar, result_precision_fractional

        return torch_tensorvar, self.precision_fractional


    def get(self, *args, **kwargs):
        """
        /!\ Return a tensorvar
        """
        if torch_utils.is_variable(self.child):
            var = self.parent
            if self.child.grad is None:
                self.child.init_grad_()
            child_child_var = self.child.get(*args, **kwargs)
            torch_utils.bind_var_like_objects(self, child_child_var)

            if hasattr(var, 'grad') and var.grad is not None:
                self.child.assign_grad_(child_child_var.grad)
                self.grad = var.grad.child
                self.grad.child = self.child.grad
                self.grad.data = var.grad.data.child
                self.grad.data.child = self.child.grad.data
            return var
        else:
            self.child = self.child.get(*args, **kwargs)
            return self.parent

    def check_and_scale_precision_if_needed(self, other):
        # checks the terms of self and other to make sure they have the same
        # precision - if not it returns two new params with the same precision


        # if other is not a fixed tensor, convert it to a fixed one
        if (not hasattr(other, 'precision_fractional')):
            other = other.fix_precision(precision_fractional=self.precision_fractional)

        if (self.precision_fractional == other.precision_fractional):
            return self.child, other.child

        elif (self.precision_fractional > other.precision_fractional):
            scaling = self.base ** (self.precision_fractional - other.precision_fractional)
            return  self.child, other.child * scaling

        else: #(self.precision_fractional < other.precision_fractional):
            scaling = self.base ** (other.precision_fractional - self.precision_fractional)
            return self.child * scaling, other.child

    def __add__(self, other):
        a, b = self.check_and_scale_precision_if_needed(other)
        return (a + b) % self.field

    def __sub__(self, other):
        a, b = self.check_and_scale_precision_if_needed(other)
        return (a - b) % self.field

    def __mul__(self, other):
        a, b = self.check_and_scale_precision_if_needed(other)
        return (a * b)# % self.field # - modulus performed later

    def __div__(self, other):
        # if other is not a fixed tensor, convert it to a fixed one
        if (not hasattr(other, 'precision_fractional')):
            other = other.fix_precision(precision_fractional = self.precision_fractional)

        if (self.precision_fractional == other.precision_fractional):
            gp_response = (self.child * 10 ** self.precision_fractional / other.child) % \
                          self.field
        elif (self.precision_fractional > other.precision_fractional):
            gp_response = (self.child / other.child * 10 ** other.precision_fractional) % \
                          self.field

        elif (self.precision_fractional < other.precision_fractional):
            gp_response = ((self.child *10 ** (2 * other.precision_fractional
                           - self.precision_fractional)) / other.child) % \
                          self.field
        return gp_response

    # def __mul__(self, other):
    #     # if other is not a fixed tensor, convert it to a fixed one
    #     if (not hasattr(other, 'precision_fractional')):
    #         other = other.fix_precision(precision_fractional = self.precision_fractional)
    #     return self.child * other.child

    def prod(self, *args, **kwargs):
        # getting the dimension of the tensor which prod will be applied to. (needed for fixing
        # the precision precision problems)
        dim = self.child.size()[args[0]]
        return self.child.prod(*args, **kwargs) / 10 ** (self.precision_fractional * dim)

    def sum(self, *args, **kwargs):
        return (self.child.sum(*args, *kwargs) / 10 ** self.precision_fractional)

    def cumsum(self, *args, **kwargs):
        return (self.child.cumsum(*args, *kwargs) / 10 ** self.precision_fractional)

    def mm(self, other):
        response = self.child.mm(other.child)
        return response

    def __repr__(self):
        if(not isinstance(self.child, _SNNTensor)):
            return "[Fixed precision]\n"+self.decode().__repr__()
        else:
            return "[Fixed precision]\n" + self.child.__repr__()

    def __str__(self):
        if (not isinstance(self.child, _SNNTensor)):
            return "[Fixed precision]\n" + self.decode().__repr__()
        else:
            return "[Fixed precision]\n" + self.child.__repr__()


class _SPDZTensor(_SyftTensor):
    """
    This tensor wraps a GeneralizedPointerTensor containing shares and knows how to
    manipulate those shares properly so that the resulting methods are themselves
    also SPDZTensors.

    This tensor is a special case tensor in multiple ways. First and foremost,
    it is the first tensor we have implemented whose .child object is a Torch
    object (a torch wrapper which is the head of another chain). This was necessary
    to allow for multiple operations to occur within each single operation within
    __add__ and __mul__.
    """

    def __init__(self,
                 shares=None,
                 child=None,
                 torch_type=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Fixme: remove the share on init, declaring a SPDZTensor should autmatically create a _GeneralizedPointerTensor

        if shares is not None:
            if isinstance(shares, sy._GeneralizedPointerTensor):
                raise TypeError('Should have a wrapper on the _GeneralizedPointerTensor')
            torch_type = shares.child.torch_type
            self.shares = shares  # shares is a Tensorvar > _GeneralizedPointerTensor
            self.child = self.shares

        elif child is not None:
            if isinstance(child, sy._GeneralizedPointerTensor):
                raise TypeError('Should have a wrapper on the _GeneralizedPointerTensor')
            torch_type = child.child.torch_type
            self.child = child
            self.shares = self.child
        else:
            raise TypeError("cannot initialize SPDZTensor with shares and child both == None")

        self.torch_type = torch_type

        # self.allow_arbitrary_arg_types_for_methods = set()
        # self.allow_arbitrary_arg_types_for_methods.add("__mul__")

    def get_shape(self):
        # skip .child since it's a wrapper
        return self.child.child.get_shape()

    def ser(self, private, as_dict=True):

        data = {
            'owner': self.owner.id,
            'id': self.id,
            'shares': self.child.ser(private=private, as_dict=True),
            'torch_type': self.torch_type
        }
        str_type = "__" + type(self).__name__ + "__"
        if as_dict:
            return {str_type: data}
        else:
            return msgpack.packb({str_type: data}, use_bin_type=True)

    @classmethod
    def deser(cls, msg_obj, worker, acquire):
        """
        General method for de-serializing an SPDZTensor
        """

        if acquire:
            child_shares = list(msg_obj['shares'].values())[0]['child']
            if '___GeneralizedPointerTensor__' in child_shares.keys():

                gpt_dct = child_shares['___GeneralizedPointerTensor__']
                shares = _GeneralizedPointerTensor.deser(gpt_dct, worker, acquire).wrap(True)

                shares.child.child = shares

                result = cls(shares=shares,
                                    id=msg_obj['id'],
                                    owner=worker,
                                    torch_type=msg_obj['torch_type'])
            elif '___LocalTensor__' in child_shares.keys():
                # shares = sy._TorchTensor.deser(msg_obj['shares'], worker, acquire)
                # result = sy._SPDZTensor(shares=shares,
                #                        id=msg_obj['id'],
                #                        owner=worker,
                #                        torch_type=msg_obj['torch_type'])
                result = cls(shares=sy.LongTensor())
            else:
                raise TypeError("Unrecognized type ", list(child_shares.keys()))

            return result
        else:
            return _SyftTensor.deser(msg_obj, worker, acquire)

    # The table of command you want to replace
    substitution_table = {
        'torch.add': 'torch.add',
        'torch.mul': 'torch.mul',
    }

    class overload_functions:
        """
        Put here the functions you want to overload
        Beware of recursion errors.
        """

        @staticmethod
        def get(attr):
            attr = attr.split('.')[-1]
            return getattr(sy._SPDZTensor.overload_functions, attr)

    def second_constructor(self):
        return self.wrap(True)

    # Put here all the methods you want to overload

    def on(self, wrapper):
        """
        Used to add a new _SPDZTensor at the top of the chain, just before the tensorvar wrapper
        """
        # Assign the newly created tensor to the good owner and torch_type
        self.torch_type = wrapper.child.torch_type
        self.owner = wrapper.child.owner

        torch_utils.wrap_command_with(self, wrapper=wrapper)

        # In case wrapper is a variable, do the same with data and grad (if necessary)
        if torch_utils.is_variable(wrapper):
            wrapper.data = _SPDZTensor(self.child.data).on(wrapper.data)
            if torch_utils.is_variable(wrapper.grad):
                wrapper.assign_grad_(_SPDZTensor(self.child.grad).on(wrapper.grad))

        return wrapper

    def share_scalar(self, scalar):
        other = torch.zeros(list(self.get_shape())).long() + scalar

        # if the parent is a Variable type then we need to cast this to
        # a Varible of longs instead (which is silly and redundant but
        # i need this to work by Monday so i'm hacking this here... realistically
        # SPDZTensor should NEVER point to variable objects TODO:fix
        if(self.torch_type == 'syft.Variable'):
            other = sy.Variable(other)

        other = other.share(*list(self.shares.child.pointer_tensor_dict.keys())).child

        return other

    def __add__(self, other):

        if (isinstance(other, (int, float, bool))):
            other = self.share_scalar(other)

        # gp_ stands for GeneralizedPointer
        gp_response = spdz.spdz_add(self.shares, other.shares)
        return gp_response

    def __sub__(self, other):

        if (isinstance(other, (int, float, bool))):
            other = self.share_scalar(other)

        gp_response = spdz.spdz_add(self.shares, spdz.spdz_neg(other.shares))
        return gp_response

    def __neg__(self):
        gp_response = spdz.spdz_neg(self.shares)
        return gp_response

    def sum(self, *args, **kwargs):
        gp_response = self.child.sum(*args, **kwargs) % spdz.field
        return gp_response

    def cumsum(self, *args, **kwargs):
        gp_response = self.child.cumsum(*args, **kwargs) % spdz.field
        return gp_response

    def __mul__(self, other):

        if(isinstance(other, type(self))):
            workers = list(self.shares.child.pointer_tensor_dict.keys())
            if torch_utils.is_variable(self.torch_type):
                gp_response = self * 1
                gp_response.data = spdz.spdz_mul(self.data.shares, other.data.shares, workers)
                #TODO: and the grad ?
            else:
                gp_response = spdz.spdz_mul(self.shares, other.shares, workers)
        else:
            gp_response = self.shares * other
        return gp_response

    def mm(self, other):
        workers = list(self.shares.child.pointer_tensor_dict.keys())
        if torch_utils.is_variable(self.torch_type):
            gp_response = self * 1
            gp_response.data = spdz.spdz_matmul(self.data.shares, other.data.shares, workers)
            # TODO: and the grad ?
        else:
            gp_response = spdz.spdz_matmul(self.shares, other.shares, workers)

        return gp_response

    def __matmul__(self, other):
        return self.mm(other)

    def sigmoid(self):
        workers = list(self.shares.child.pointer_tensor_dict.keys())
        W0, W1, W3, W5 = spdz.generate_sigmoid_shares_communication(self.shape, workers)
        x2 = x * x
        x3 = x * x2
        x5 = x3 * x2
        temp5 = x5 * W5
        temp3 = x3 * W3
        temp1 = x * W1
        temp53 = temp5 + temp3
        temp531 = temp53+ temp1
        return W0 + temp531

    @classmethod
    def handle_call(cls, command, owner):
        """
        This is a special handle_call method which is compatible with
        .child objects that are themselves torch objects (wrappers) of
        other methods.
        :param command:
        :param owner:
        :return:
        """
        attr = command['command']
        args = command['args']
        kwargs = command['kwargs']
        self = command['self']

        if attr == '__mul__':
            gp_response = cls.__mul__(self, *args, **kwargs)
        elif attr == '__add__':
            gp_response = cls.__add__(self, *args, **kwargs)
        elif attr == '__sub__':
            gp_response = cls.__sub__(self, *args, **kwargs)
        elif attr == 'sum':
            gp_response = cls.sum(self, *args, **kwargs)
        elif attr == 'cumsum':
            gp_response = cls.sum(self, *args, **kwargs)
        elif attr == 'mm':
            gp_response = cls.mm(self, *args, **kwargs)
        else:
            gp_response = getattr(self.child, attr)(*args, **kwargs)

        if torch_utils.is_variable(gp_response.child.torch_type):
            var_data_type = gp_response.child.data.torch_type
            variable = sy.Variable(torch.guard[var_data_type]())
            variable.init_grad_()
            mpc_node = type(self)(gp_response)
            mpc_node.data = type(self)(gp_response.data)
            mpc_node.grad = type(self)(gp_response.grad)
            mpc_node.grad.data = type(self)(gp_response.grad.data)
            mpc_node.grad.data.child.child = None # FIXME: is it necessary?
            torch_utils.bind_var_like_objects(variable, mpc_node, grad=True)
            return variable
        else:
            response = type(self)(gp_response).wrap(True)
            return response

    def send(self, *workers):
        assert len(workers) > 0, "Please provide workers to receive the data"
        self.n_workers = len(workers)
        self.shares = self.share(self.var, self.n_workers)
        self.child = self.shares
        self.workers = workers
        for share, worker in zip(self.shares, self.workers):
            share.send(worker)

    def get(self, deregister_ptr=False):
        if torch_utils.is_variable(self.child):
            var = sy.Variable(self.data.get())
            var.child = None
            if hasattr(self, 'grad') and self.grad is not None:
                var_grad = self.grad.shares.child.sum_get()
                value = var_grad.data % spdz.field
                # TODO: Add this thing for negative values
                # gate = (value > spdz.torch_max_value).long()
                # neg_nums = (value - spdz.torch_field) * gate
                # pos_nums = value * (1 - gate)
                # result = neg_nums + pos_nums
                var_grad.data = value
                var.init_grad_()
                var.assign_grad_(var_grad)
            return var
        # TODO: have deregister_ptr do something
        value = self.shares.child.sum_get() % spdz.field

        gate = (value > spdz.torch_max_value).long()

        neg_nums = (value - spdz.torch_field) * gate
        pos_nums = value * (1 - gate)
        result = neg_nums + pos_nums
        return result


class _SNNTensor(_SPDZTensor, _SyftTensor):

    """
    This tensor extends the _SPDZTensor class with additional functionality for
    an encrypted comparison operator, which can compare shared values with either
    other shared values or with plaintext values. This functionality is also core
    to higher level functions such as argmax, softmax, ReLU non-linearities as well
    as clipping the unstable tails of polynomial approximations of non-linearities
    # such as Sigmoid.
    """

    class overload_functions:
        """
        Put here the functions you want to overload
        Beware of recursion errors.
        """

        @staticmethod
        def get(attr):
            attr = attr.split('.')[-1]
            return getattr(sy._SNNTensor.overload_functions, attr)

    def second_constructor(self):
        return self.wrap(True)

    # Put here all the methods you want to overload

    def on(self, wrapper):
        """
        Used to add a new _SPDZTensor at the top of the chain, just before the tensorvar wrapper
        """
        # Assign the newly created tensor to the good owner and torch_type
        self.torch_type = wrapper.child.torch_type
        self.owner = wrapper.child.owner

        torch_utils.wrap_command_with(self, wrapper=wrapper)

        # In case wrapper is a variable, do the same with data and grad (if necessary)
        if torch_utils.is_variable(wrapper):
            wrapper.data = _SNNTensor(self.child.data).on(wrapper.data)
            if torch_utils.is_variable(wrapper.grad):
                wrapper.assign_grad_(_SNNTensor(self.child.grad).on(wrapper.grad))

        return wrapper

    def relu(self):
        return relu(self.parent)

    def positive(self):
        return relu_deriv(self.parent)

    def __gt__(self, other):
        return (self.parent - other.parent - 1).positive()

    def __ge__(self, other):
        return (self.parent - other.parent).positive()

    def __lt__(self, other):
        return (other.parent - self.parent - 1).positive()

    def __le__(self, other):
        return (other.parent - self.parent).positive()

    def __eq__(self, other):
        return (self.parent >= other.parent) * (self.parent <= other.parent)

class _TorchObject(object):
    """
    This tensor is simply a more convenient way to add custom
    functions to all Torch tensor types, including Torch Variable.
    Note that it is the parent class of the two following classes:
    _TorchTensor and a_TorchVariable
    """

    __module__ = 'syft'

    def __gt__(self, *args, **kwargs):
        try:
            return self.child > args[0].child
        except:
            return self.native___gt__(*args, **kwargs)

    def __lt__(self, *args, **kwargs):
        try:
            return self.child < args[0].child
        except:
            return self.native___lt__(*args, **kwargs)

    def __le__(self, *args, **kwargs):
        try:
            return self.child <= args[0].child
        except:
            return self.native___le__(*args, **kwargs)

    def __ge__(self, *args, **kwargs):
        try:
            return self.child >= args[0].child
        except:
            return self.native___ge__(*args, **kwargs)

    def __eq__(self, *args, **kwargs):
        if(isinstance(self.child, _LocalTensor)):
            return self.native___eq__(*args, **kwargs)
        else:
            try:
                return self.child == args[0].child
            except:
                return self.native___eq__(*args, **kwargs)

    def get_shape(self):
        return self.child.get_shape()

    def relu(self, *args, **kwargs):
        return self.child.relu(*args, **kwargs)

    def positive(self, *args, **kwargs):
        return self.child.positive(*args, **kwargs)

    def native_get_shape(self):
        return self.get_shape()

    def share(self, *workers):
        """
        Create additive shares of a tensorvar and send them to workers
        """
        if isinstance(self.child, _PointerTensor):
            response = self.child.share(*workers)
            if torch_utils.is_variable(self):
                self_copy = self
                self_copy.child = response
                self_copy.data.child = response.data
                self_copy.grad.child = response.grad
                self_copy.grad.data.child = response.grad.data
                return self_copy
            else:
                return response.wrap(True)

        elif isinstance(self.child, _FixedPrecisionTensor):
            var_shared = self.child.child.share(*workers)
            self.child.child = var_shared
            if torch_utils.is_variable(self):
                self.data.child.child = var_shared.data
                if hasattr(self, 'grad') and self.grad is not None:
                    self.grad.child.child = var_shared.grad
                    self.grad.data.child.child = var_shared.grad.data
            return self

        else:
            is_variable = torch_utils.is_variable(self)
            if is_variable:
                if not hasattr(self, 'grad') or self.grad is None:
                    self.init_grad_()
            n_workers = len(workers)
            x_enc = self._encode()
            shares = self._share(n_workers)

            pointer_shares_dict = {}
            for share, worker in zip(shares, workers):
                share.send(worker)
                pointer_shares_dict[worker] = share.child

            self_copy = self*1
            if is_variable:
                self_copy.init_grad_()
            x_gp = _GeneralizedPointerTensor(pointer_shares_dict, torch_type='syft.LongTensor').on(self_copy)
            if is_variable:
                torch_utils.link_var_chain_to_data_and_grad_chains(x_gp, x_gp.data, x_gp.grad)
            x_mpc = _SNNTensor(x_gp, torch_type='syft.LongTensor').on(self)
            if is_variable:
                torch_utils.link_var_chain_to_data_and_grad_chains(x_mpc, x_mpc.data, x_mpc.grad)
            return x_mpc

    def native_share(self, *workers):
        out = self.share(*workers)
        return out

    def _share(self, n_workers):
        if torch_utils.is_variable(self):
            data_shares = self.data._share(n_workers)
            shares = []
            for data_share in data_shares:
                shares.append(sy.Variable(data_share))
            return shares
        else:
            if not isinstance(self, torch.LongTensor):
                raise TypeError(
                    "Can only MPCShare LongTensor type. You tried to share " + str(type(self).__name__) + "." +
                    " Do you need to call .fix_precision() first?")
            return spdz.share(self, n_workers)

    def _encode(self):
        return spdz.encode(self)

    def fix_precision(self,
                      field=2**31-1,
                      base=10,
                      precision_fractional=3,
                      already_encoded=False):

        if torch_utils.is_variable(self):
            if not hasattr(self, 'grad') or self.grad is None:
                self.init_grad_()

        if isinstance(self.child, _PointerTensor):
            return self.owner._execute_call('fix_precision', self)
        else:
            fpt = lambda tensorvar, is_encoded: _FixedPrecisionTensor(tensorvar,
                                                                      torch_type=tensorvar.child.torch_type,
                                                                      field=field,
                                                                      base=base,
                                                                      precision_fractional=precision_fractional,
                                                                      already_encoded=is_encoded).wrap(True)

            if torch_utils.is_variable(self):
                _var = fpt(self, already_encoded)
                # This 2nc fpt() is just a linking:
                # Var ------> FixP -------> Var
                #  \                         \
                # data -----> FixP - - - -> data
                #                   (link)
                _var.data.child = fpt(_var.child.child.data, True).child
                _var.data.child.torch_type = self.data.child.torch_type
                # Add the missing .data link in the last figure
                _var.child.data = _var.data.child
                # Do the same with gradient
                if self.grad is not None:
                    _var.init_grad_()
                    _var.grad = fpt(self.grad, already_encoded)
                    _var.grad.data.child = fpt(_var.grad.child.child.data, True).child
                    _var.grad.data.child.torch_type = self.grad.data.child.torch_type
                    _var.grad.child.data = _var.grad.data.child
                    _var.child.grad = _var.grad.child
                return _var
            else:
                return fpt(self, already_encoded)

    def native_fix_precision(self, *args, **kwargs):
        return self.fix_precision(*args, **kwargs)

    def fix_precision_(self, *args, **kwargs):
        tensorvar = self.fix_precision(*args, **kwargs)
        if torch_utils.is_variable(self):  # grad are already created
            torch_utils.bind_var_like_objects(self, tensorvar.child, grad=True)
        else:
            self.child = tensorvar.child

    def native_fix_precision_(self, *args, **kwargs):
        return self.fix_precision_(*args, **kwargs)

    def sum_get(self, *args, **kwargs):
        return self.child.sum_get(*args, **kwargs)

    def set_id(self, new_id):
        self.child.set_id(new_id)
        return self

    def __str__(self):
        return self.native___str__()

    def __repr__(self):

        if torch_utils.is_tensor(self) and hasattr(self, 'child') and not isinstance(self.child, (
                sy._LocalTensor, sy._PointerTensor)):

            if isinstance(self.child, sy._FixedPrecisionTensor):
                return self.child.__repr__()

            x_ = type(self)()
            x_.native_set_(self)
            return "[Head of chain]\n" + x_.native___repr__()

        if torch_utils.is_variable(self) and hasattr(self, 'child') and not isinstance(self.child, (
                sy._LocalTensor, sy._PointerTensor)):
            x_ = type(self)(self.data)
            x_.native_set_(self)
            return "[Head of chain]\n" + x_.native___repr__()

        return self.native___repr__()

    def create_pointer(self, register=False, location=None, ptr_id=None):

        return self.child.create_pointer(parent=self, register=register, location=location,
                                         ptr_id=ptr_id).wrap()

    def move(self, worker, new_id=None):
        """
        Give the end leaf of the chain to worker,
        just like if the last elmt was send its child
        to worker
        self->alice->obj [worker] => self->alice->worker->obj
        """
        raise NotImplementedError('Move is not supported anymore.')


class _TorchTensor(_TorchObject):

    def __str__(self):
        if isinstance(self.child, _PointerTensor):
            return type(self).__name__ + self.child.__str__() + ""
        elif isinstance(self.child, _LocalTensor) and torch_utils.is_tensor_empty(self):
            if hasattr(self.child, 'child'):
                return self.child.child.native___str__()
            else:
                return "Empty Wrapper:\n" + self.native___str__()
        else:
            if not isinstance(self.child, (sy._LocalTensor, sy._PointerTensor)):
                x_ = type(self)()
                x_.native_set_(self)
                return "[Head of chain]\n" + x_.native___repr__()
            return self.native___str__()

    @classmethod
    def handle_call(cls, command, owner):

        attr = command['command']
        args = command['args']
        kwargs = command['kwargs']
        self = command['self']

        return getattr(self, attr)(*args, **kwargs)

    def ser(self, private, as_dict=True):
        key = '__' + type(self).__name__ + '__'
        data = self.tolist() if not private else []
        tensor_msg = {
            'type': str(self.__class__).split("'")[1],
            'torch_type': 'syft.' + type(self).__name__,
            'data': data,
            'child': self.child.ser(private)
        }
        if as_dict:
            return {key: tensor_msg}
        else:
            return msgpack.packb({key: tensor_msg}, use_bin_type=True)

    @staticmethod
    def deser(obj_type, msg_obj, worker, acquire):
        child_type, child_obj = torch_utils.extract_type_and_obj(msg_obj['child'])
        syft_obj = sy._SyftTensor.deser_routing(child_type, child_obj, worker, acquire)

        # If we have retrieved an already existing object (TODO: add checks) then return it
        if syft_obj.parent is not None and syft_obj.child is not None:
            return syft_obj.parent

        tensorvar = torch.guard['syft.' + obj_type](msg_obj['data'])
        torch_utils.wrap_command_with(syft_obj, tensorvar)

        # TODO: Find a smart way to skip register and not leaking the info to the local worker
        # This would imply overload differently the __init__ to provide an owner for the child attr.
        worker.hook.local_worker.de_register(tensorvar)

        # Ensure that the loop is made, if needed
        if isinstance(torch_utils.find_tail_of_chain(tensorvar), sy._LocalTensor):
            torch_utils.fix_chain_ends(tensorvar)

        return tensorvar

    def broadcast(self, workers):
        """
        Send to multiple workers and get back a _GeneralizedPointerTensor
        :return:
        """
        # TODO: Doublon with the new functionality send(*worker)
        # Even if .send is on Var and .broadcast en _GenPtrT
        pointers_dict = {}
        for worker in workers:
            pointers_dict[worker] = self.clone().send(worker).child
        return _GeneralizedPointerTensor(pointers_dict).on(self)

    def send(self, *workers, ptr_id=None):
        """
        Give the root of the chain held by self to worker
        self->alice->obj [worker] => self->worker->alice->obj

        Args:
            worker: the recipient of the transfer
            ptr_id: the id of the object when sent:
                x.send(bob, 1000)
                will result in bob having the tensor x with id 1000
        """
        assert len(workers) > 0, "Please provide workers to receive the data"

        if len(workers) == 1:
            worker = workers[0]
        else:
            gpt_dict = {}
            for worker in workers:
                gpt_dict[worker] = (self*1).send(worker).child
            sy._GeneralizedPointerTensor(gpt_dict).on(self)
            return self

        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        if ptr_id is None:
            ptr_id = random.randint(0, 10e10)

        obj_id = self.child.id

        # creates a pointer to LocalTensor without a Torch object wrapping it because
        # we're going to set self.child to be this pointer.
        # we set register=True because we want it to be registered locally

        self.owner.send_obj(self, ptr_id, worker)

        # clears data which could be cached in the wrapper (which is self)
        # which would be confusing for folks
        self.native_set_()

        # set this wrapper's child to be the newly created PointerTensor
        self.child.id = obj_id
        syft_pointer = self.child.create_pointer(location=worker, id_at_location=ptr_id, register=True)
        torch_utils.wrap_command_with(syft_pointer, self)
        self.parent = None

        return self

    def get(self, deregister_ptr=True, update_ptr_wrapper=True):
        """
        Get a remote tensor back to the local worker.
        :param deregister_ptr: should we de-register from the remote. Default to True
        :param update_ptr_wrapper: If true, by default, change the pointer variable (wrapper)
        to instead wrap the SyftTensor object that was returned so that any variable that may
        still exist referencing this pointer will simply call local data instead of sending
        messages elsewhere, or a closer pointer
        :return: self
        """

        # returns a Tensor object wrapping a SyftTensor
        tensor = self.child.get(deregister_ptr=deregister_ptr)

        # GeneralizedPointerTensor returns a list
        if(isinstance(tensor, list)):
            return tensor

        # if this is the case, then child is probably
        # a wrapper which contains other torch objects
        # such as FixedPrecisionTensor or SPDZTensor
        # so all we really need to do is make sure self.child
        # is correct and then return self.
        if(torch_utils.is_syft_tensor(tensor)):
            self.child = tensor
            return self

        torch_utils.assert_has_only_torch_tensorvars(tensor)
        # this will change the pointer variable (wrapper) to instead wrap the
        # SyftTensor object that was returned so that any variable that may
        # still exist referencing this pointer will simply call local data instead
        # of sending messages elsewhere, or a closer pointer
        if update_ptr_wrapper:
            syft_tensor = tensor.child
            self.child = syft_tensor
            # In case we have a final get() (ie returning a FloatTensor), we have e.g.
            # x = Float(...)
            # x.send(...)
            # x2 = x.get()
            # We  have x2: [no dim]->[_Local]->[Float()]
            # Whereas we expect x2: [Float()]
            # So we use the .set_() method, to change the storage of [no dim]
            if not isinstance(syft_tensor, sy._PointerTensor) \
                    and tensor is not None \
                    and tensor.dim() > 0:
                self.native_set_(tensor)
            torch_utils.fix_chain_ends(self)
            torch_utils.assert_is_chain_well_formed(self)

        return self

    # in the case of fixed precision tensors, torch tensors need this function
    def decode(self):
        return self.child.decode()

    def decode_(self):
        self.child = self.child.decode().child


class _TorchVariable(_TorchObject):

    def send(self, *workers, new_id=None, new_data_id=None, new_grad_id=None, new_grad_data_id=None):
        """
        Give the root of the chain held by self to worker
        self->alice->obj [worker] => self->worker->alice->obj
        Because there are Variable involved, there are actually 4 chains involved,
        the variable chain, variable.data, variable.grad, variable.grad.data
        """
        assert len(workers) > 0, "Please provide workers to receive the data"

        if len(workers) == 1:
            worker = workers[0]
        else:
            gpt_dict = {}
            if not hasattr(self, 'grad') or self.grad is None:
                self.init_grad_()
            for worker in workers:
                gpt_dict[worker] = (self*1).send(worker).child
            sy._GeneralizedPointerTensor(gpt_dict).on(self)
            torch_utils.link_var_chain_to_data_and_grad_chains(self, self.data, self.grad)
            return self

        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        # Init new remote ids if needed
        (new_id, new_data_id, new_grad_id, new_grad_data_id) = utils.map_tuple(None,
                                           (new_id, new_data_id, new_grad_id,new_grad_data_id),
                                           lambda id: id if id is not None else random.randint(0, 10e10))

        # Store tensorvar ids
        obj_id = self.child.id
        obj_data_id = self.data.child.id
        obj_grad_id = self.grad.child.id if self.grad is not None else None
        obj_grad_data_id = self.grad.data.child.id if self.grad is not None else None

        self.owner.send_obj(self,
                            new_id,
                            worker,
                            new_data_id=new_data_id,
                            new_grad_id=new_grad_id,
                            new_grad_data_id=new_grad_data_id)

        # Clear data which could be cached in the wrapper (which is self)
        utils.map_tuple(None, (self, self.data, self.grad, self.grad.data), lambda x: x.native_set_())

        # For all the objects, create a pointer and insert it as a direct child
        for id, remote_id, wrapper in zip(
                [obj_id, obj_data_id, obj_grad_id, obj_grad_data_id],
                [new_id, new_data_id, new_grad_id, new_grad_data_id],
                [self, self.data, self.grad, self.grad.data]):
            wrapper.child.id = id
            pointer = wrapper.child.create_pointer(location=worker, id_at_location=remote_id, register=True)
            torch_utils.wrap_command_with(pointer, wrapper)
            wrapper.parent = None

        torch_utils.link_var_chain_to_data_and_grad_chains(self, self.data, self.grad)

        return self

    def get(self, deregister_ptr=True, update_ptr_wrapper=True):
        """
        Get a remote variable back to the local worker.
        :param deregister_ptr: should we de-register from the remote. Default to True
        :param update_ptr_wrapper: If true, by default, change the pointer variable (wrapper)
        to instead wrap the SyftTensor object that was returned so that any variable that may
        still exist referencing this pointer will simply call local data instead of sending
        messages elsewhere, or a closer pointer
        :return: self
        """
        if isinstance(self.child, sy._GeneralizedPointerTensor) and update_ptr_wrapper:
            raise TypeError("Can't update the wrapper of a _GeneralizedPointerTensor. Set update_ptr_wrapper=False.")

        # returns a Variable object wrapping a SyftTensor
        variable = self.child.get(deregister_ptr=deregister_ptr)
        torch_utils.assert_has_only_torch_tensorvars(variable)
        # this will change the wrapper variable to instead wrap the
        # SyftTensor object that was returned so that any variable that may
        # still exist referencing this pointer will simply call local data instead
        # of sending messages elsewhere, or a closer pointer
        if update_ptr_wrapper:
            self.child = variable.child
            self.data.child = variable.data.child
            if self.grad is not None and variable.grad is not None:
                self.grad.child = variable.grad.child
                self.grad.data.child = variable.grad.data.child

            # In case we have a final get() (ie returning a FloatTensor), we have e.g.
            # x = Float(...)
            # x.send(...)
            # x2 = x.get()
            # We  have x2: [no dim]->[_Local]->[Float()]
            # Whereas we expect x2: [Float()]
            # So we use the .set_() method, to change the storage of [no dim]
            if not isinstance(variable.child, sy._PointerTensor) \
                    and variable.data is not None \
                    and variable.data.dim() > 0:
                self.native_set_(variable)
                if self.grad is not None and variable.grad is not None:
                    self.grad.data = variable.grad.data

            torch_utils.fix_chain_ends(self)
            if self.grad is not None:
                torch_utils.link_var_chain_to_data_and_grad_chains(self, self.data, self.grad)
            else:
                torch_utils.link_var_chain_to_data_chain(self, self.data)

            torch_utils.fix_chain_ends(self)
            torch_utils.assert_is_chain_well_formed(self)
            return self

        return variable

    def ser(self, private, as_dict=True, is_head=False):
        key = '__' + type(self).__name__ + '__'

        tensor_msg = {
            'type': str(self.__class__).split("'")[1],
            'torch_type': 'syft.' + type(self).__name__,
            'data': self.data.ser(private) if is_head else [],
            'child': self.child.ser(private),
            'requires_grad': self.requires_grad
        }
        if is_head:
            if self.grad is not None:
                tensor_msg['grad'] = self.grad.ser(private, as_dict, is_head)
            elif self.data.dim() > 0:
                # Create a .grad just if there is some data in the tensor (to avoid recursion errors)
                self.init_grad_()
                tensor_msg['grad'] = self.grad.ser(private, as_dict, is_head)

        if as_dict:
            return {key: tensor_msg}
        else:
            return msgpack.packb({key: tensor_msg}, use_bin_type=True)

    @staticmethod
    def deser(obj_type, msg_obj, worker, acquire, is_head=False):
        child_type, msg_child= torch_utils.extract_type_and_obj(msg_obj['child'])
        var_syft_obj = sy._SyftTensor.deser_routing(child_type, msg_child, worker, acquire)

        if var_syft_obj.parent is not None and var_syft_obj.child is not None:
            return var_syft_obj.parent

        # Deser the var.data
        try:
            var_data_type, var_data_tensor = torch_utils.extract_type_and_obj(msg_obj['data'])
            if is_head:
                var_data = torch.guard[var_data_type].deser(var_data_type, var_data_tensor, worker, acquire)
            else:
                var_data = torch.guard[var_data_type]()
        except AttributeError:
            var_data = torch.guard['FloatTensor']()
        worker.hook.local_worker.de_register(var_data)

        variable = sy.Variable(var_data, requires_grad=msg_obj['requires_grad'])

        # Deser the var.grad
        if 'grad' in msg_obj:
            var_grad_type, var_grad_tensor = torch_utils.extract_type_and_obj(msg_obj['grad'])
            if is_head:
                var_grad = torch.guard[var_grad_type].deser(var_grad_type, var_grad_tensor, worker, acquire, is_head)
            else:
                var_grad = torch.guard[var_grad_type]()
            worker.hook.local_worker.de_register(var_grad)
            variable.assign_grad_(var_grad)
        else:
            var_grad = None

        # TODO: Find a smart way to skip register and not leaking the info to the local worker
        # This would imply overload differently the __init__ to provide an owner for the child attr.
        worker.hook.local_worker.de_register(variable)
        worker.hook.local_worker.de_register(variable.data)
        if variable.grad is not None:
            worker.hook.local_worker.de_register(variable.grad)
            worker.hook.local_worker.de_register(variable.grad.data)

        variable.child = var_syft_obj
        var_syft_obj.parent = variable

        # Re-assign the data, and propagate deeply
        torch_utils.fix_chain_ends(variable)
        if var_grad is None:
            torch_utils.link_var_chain_to_data_chain(variable, var_data)
        else:
            torch_utils.link_var_chain_to_data_and_grad_chains(variable, var_data, var_grad)

        return variable

    def init_grad_(self):
        """
        Initialise grad as an empty tensor
        """
        if self.grad is None or torch_utils.is_tensor_empty(self.grad):
            self.grad = sy.Variable(sy.zeros(self.size()).type(type(self.data)))
            self.grad.native_set_()
            self.grad.child.owner = self.owner
            self.grad.data.child.owner = self.owner

    def assign_grad_(self, var_grad):
        """
        Assign to self.grad any type of variable
        """
        # save the var_grad.data
        var_grad_data = var_grad.data

        # Transform var_grad into an envelope compatible with .grad assignment
        if self.size() != var_grad.size():
            var_grad.data = sy.zeros(self.data.size())
        if type(var_grad.data) != type(self.data):
            var_grad.data = var_grad.data.type(type(self.data))

        self.grad = var_grad

        # put back original var_grad.data
        self.grad.data = var_grad_data

    # in the case of fixed precision tensors, torch tensors need this function
    def decode(self):
        var_data = self.data.decode()
        if var_data is not None:
            var = sy.Variable(var_data)
            var.child = self.child.child.child
            if hasattr(self, 'grad') and self.grad is not None:
                var.assign_grad_(self.grad.decode())
        else:
            var = sy.Variable()
        torch_utils.fix_chain_ends(var)
        return var

    def decode_(self):
        var_data = self.data.decode()
        if var_data is not None:
            self.data = var_data
            if hasattr(self, 'grad') and self.grad is not None:
                self.grad.decode_()
        else:
            self.data.child = self.data.child.child.child
        self.child = self.child.child.child
        torch_utils.fix_chain_ends(self)
