import json
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
            return json.dumps({'__{}__'.format(self.__class__.__name__): data}) + "\n"

    @classmethod
    def deser_routing(cls, dct, worker, acquire):
        """
        Method analysing the dict given to see which Syft Tensor should deserialized,
        and forwarding the call

        [Is this case note that the dct param is assumed to have a single key, which is
        compatible with our encode/decode process (ex: {'___PointerTensor__': {...} })]
        """
        pat = re.compile('__(.+)__')
        for key, obj in dct.items():  # A trick, we don't really loop
            obj_type = pat.search(key).group(1)
            if torch_utils.is_syft_tensor(obj_type):
                if obj_type == '_LocalTensor':
                    return sy._LocalTensor.deser(obj, worker, acquire)
                elif obj_type == '_PointerTensor':
                    return sy._PointerTensor.deser(obj, worker, acquire)
                else:
                    syft_type = torch.guard['syft.' + obj_type]
                    return syft_type.deser(obj, worker, acquire)

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
                syft_child = cls.deser_routing(msg_obj['child'], worker, acquire)
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
        exclude = ['on', '__init__', 'native___init__', '__repr__', '__str__', 'create_pointer',
                   'ser', 'deser', 'handle_call']
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
        tensor_command, torch_type = torch_utils.prepare_child_command(syft_command,
                                                                       replace_tensorvar_with_child=True)
        torch_utils.assert_has_only_torch_tensorvars(tensor_command)

        attr = tensor_command['command']
        args = tensor_command['args']
        kwargs = tensor_command['kwargs']
        has_self = tensor_command['has_self']

        if has_self:
            self = tensor_command['self']
            attr = torch._command_guard(attr, torch.tensorvar_methods)
            command = getattr(self, "native_" + attr)
        else:
            attr = torch._command_guard(attr, torch.torch_modules)
            elems = attr.split('.')
            elems[-1] = 'native_' + elems[-1]
            native_func_name = '.'.join(elems)
            command = eval(native_func_name)

        response = command(*args, **kwargs)

        # TODO : control registration process
        if response is None:
            return response

        if owner.id != owner.hook.local_worker.id:
            if isinstance(response, (int, float, bool)):
                response = torch_type([response])
            elif isinstance(response, (np.ndarray, )):
                logging.warning("[np.ndarray] Hardcoding FloatTensor")
                response = sy.FloatTensor(response)
        else:
            if isinstance(response, (int, float, bool, np.ndarray)):
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

        elif hasattr(response, 'child') and (isinstance(response.child, (_SPDZTensor, _FixedPrecisionTensor))):
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
            return json.dumps({'___LocalTensor__': data}) + "\n"

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
                 raise TypeError('Should use sy._Pointer without Torch wrapper.')
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
            return json.dumps({'___GeneralizedPointerTensor__': data}) + "\n"

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
        try:

            attr_type = "syft."+type(syft_command['attr'][0]).__name__
        except:
            attr_type = "syft.LongTensor"

        syft_commands = torch_utils.split_to_pointer_commands(syft_command)
        result_dict = {}
        for worker_id in syft_commands.keys():
            syft_command = syft_commands[worker_id]
            result_dict[worker_id] = sy._PointerTensor.handle_call(syft_command, owner)

        #TODO: @trask @theo could you take a look at this if you have better ideas on how to get these parameters
        gpt =  _GeneralizedPointerTensor(result_dict,
                                         parent=None,
                                         torch_type=attr_type,
                                         id=None,
                                         owner=owner,
                                         skip_register=False)
        # Fixme: Add a generic child depending on a torch_type
        gpt.child = torch.guard[gpt.torch_type]([])
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

            # if wrapper.grad is None and wrapper.data.dim() > 0:
            #     # create an empty envelope in wrapper.grad
            #     wrapper.init_grad_()
            #     # Create the init arg:
            #     grad_data_pointer_dict = {
            #         w: p.grad #.data
            #         for w, p in self.pointer_tensor_dict.items()
            #     }
            #     # Build the chain with _PlusIsMinusTensor
            #     wrapper_grad = _GeneralizedPointerTensor(grad_data_pointer_dict).on(wrapper.grad)
            #     # Insert the gradient within its chain
            #     wrapper.grad.native_set_(wrapper_grad)

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

        return self.handle_call(cmd, self.owner)

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

        # Else we send the command
        response = owner.send_torch_command(recipient=location, message=command)

        torch_utils.assert_has_only_torch_tensorvars(response)

        # If the command is an in-place method, we only need to return the same wrapper to the same
        # pointer, instead jof returning the new wrapper created in response
        if has_self and utils.is_in_place_method(attr):
            return syft_command['self']

        # Perform the un-wrap: remove the head on all chains (also .data and .grad if any)
        response, _ = torch_utils.get_child_command(response)
        # response is now a _Pointer, with a .data attr which is a _Pointer, etc.

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
            return json.dumps({'___PointerTensor__': data}) + "\n"

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
            tensorvar.data.child.id = self.parent.data.child.id

        # Register the result
        self.owner.register(syft_tensor)
        if syft_tensor.torch_type == 'syft.Variable':
            self.owner.register(tensorvar.data.child)

        torch_utils.fix_chain_ends(tensorvar)

        return tensorvar

    def get_shape(self):
        cmd = {}
        cmd['command'] = "get_shape"
        cmd['args'] = []
        cmd['kwargs'] = {}
        cmd['has_self'] = True
        cmd['self'] = self

        return sy.Size(self.handle_call(cmd, self.owner).get().int().tolist())


class _FixedPrecisionTensor(_SyftTensor):
    """
    TODO: write this

    """

    def __init__(self,
                 child=None,
                 owner=None,
                 torch_type=None,
                 bits=31,
                 base=10,
                 precision_fractional=6,
                 already_encoded=False):

        if torch_type is None:
            if not already_encoded:
                torch_type = "syft." + type(child).__name__
            else:
                torch_type = 'syft.FloatTensor' # FIXME or sy.Variable

        super().__init__(child=child, owner=owner, torch_type=torch_type)

        self.bits = bits
        self.field = 2 ** bits
        self.base = base
        self.precision_fractional = precision_fractional
        self.torch_max_value = torch.LongTensor([round(self.field / 2)])

        if already_encoded:
            self.child = child
        else:
            self.encode(child)

    def ser(self, private, as_dict=True):

        data = {
            'owner': self.owner.id,
            'id': self.id,
            'child': self.child.ser(private=private, as_dict=True),
            'torch_type': self.torch_type,
            'bits': self.bits,
            'base': self.base,
            'precision_fractional': self.precision_fractional,
        }
        if as_dict:
            return {'___FixedPrecisionTensor__': data}
        else:
            return json.dumps({'___FixedPrecisionTensor__': data}) + "\n"

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
                                        bits=msg_obj['bits'],
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
        torch_utils.enforce_owner(field_element, owner)
        self.child = field_element
        return self

    def decode(self):
        self.child.child = None # <-- This is doing magic things
        value = self.child % self.field
        if len(value.size()) == 0:
            raise TypeError("Can't decode empty tensor")
        gate = value.native_gt(self.torch_max_value).long()
        neg_nums = (value - spdz.torch_field) * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums).float() / (self.base ** self.precision_fractional)
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
            if attr == '__add__':
                torch_tensorvar = cls.__add__(self, *args, **kwargs)
                return torch_tensorvar.fix_precision(already_encoded=True)
            if attr == 'share':
                return self.share(*args, **kwargs)
            else:
                result_child = getattr(self.child, attr)(*args, **kwargs)
                return _FixedPrecisionTensor(result_child).wrap(True)

    def get(self, *args, **kwargs):
        self.child = self.child.get(*args, **kwargs)
        return self

    def __add__(self, other):
        response = (self.child + other.child) % self.field

        return response

    def __repr__(self):
        if(not isinstance(self.child, _SPDZTensor)):
            return "[Fixed precision]\n"+self.decode().__repr__()
        else:
            return "[Fixed precision]\n" + self.child.__repr__()

    def __str__(self):
        if (not isinstance(self.child, _SPDZTensor)):
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
                 torch_type='syft.LongTensor',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Fixme: remove the share on init, declaring a SPDZTensor should autmatically create a _GeneralizedPointerTensor

        if shares is not None:
            if isinstance(shares, sy._GeneralizedPointerTensor):
                raise TypeError('Should have a wrapper on the _GeneralizedPointerTensor')

            self.shares = shares  # shares is a _GeneralizedPointerTensor
            self.child = self.shares

        elif child is not None:
            if isinstance(child, sy._GeneralizedPointerTensor):
                raise TypeError('Should have a wrapper on the _GeneralizedPointerTensor')

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
        if as_dict:
            return {'___SPDZTensor__': data}
        else:
            return json.dumps({'___SPDZTensor__': data}) + "\n"

    @classmethod
    def deser(cls, msg_obj, worker, acquire):
        """
        General method for de-serializing an SPDZTensor
        """

        if acquire:
            gpt_dct = list(msg_obj['shares'].items())[0][1]['child']['___GeneralizedPointerTensor__']
            shares = _GeneralizedPointerTensor.deser(gpt_dct, worker, acquire).wrap(True)

            shares.child.child = shares

            result = _SPDZTensor(shares=shares,
                                id=msg_obj['id'],
                                owner=worker,
                                torch_type=msg_obj['torch_type'])

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

    def on(self, shares):
        return self.wrap(True)

    def __add__(self, other):
        # gp_ stands for GeneralizedPointer
        gp_response = spdz.spdz_add(self.shares, other.shares)
        response = _SPDZTensor(gp_response).wrap(True)
        return response

    def __sub__(self, other):
        gp_response = spdz.spdz_add(self.shares, spdz.spdz_neg(other.shares))
        response = _SPDZTensor(gp_response).wrap(True)
        return response

    def __neg__(self):
        gp_response = spdz.spdz_neg(self.shares)
        response = _SPDZTensor(gp_response).wrap(True)
        return response

    def sum(self, *args, **kwargs):
        result_child = self.child.sum(*args, **kwargs) % spdz.field
        response = _SPDZTensor(result_child).wrap(True)
        return response

    def cumsum(self, *args, **kwargs):

        result_child = self.child.cumsum(*args, **kwargs) % spdz.field
        response = _SPDZTensor(result_child).wrap(True)
        return response

    def __mul__(self, other):
        if(isinstance(other, _SPDZTensor)):
            workers = list(self.shares.child.pointer_tensor_dict.keys())
            gp_response = spdz.spdz_mul(self.shares, other.shares, workers)
        else:
            gp_response = self.shares * other

        response = _SPDZTensor(gp_response).wrap(True)
        return response

    def mm(self, other):
        workers = list(self.shares.child.pointer_tensor_dict.keys())
        gp_response = spdz.spdz_matmul(self.shares, other.shares, workers)
        response = _SPDZTensor(gp_response).wrap(True)
        return response

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
            return cls.__mul__(self, *args, **kwargs)
        elif attr == '__add__':
            return cls.__add__(self, *args, **kwargs)
        elif attr == '__sub__':
            return cls.__sub__(self, *args, **kwargs)
        elif attr == 'sum':
            return cls.sum(self, *args, **kwargs)
        elif attr == 'mm':
            return cls.mm(self, *args, **kwargs)
        else:
            result_child = getattr(self.child, attr)(*args, **kwargs)
            return _SPDZTensor(result_child).wrap(True)

    def send(self, workers):
        self.n_workers = len(workers)
        self.shares = self.share(self.var, self.n_workers)
        self.child = self.shares
        self.workers = workers
        for share, worker in zip(self.shares, self.workers):
            share.send(worker)

    def get(self, deregister_ptr=False):
        # TODO: have deregister_ptr do something
        value = self.shares.child.sum_get() % spdz.field

        gate = (value > spdz.torch_max_value).long()

        neg_nums = (value - spdz.torch_field) * gate
        pos_nums = value * (1 - gate)
        result = neg_nums + pos_nums

        return result


class _TorchObject(object):
    """
    This tensor is simply a more convenient way to add custom
    functions to all Torch tensor types, including Torch Variable.
    Note that it is the parent class of the two following classes:
    _TorchTensor and a_TorchVariable
    """

    __module__ = 'syft'

    def get_shape(self):
        return self.child.get_shape()

    def native_get_shape(self):
        return self.get_shape()

    def share(self, *workers):
        """
        Create additive shares of a tensorvar and send them to workers
        """
        if isinstance(self.child, _PointerTensor):
            return self.child.share(*workers).wrap(True)

        elif isinstance(self.child, _FixedPrecisionTensor):
            self.child.child = self.child.child.share(*workers)
            return self

        else:
            n_workers = len(workers)
            x_enc = self._encode()
            shares = self._share(n_workers)

            pointer_shares_dict = {}
            for share, worker in zip(shares, workers):
                share.send(worker)
                pointer_shares_dict[worker] = share.child
            x_gp = _GeneralizedPointerTensor(pointer_shares_dict, torch_type='syft.LongTensor').on(self)
            x_spdz = _SPDZTensor(x_gp, torch_type='syft.LongTensor').wrap(True)

            return x_spdz

    def native_share(self, *workers):
        out = self.share(*workers)
        return out

    def _share(self, n_workers):
        if(not isinstance(self, torch.LongTensor)):
            raise TypeError("Can only MPCShare LongTensor type. You tried to share "+str(type(self).__name__)+"." +
                            " Do you need to call .fix_precision() first?")
        return spdz.share(self, n_workers)

    def _encode(self):
        return spdz.encode(self)

    def fix_precision(self,
                      bits=31,
                      base=10,
                      precision_fractional=6,
                      already_encoded=False):
        # TODO: Should fix_me be an inplace op?

        if isinstance(self.child, _PointerTensor):
            return self.owner._execute_call('fix_precision', self)
            self_ = self.child

            cmd = {
                'command': 'fix_precision',
                'self': self_,
                'args': [],
                'kwargs': {},
                'has_self': True
            }

            ptr = self_.handle_call(cmd, self_.owner)
            return ptr.wrap(True)
        else:
            fpt = lambda tensorvar, is_encoded: _FixedPrecisionTensor(tensorvar,
                                                                      torch_type=tensorvar.child.torch_type,
                                                                      bits=bits,
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
                # Add the missing .data link in the last figure
                _var.child.data = _var.data.child
                # Do the same with gradient
                if self.grad is not None:
                    _var.init_grad_()
                    _var.grad = fpt(self.grad, already_encoded)
                    _var.grad.data.child = fpt(_var.grad.child.child.data, True).child
                    _var.grad.child.data = _var.grad.data.child
                return _var
            else:
                return fpt(self, already_encoded)

    def native_fix_precision(self, *args, **kwargs):
        return self.fix_precision(*args, **kwargs)

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
            return json.dumps({key: tensor_msg}) + "\n"

    @staticmethod
    def deser(msg_obj, worker, acquire):

        obj_type, msg_obj = torch_utils.extract_type_and_obj(msg_obj)
        syft_obj = sy._SyftTensor.deser_routing(msg_obj['child'], worker, acquire)

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


class _TorchVariable(_TorchObject):

    def send(self, *workers, new_id=None, new_data_id=None, new_grad_id=None, new_grad_data_id=None):
        """
        Give the root of the chain held by self to worker
        self->alice->obj [worker] => self->worker->alice->obj
        Because there are Variable involved, there are actually 4 chains involved,
        the variable chain, variable.data, variable.grad, variable.grad.data
        """

        if len(workers) == 1:
            worker = workers[0]
        else:
            gpt_dict = {}
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
            return json.dumps({key: tensor_msg}) + "\n"

    @staticmethod
    def deser(msg_obj, worker, acquire, is_head=False):
        obj_type, msg_obj = torch_utils.extract_type_and_obj(msg_obj)
        var_syft_obj = sy._SyftTensor.deser_routing(msg_obj['child'], worker, acquire)

        if var_syft_obj.parent is not None and var_syft_obj.child is not None:
            return var_syft_obj.parent

        # Deser the var.data
        try:
            var_data_type, var_data_tensor = torch_utils.extract_type_and_obj(msg_obj['data'])
            if is_head:
                var_data = torch.guard['syft.' + var_data_type].deser(msg_obj['data'], worker, acquire)
            else:
                var_data = torch.guard['syft.' + var_data_type]()
        except AttributeError:
            var_data = torch.guard['syft.FloatTensor']()
        worker.hook.local_worker.de_register(var_data)

        variable = sy.Variable(var_data, requires_grad=msg_obj['requires_grad'])

        # Deser the var.grad
        if 'grad' in msg_obj:

            var_grad_type, var_grad_tensor = torch_utils.extract_type_and_obj(msg_obj['grad'])
            if is_head:
                var_grad = torch.guard['syft.' + var_grad_type].deser(msg_obj['grad'], worker, acquire, is_head)
            else:
                var_grad = torch.guard['syft.' + var_grad_type]()
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
        if var_grad is None:
            torch_utils.link_var_chain_to_data_chain(variable, var_data)
        else:
            torch_utils.link_var_chain_to_data_and_grad_chains(variable, var_data, var_grad)

        return variable

    def init_grad_(self):
        """
        Initialise grad as an empty tensor
        """
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
        var_grad.data = var_grad.data.type(type(self.data))

        self.grad = var_grad

        # put back original var_grad.data
        self.grad.data = var_grad_data

    # in the case of fixed precision tensors, torch tensors need this function
    def decode(self):
        var = sy.Variable(self.data.decode())
        var.child = self.child.child.child
        # if hasattr(self, 'grad') and self.grad is not None:
        #     var.grad = self.grad.decode()
        return var

    def decode_(self):
        self.data = self.data.decode()
        self.child = self.child.child.child
        # if hasattr(self, 'grad') and self.grad is not None:
        #    self.grad.decode_()


