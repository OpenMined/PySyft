import json
import logging
import random
import re
import torch
import syft as sy
from syft.core.frameworks.torch import torch_utils
from syft.core.frameworks.torch.tensor import _PointerTensor
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

    def set_id(self, new_id):
        """
        This changes the id of a tensor.
        :param new_id: a string or integer id
        :return: returns self, for convenience.
        """


        if(new_id not in self.owner._objects):
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

    def wrap(self):
        """
        Wrap a syft node with a torch wrapper
        """
        wrapper = torch.guard[self.torch_type]()
        self.owner.rm_obj(wrapper.child.id)
        wrapper.child = self
        torch_utils.fix_chain_ends(wrapper)
        return wrapper

    @classmethod
    def syft_wrap(cls, result, owner):
        """
        Wrap a torch node with a syft wrapper
        """
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
