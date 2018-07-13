import json
import torch
import random
import syft as sy
from ... import utils

class _SyftTensor(object):
    ""
    
    def __init__(self, child, parent, id=None, owner=None):
        self.child = child
        self.parent = parent

        if(self.child is not None):
            self.child.parent = self

        if(owner is not None):
            self.owner = owner

    def find_torch_object_in_family_tree(self):

        ch = self.child
        while(True):
            if issubclass(type(ch), torch.Tensor):
                return ch

            ch = ch.child

    @property
    def parent(self):
        if(hasattr(self, '_parent') and self._parent is not None):
            return self._parent
        else:
            self._parent = self.find_torch_object_in_family_tree()
            return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    def create_pointer(self, register=False):
        ptr = _PointerTensor(child=None,
                             parent=None,
                             location=self.owner.id,
                             id_at_location=self.id,
                             owner=self.owner,
                             torch_type="syft."+type(self.find_torch_object_in_family_tree()).__name__)

        if(not register):
            ptr.owner.rm_obj(ptr.id)

        return ptr

    # def __str__(self):
        # return "blah"

    def add_type_specific_attributes(self, tensor_msg):
        return tensor_msg

    def ser(self, include_data=True, *args, **kwargs):

        tensor_msg = {}
        tensor_msg['type'] = str(self.__class__).split("'")[1]
        if hasattr(self, 'child') and self.child is not None:
            tensor_msg['child'] = self.child.ser(include_data=include_data,
                                                 stop_recurse_at_torch_type=True)
        tensor_msg['id'] = self.id
        owner_type = type(self.owner)
        if (owner_type is int or owner_type is str):
            tensor_msg['owner'] = self.owner
        else:
            tensor_msg['owner'] = self.owner.id

        tensor_msg = self.add_type_specific_attributes(tensor_msg)

        return tensor_msg

    @staticmethod
    def deser(msg, highest_level=True):

        if isinstance(msg, str):
            msg_obj = json.loads(msg)
        else:
            msg_obj = msg

        obj_type = guard[msg_obj['type']]

        if('child' in msg_obj):
            # deserialize syft object and children

            child, leaf = _SyftTensor.deser(msg_obj['child'], highest_level=False)
            obj = obj_type(child=child, parent=None)
            obj.id = msg_obj['id']

        elif('data' in msg_obj):
            # deserialize torch variable object
            obj = obj_type(msg_obj['data'])
            return obj, obj
        else:
            # deserialize data-less object - likely a pointer
            obj = obj_type(child = None,
                           parent = None,
                           id=msg_obj['id'],
                           location = msg_obj['location'],
                           id_at_location = msg_obj['id_at_location'],
                           torch_type = msg_obj['torch_type'])
            return obj

        if(highest_level):
            leaf.child = obj
            return leaf
        
        return obj, leaf

        


class _LocalTensor(_SyftTensor):

    def __init__(self, child, parent, owner=None):
        super().__init__(child=child, parent=parent, owner=owner)
        
    def __add__(self, other):
        """
        An example of how to overload a specific function given that 
        the default behavior in LocalTensor (for all other operations)
        is to simply call the native PyTorch functionality.
        """
        
        # custom stuff we can add
        # print("adding2")
        
        # calling the native PyTorch functionality at the end
        return self.child.add(other)

class _PointerTensor(_SyftTensor):
    
    def __init__(self, child, parent, location=None, id_at_location=None, id=None, owner=None, torch_type=None):
        super().__init__(child=child, parent=parent, owner=owner, id=id)
        self.location = self.owner.get_worker(location)
        self.id_at_location = id_at_location
        self.torch_type = torch_type

    def __add__(self, *args, **kwargs):

        # Step 1: Compiles Command
        command = self.compile_command("__add__",
                                  args,
                                  kwargs,
                                  True)

        response = self.owner.send_torch_command(recipient=self.location,
                                                 message=command)
        return sy.deser(response).wrap()

    def wrap(self):
        wrapper = guard[self.torch_type]()
        self.owner.rm_obj(wrapper.child.id)
        wrapper.child = self
        return wrapper

    def add_type_specific_attributes(self, tensor_msg):
        tensor_msg['location'] = self.location
        tensor_msg['id_at_location'] = self.id_at_location
        tensor_msg['torch_type'] = self.torch_type
        return tensor_msg

    def compile_command(self, attr, args, kwargs, has_self):
    
        command = {}

        command['has_self'] = has_self


        command['self'] = self.id_at_location

        command['command'] = attr
        command['args'] = utils.map_tuple(None, args, self._tensors_to_str_ids)
        command['kwargs'] = utils.map_dict(None, kwargs, self._tensors_to_str_ids)
        command['arg_types'] = [type(x).__name__ for x in args]
        command['kwarg_types'] = [type(kwargs[x]).__name__ for x in kwargs]

        kwarg_types = command['arg_types']
        arg_types = command['arg_types']

        return command

    @staticmethod
    def _tensors_to_str_ids(tensor):
        """This method takes a tensor/var/param and replaces it with a
        string containing it's ID and special flag for recognizing that
        it's a tensor type arg instead of a string.

        This method also works for an iterable of tensors (e.g. `torch.cat([x1, x2, x3])`)
        """
        if hasattr(torch, 'native_is_tensor'):
            check = torch.native_is_tensor
        else:
            check = torch.is_tensor
        try:
            _is_param = isinstance(tensor, torch.nn.Parameter)
            if check(tensor) or isinstance(tensor, torch.autograd.Variable) or _is_param:
                return tensor.child.id_at_location
            else:
                [_tensors_to_str_ids(i) for i in tensor]
        except (AttributeError, TypeError):
            return tensor


class _FixedPrecisionTensor(_SyftTensor):
    
    def __init__(self, child, parent, owner=None):
        super().__init__(child=child, parent=parent, owner=owner)

class _TorchTensor(object):
    """
    This tensor is simply a more convenient way to add custom
    functions to all Torch tensor types.
    """

    __module__ = 'syft'

    def __str__(self):
        return self.native___str__()

    def __repr__(self):
        return self.native___repr__()

    def create_pointer(self, register=False):
        return self.child.create_pointer(register=register)

    def send(self, worker, new_id=random.randint(0,9999999999)):


        self.owner.send_obj(self,
                            new_id,
                            worker,
                            delete_local=True)

        self.set_(sy.zeros(0))

        self.child = sy._PointerTensor(child=None,
                                       parent=self,
                                       location=worker,
                                       id_at_location=new_id)

        return self

    def ser(self, include_data=True, stop_recurse_at_torch_type=False):
        """Serializes a {} object to JSON.""".format(type(self))
        if(not stop_recurse_at_torch_type):
            serializations = self.child.ser(include_data=include_data)
            return json.dumps(serializations) + "\n"
        else:
            tensor_msg = {}
            tensor_msg['type'] = str(self.__class__).split("'")[1]
            if include_data:
                tensor_msg['data'] = self.tolist()

            return tensor_msg

guard = {
    'syft.core.frameworks.torch.tensor._PointerTensor': _PointerTensor,
    'syft.core.frameworks.torch.tensor._SyftTensor': _SyftTensor,
    'syft.core.frameworks.torch.tensor._LocalTensor': _LocalTensor,
    'syft.core.frameworks.torch.tensor._FixedPrecisionTensor': _FixedPrecisionTensor,
    'syft.FloatTensor': torch.FloatTensor,
    'syft.DoubleTensor': torch.DoubleTensor,
    'syft.HalfTensor': torch.HalfTensor,
    'syft.ByteTensor': torch.ByteTensor,
    'syft.CharTensor': torch.CharTensor,
    'syft.ShortTensor': torch.ShortTensor,
    'syft.IntTensor': torch.IntTensor,
    'syft.LongTensor': torch.LongTensor
}
