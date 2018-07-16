import json
import torch
import random
import syft as sy
from ... import utils
import logging

class _SyftTensor(object):
    ""

    def __init__(self, child, parent, torch_type, id=None, owner=None, skip_register=False):
        self.child = child
        self.parent = parent
        self.torch_type = torch_type

        if(self.child is not None):
            self.child.parent = self

        if(owner is not None):
            self.owner = owner

    def copy_params(self, other):
        self.id = other.id

    def find_pointer(self):
        ch = self
        if isinstance(ch, sy._PointerTensor):
            return ch
        else:
            return None

    def find_torch_object_in_family_tree(self, parent=None):

        if(parent is not None and isinstance(parent, torch.Tensor)):
            return parent

        ch = self.child
        while(True):
            if issubclass(type(ch), torch.Tensor):
                return ch
            if(hasattr(ch, 'child')):
                ch = ch.child
            else:
                # FALLABCK: sometimes you have to make your
                # own parent so that PyTorch is happy to
                # run operations with torch tensor types
                x = sy.FloatTensor()
                x.child = self
                self.parent = x
                return x


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

    def create_pointer(self, parent, register=False):
        ptr = _PointerTensor(child=None,
                             parent=parent,
                             torch_type="syft."+type(self.find_torch_object_in_family_tree(parent)).__name__,
                             location=self.owner.id,
                             id_at_location=self.id,
                             owner=self.owner,
                             skip_register=(not register))

        #if(not register):
        #    ptr.owner.rm_obj(ptr.id)

        return ptr

    # def __str__(self):
        # return "blah"

    def add_type_specific_attributes(self, tensor_msg):
        return tensor_msg

    def ser(self, include_data=True, *args, **kwargs):

        tensor_msg = {}
        tensor_msg['type'] = str(self.__class__).split("'")[1]
        tensor_msg['torch_type'] = "syft."+type(self.parent).__name__
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
    def deser(msg, owner=None, highest_level=True):

        if isinstance(msg, str):
            msg_obj = json.loads(msg)
        else:
            msg_obj = msg

        obj_type = guard[msg_obj['type']]

        if('child' in msg_obj):
            # deserialize syft object and children

            child, leaf = _SyftTensor.deser(msg_obj['child'], owner=owner, highest_level=False)
            # likely using VirtualWorkers and accidentally registered this
            # object to the default local_worker
            if(child.owner.id != owner.id):
                child.owner.rm_obj(child.id)

            obj = obj_type.deser(msg_obj=msg_obj, child=child, owner=owner)
            # obj = obj_type(child=child, owner=owner, id=msg_obj['id'], parent=None)

        elif('data' in msg_obj):
            # deserialize torch variable object
            obj = obj_type(msg_obj['data'])

            # unfortunately the LocalTensor that gets initialzied when
            # creating the lowest level torch.Tensor object is always
            # redundant, so we need to remove it.
            # TOOD: figure out how to avoid this performance waste.
            obj.owner.rm_obj(obj.id)

            return obj, obj
        else:
            # deserialize data-less object - likely a pointer
            obj = obj_type.deser(msg_obj=msg_obj, child=None, owner=owner)
            # obj = obj_type(child = None,
            #                parent = None,
            #                id=msg_obj['id'],
            #                location = msg_obj['location'],
            #                id_at_location = msg_obj['id_at_location'],
            #                torch_type = msg_obj['torch_type'])
            return obj

        if(highest_level):
            leaf.child = obj
            return leaf
        
        return obj, leaf

    def __str__(self):
        return "<"+type(self).__name__+" - id:" + str(self.id) + " owner:" + str(self.owner.id) + ">"        

    def __repr__(self):
        return self.__str__()


class _LocalTensor(_SyftTensor):

    def __init__(self, child, parent, torch_type, owner=None, id=None, skip_register=False):
        super().__init__(child=child, parent=parent, torch_type=torch_type, owner=owner, id=id, skip_register=skip_register)
        
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

    @staticmethod
    def deser(msg_obj, child, owner):
        return _LocalTensor(child=child,
                            owner=owner,
                            torch_type=msg_obj['torch_type'],
                            id=msg_obj['id'],
                            parent=None)

    def get(self, parent):
        raise Exception("Cannot call .get() on a tensor you already have.")

class _PointerTensor(_SyftTensor):
    
    def __init__(self, child, parent, torch_type, location=None, id_at_location=None, id=None, owner=None, skip_register=False):
        super().__init__(child=child, parent=parent, torch_type=torch_type, owner=owner, id=id, skip_register=skip_register)
        if(location is None):
            raise Exception("Must have location")
        self.location = self.owner.get_worker(location)
        self.id_at_location = id_at_location
        self.torch_type = torch_type
        if self.location == self.owner:
            logging.warning("Do you really want a pointer pointing to itself? (self.location == self.owner)")

    def __add__(self, *args, **kwargs):

        # Step 1: Compiles Command
        command = self.compile_command("__add__",
                                  args,
                                  kwargs,
                                  True)

        response = self.owner.send_torch_command(recipient=self.location,
                                                 message=command)
        return sy.deser(response).wrap()

    def __str__(self):
        return "<"+type(self).__name__+" - id:" + str(self.id) + " owner:" + str(self.owner.id) +  " loc:" + str(self.location.id) + " id@loc:"+str(self.id_at_location)+">"        

    def deser(msg_obj, child, owner):
        obj = _PointerTensor(child=child,
                             parent=None,
                             owner=owner,
                             id=msg_obj['id'],                             
                             location=msg_obj['location'],
                             id_at_location=msg_obj['id_at_location'],
                             torch_type = msg_obj['torch_type']
                             )
        return obj

    def wrap(self):
        wrapper = guard[self.torch_type]()
        self.owner.rm_obj(wrapper.child.id)
        wrapper.child = self
        return wrapper

    def get(self, parent):

        obj, cleanup = self.owner.request_obj(self.id_at_location, self.location)

        if(isinstance(obj, torch.Tensor)):
            parent.native_set_(obj)

        self.owner.get_worker(obj.owner).rm_obj(obj.id)
        self.owner.set_obj(self.id, obj.child)

        if(hasattr(obj, 'child') and obj.child is not None):
            obj.child.id = self.id
            self = obj.child
        else:
            obj.id = self.id
            self = obj
        parent.child = self
        return self

    def add_type_specific_attributes(self, tensor_msg):
        tensor_msg['location'] = self.location if isinstance(self.location, str) else self.location.id
        tensor_msg['id_at_location'] = self.id_at_location
        tensor_msg['torch_type'] = self.torch_type
        return tensor_msg

    def compile_command(self, attr, args, kwargs, has_self): #self, attr, args, kwargs, has_self):
        command = {}
        command['has_self'] = has_self
        if has_self:
            command['self'] = self # TODO .id_at_location
            #args = args[1:] # TODO compare to master
        command['command'] = attr
        command['args'] = args
        command['kwargs'] = kwargs

        encoder = utils.PythonEncoder()
        command, tensorvars = encoder.encode(command, retrieve_tensorvar=True)
        return command, tensorvars

    @staticmethod
    def _tensors_to_str_ids(tensor):
        """This method takes a tensor/var/param and replaces it with a
        string containing it's ID and special flag for recognizing that
        it's a tensor type arg instead of a string.

        This method also works for an iterable of tensors (e.g. `torch.cat([x1, x2, x3])`)
        """
        if issubclass(tensor.__class__, sy._SyftTensor):
            raise TypeError('Calling _tensors_to_str_ids on non-tensor/var/param but sy._SyftTensor')

        if isinstance(tensor, (int, str)):
            return tensor

        if hasattr(torch, 'native_is_tensor'):
            check = torch.native_is_tensor
        else:
            check = torch.is_tensor
        try:
            _is_param = isinstance(tensor, torch.nn.Parameter)
            if check(tensor) or isinstance(tensor, torch.autograd.Variable) or _is_param:
                return tensor.child.id_at_location
            else:
                [_PointerTensor._tensors_to_str_ids(i) for i in tensor]
        except (AttributeError, TypeError):
            return tensor


class _FixedPrecisionTensor(_SyftTensor):
    
    def __init__(self, child, parent, torch_type, owner=None):
        super().__init__(child=child, parent=parent, torch_type=torch_type, owner=owner)

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
        return self.child.create_pointer(parent=self, register=register)

    def get(self):
        new_child_obj = self.child.get(parent=self)
        return self

    def move(self, worker, new_id=None):
        """
        Give the end leaf of the chain to worker,
        just like if the last elmt was send its child
        to worker
        self->alice->obj [worker] => self->alice->worker->obj
        """
        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        if new_id is None:
            new_id = random.randint(0,9999999999)

        pointer = self.child.find_pointer()

        if pointer is None:
            return self.send(worker, new_id)

        command = pointer.compile_command('move',
                                          (worker.id, new_id),
                                          {},
                                          True)

        response = pointer.owner.send_torch_command(recipient=pointer.location,
                                                    message=command)
        return self

    def send(self, worker, new_id=None):
        """
        Give the root of the chain held by self to worker
        self->alice->obj [worker] => self->worker->alice->obj
        """
        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        if new_id is None:
            new_id = random.randint(0,9999999999)

        init_id = self.id

        self.owner.send_obj(self,
                            new_id,
                            worker,
                            delete_local=True)

        self.native_set_()

        self.child = sy._PointerTensor(child=self,
                                       parent=self,
                                       id=init_id,
                                       torch_type='syft.'+type(self).__name__,
                                       location=worker,
                                       id_at_location=new_id)

        return self

    def ser(self, include_data=True, stop_recurse_at_torch_type=False):
        """Serializes a {} object to JSON.""".format(type(self))
        if(not stop_recurse_at_torch_type):
            serializations = self.child.ser(include_data=include_data)
            serializations['torch_type'] = "syft."+type(self).__name__
            return json.dumps(serializations) + "\n"
        else:
            tensor_msg = {}
            tensor_msg['type'] = str(self.__class__).split("'")[1]
            tensor_msg['torch_type'] = "syft."+type(self).__name__
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
