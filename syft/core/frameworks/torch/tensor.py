import json
import torch
import random
import syft as sy
from ... import utils
import logging

import traceback

class _SyftTensor(object):
    ""
    def __init__(self, child, parent, torch_type, id=None, owner=None, skip_register=False):
        self.child = child
        self.parent = parent
        self.torch_type = torch_type

        if(self.child is not None):
            self.child.parent = self

        if(owner is not None):
            if not isinstance(owner, sy.core.workers.BaseWorker):
                owner = self.child.owner.get_worker(owner)
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

        if parent is not None and isinstance(parent, torch.Tensor):
            return parent

        ch = self.child
        while True:
            if type(ch) in torch.tensorvar_types:
                return ch
            if hasattr(ch, 'child'):
                ch = ch.child
            else:
                # FALLBACK: sometimes you have to make your
                # own parent so that PyTorch is happy to
                # run operations with torch tensor types
                x = guard[self.torch_type]()
                x.child = self
                self.parent = x
                return x

    def wrap(self):

        wrapper = self.find_torch_object_in_family_tree()
        wrapper.child = self
        return wrapper

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

    def create_pointer(self, parent=None, ptr_id=None, owner=None, location=None, id_at_location=None, register=False):
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

        if ptr_id is None:
            # Normally if there is no id specified, we keep the same as the original pointer
            # Except if the pointer is local (we don't want to overwrite!)
            if not local_pointer:
                ptr_id = self.id
            else:
                ptr_id = random.randint(0, 9999999999)

        if hasattr(self, 'torch_type'):
            torch_type = self.torch_type
        else:
            logging.warning('The tensor has not torch_type. Is it well formed?')
            torch_type = "syft." + type(self.find_torch_object_in_family_tree(parent)).__name__

        ptr = _PointerTensor(child=None,
                             parent=parent,
                             id = ptr_id,
                             torch_type=torch_type,
                             location=location,
                             id_at_location=id_at_location,
                             owner=owner,
                             skip_register=(not register))

        if not register:
           ptr.owner.rm_obj(ptr.id)

        return ptr

    def add_type_specific_attributes(self, tensor_msg):
        return tensor_msg

    def ser(self, include_data=True, *args, **kwargs):
        pass

    @staticmethod
    def deser(msg, owner, highest_level=True):
        pass

    def __str__(self):
        return "["+type(self).__name__+" - id:" + str(self.id) + " owner:" + str(self.owner.id) + "]"

    def __repr__(self):
        return self.__str__()


class _LocalTensor(_SyftTensor):

    def __init__(self, child, parent, torch_type, owner=None, id=None, skip_register=False):
        super().__init__(child=child, parent=parent, torch_type=torch_type, owner=owner, id=id, skip_register=skip_register)

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

    def __add__(self, other):
        """
        An example of how to overload a specific function given that
        the default behavior in LocalTensor (for all other operations)
        is to simply call the native PyTorch functionality.
        """

        # custom stuff we can add

        # calling the native PyTorch functionality at the end
        return self.child.add(other)

    @staticmethod
    def deser(msg_obj, register=True):

        if('child' not in msg_obj):

            # create empty object as a backup if no child is provided
            if('torch_type' in msg_obj):
                child_type = guard[msg_obj['torch_type']]
                child = child_type()
            else:
                raise Exception("Object must either have a child object or at least"+\
                                "a decided Torch type in which data will be stored")
        else:
            # get the type of the child to call the correct deser function
            child_type = guard[msg_obj['child']['type']]

            if(child_type not in torch.tensor_types):
                raise Exception("LocalTensor child object must be a Torch tensor")

            # create child object - don't deregister it yet because we need to get a
            # reference to the local worker
            child = child_type.deser(msg_obj['child'], register=True, suppress_warning=True)

        # get reference to child owner (to have access to the local_worker object)
        child_worker_reference = child.owner

        # ok... now we can deregister it. This is a little bit of a hack but it works
        # TODO: perhaps there's a better strategy for getting access to the local_worker
        # object?
        child.owner.de_register_object(child)

        owner = child_worker_reference.get_worker(msg_obj['owner'])

        if(register):
            if msg_obj['id'] in owner._objects:
                msg = "Cannot deserialize and register a tensor that already exists.\n"
                msg += "Either set register=False, remove the current tensor, or initialize\n"
                msg += "this tensor with a different id."
                raise Exception(msg)

        result = _LocalTensor(child=child,
                             owner=owner,
                             torch_type=msg_obj['torch_type'],
                             id=msg_obj['id'],
                             parent=child,
                             skip_register=not register)

        return result

    def get(self, parent, deregister_ptr=None):
        raise Exception("Cannot call .get() on a tensor you already have.")


class _PointerTensor(_SyftTensor):

    def __init__(self, child, parent, torch_type, location=None, id_at_location=None, id=None, owner=None, skip_register=False):
        super().__init__(child=child, parent=parent, torch_type=torch_type, owner=owner, id=id, skip_register=skip_register)
        if(location is None):
            raise Exception("Must have location")
        self.location = self.owner.get_worker(location)
        self.id_at_location = id_at_location
        self.torch_type = torch_type

        # pointers to themselves that get registered should trigger the flat
        # if it's not getting registered the pointer is probably about to be
        # sent over the wire
        if self.location == self.owner and not skip_register:
            logging.warning("Do you really want a pointer pointing to itself? (self.location == self.owner)")

    def __str__(self):
        return "["+type(self).__name__+" - id:" + str(self.id) + " owner:" + str(self.owner.id) +  " loc:" + str(self.location.id) + " id@loc:"+str(self.id_at_location)+"]"

    @staticmethod
    def deser(msg_obj, child, owner):

        if 'id' not in msg_obj.keys():
            msg_obj['id'] = random.randint(0,9999999999)
        obj = _PointerTensor(child=child,
                             parent=None,
                             owner=owner,
                             id=msg_obj['id'],
                             location=msg_obj['location'],
                             id_at_location=msg_obj['id_at_location'],
                             torch_type = msg_obj['torch_type']
                             )
        return obj

    def wrap(self): # TODO do it in a smart (and dual?) way
        wrapper = guard[self.torch_type]()
        self.owner.rm_obj(wrapper.child.id)
        wrapper.child = self
        return wrapper

    def get(self, parent, deregister_ptr=True):

        # Remove this pointer
        if deregister_ptr:
           self.owner.rm_obj(self.id)

        # if the pointer happens to be pointing to a local object,
        # just return that object (this is an edge case)
        if self.location == self.owner:
            return self.owner._objects[self.id_at_location]

        # get SyftTensor (Local or Pointer) from remote machine
        syft_tensor = self.owner.request_obj(self.id_at_location, self.location)
        utils.assert_has_only_syft_tensors(syft_tensor)
        # TODO: Maybe we x.get() should change its id with the id of the remote (but then beware that the registration is properly done), but maybe not.
        syft_tensor.id = self.id

        # Register the result
        self.owner.register(syft_tensor)

        return syft_tensor

    def add_type_specific_attributes(self, tensor_msg):
        tensor_msg['location'] = self.location if isinstance(self.location, str) else self.location.id
        tensor_msg['id_at_location'] = self.id_at_location
        tensor_msg['torch_type'] = self.torch_type
        return tensor_msg

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

class _TorchObject(object):
    """
    This tensor is simply a more convenient way to add custom
    functions to all Torch tensor types.
    """

    __module__ = 'syft'

    def __str__(self):
        return self.native___str__()

    def __repr__(self):
        return self.native___repr__()

    def create_pointer(self, register=False, location=None, ptr_id=None):

        return self.child.create_pointer(parent=self, register=register, location=location, ptr_id=ptr_id).wrap()

    def create_local_tensor(self, worker, id=None):
        tensor = sy._LocalTensor(child=self,
                                parent=self,
                                torch_type='syft.' + type(self).__name__,
                                owner=worker,
                                id=None)
        return tensor      


    def get(self, deregister_ptr=True, update_ptr_wrapper=True):

        # returns a SyftTensor object (Local or Pointer) without a parent wrapper
        syft_tensor = self.child.get(parent=self, deregister_ptr=deregister_ptr)

        # this will change the pointer variable (wrapper) to instead wrap the
        # SyftTensor object that was returned so that any variable that may
        # still exist referencing this pointer will simply call local data instead
        # of sending messages elsewhere, or a closer pointer
        if update_ptr_wrapper:
            # TODO: modifying self is maybe not a good practice, but it's the only way I found
            # to ensure that x.get() which shows a tensor is in fact of x.dim() = 1 and not 0
            self.child = syft_tensor

        return self

    def move(self, worker, new_id=None):
        """
        Give the end leaf of the chain to worker,
        just like if the last elmt was send its child
        to worker
        self->alice->obj [worker] => self->alice->worker->obj
        """
        raise Exception('Move is not supported anymore.')
        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        if new_id is None:
            new_id = random.randint(0,9999999999)

        pointer = self.child.find_pointer()

        if pointer is None:
            return self.send(worker, new_id)

        command, _ = pointer.compile_command('move',
                                          (worker.id, new_id),
                                          {},
                                          True)

        response = pointer.owner.send_torch_command(recipient=pointer.location,
                                                    message=command)
        return self

class _TorchTensor(_TorchObject):

    def ser(self, include_data=True, stop_recurse_at_torch_type=False, as_dict=True):
        """Serializes a {} object to JSON.""".format(type(self))
        if(not stop_recurse_at_torch_type):
            serializations = self.child.ser(include_data=include_data, as_dict=True)
            serializations['torch_type'] = "syft."+type(self).__name__
            if(as_dict):
                return serializations
            else:
                return json.dumps(serializations) + "\n"
        else:
            tensor_msg = {}
            tensor_msg['type'] = str(self.__class__).split("'")[1]
            tensor_msg['torch_type'] = "syft."+type(self).__name__
            if include_data:
                tensor_msg['data'] = self.tolist()

            if(as_dict):
                return tensor_msg
            else:
                return json.dumps(tensor_msg) + "\n"

    @staticmethod
    def deser(msg_obj, register=True, suppress_warning=False):

        if('data' in msg_obj):
            if register and not suppress_warning:
                msg = "Registering a data holding tensor is not advised.\n"
                msg += "our system is designed for data tensors to only be \n"
                msg += "called by LocalTensor pointers which are themselves registered.\n"
                msg += "The system will attempt to handle this gracefully but this could\n"
                msg += "result in undefined behavior. Are you sure you want to do this?"
                msg += "If so, set suppress_warning=True to not display this message."
                logging.warn(msg)
            torch_type = guard[msg_obj['torch_type']]
            result = torch_type(msg_obj['data'])

            if(not register):
                result.owner.de_register_object(result)
        else:
            child_type = guard[msg_obj['type']]
            result = child_type.deser(msg_obj=msg_obj, register=register).wrap()

        return result

    def send(self, worker, ptr_id=None):
        """
        Give the root of the chain held by self to worker
        self->alice->obj [worker] => self->worker->alice->obj

        Args:
            ptr_id: the id of the object when sent:
                x.send(bob, 1000)
                will result in bob having the tensor x with id 1000
        """

        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        if ptr_id is None:
            ptr_id = random.randint(0, 9999999999)

        obj_id = self.child.id

        # creates a pointer to LocalTensor without a Torch object wrapping it because
        # we're going to set self.child to be this pointer.
        # we set register=True because we want it to be registered locally


        self.owner.send_obj(self.child,
                            ptr_id,
                            worker)

        # clears data which could be cached in the wrapper (which is self)
        # which would be confusing for folks
        self.native_set_()

        # set this wrapper's child to be the newly created PointerTensor
        self.child.id = obj_id
        x_ptr = self.child.create_pointer(location=worker, id_at_location=ptr_id, register=True)
        self.child = x_ptr

        return self


    def __str__(self):
        if isinstance(self.child, _PointerTensor):
            return type(self).__name__+self.child.__str__()+""
        elif(isinstance(self.child, _LocalTensor)):
            if(hasattr(self.child, 'child')):
                return self.child.child.native___str__()
            else:
                return "Empty Wrapper:\n" + self.native___str__()
        else:
            return self.native___str__()

class _TorchVariable(_TorchObject):

    def send(self, worker, new_id=None, new_data_id=None):
        """
        Give the root of the chain held by self to worker
        self->alice->obj [worker] => self->worker->alice->obj
        """

        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        if new_id is None:
            new_id = random.randint(0,9999999999)

        if new_data_id is None:
            new_data_id = random.randint(0,9999999999)

        # if new_grad_id is None:
        #     new_grad_id = random.randint(0,9999999999)

        old_data_id = self.data.id
        self.data.child.id = new_data_id

        init_id = self.id

        self.owner.send_obj(self,
                            new_id,
                            worker)

        self.native_set_()

        self.child = sy._PointerTensor(child=self,
                                       parent=self,
                                       id=init_id,
                                       torch_type='syft.'+type(self).__name__,
                                       location=worker,
                                       id_at_location=new_id)

        self.data.child = sy._PointerTensor(child=self,
                                            parent=self,
                                            id=old_data_id,
                                            torch_type='syft.'+type(self).__name__,
                                            location=worker,
                                            id_at_location=new_data_id)

        return self

    def get(self):
        new_child_obj = self.child.get(parent=self)
        new_data_obj = self.data.child.get(parent=self)
        self.child = new_child_obj
        self.data.child = new_data_obj

        self.native_set_(self.child.child)

        return self

    def ser(self):
        pass

guard = {
    'syft.core.frameworks.torch.tensor.Variable': torch.autograd.Variable,
    'syft.core.frameworks.torch.tensor._PointerTensor': _PointerTensor,
    'syft.core.frameworks.torch.tensor._SyftTensor': _SyftTensor,
    'syft.core.frameworks.torch.tensor._LocalTensor': _LocalTensor,
    'syft.core.frameworks.torch.tensor._FixedPrecisionTensor': _FixedPrecisionTensor,
    'syft.core.frameworks.torch.tensor.FloatTensor': torch.FloatTensor,
    'syft.core.frameworks.torch.tensor.DoubleTensor': torch.DoubleTensor,
    'syft.core.frameworks.torch.tensor.HalfTensor': torch.HalfTensor,
    'syft.core.frameworks.torch.tensor.ByteTensor': torch.ByteTensor,
    'syft.core.frameworks.torch.tensor.CharTensor': torch.CharTensor,
    'syft.core.frameworks.torch.tensor.ShortTensor': torch.ShortTensor,
    'syft.core.frameworks.torch.tensor.IntTensor': torch.IntTensor,
    'syft.core.frameworks.torch.tensor.LongTensor': torch.LongTensor,
    'syft.Variable': torch.autograd.Variable,
    'syft.FloatTensor': torch.FloatTensor,
    'syft.DoubleTensor': torch.DoubleTensor,
    'syft.HalfTensor': torch.HalfTensor,
    'syft.ByteTensor': torch.ByteTensor,
    'syft.CharTensor': torch.CharTensor,
    'syft.ShortTensor': torch.ShortTensor,
    'syft.IntTensor': torch.IntTensor,
    'syft.LongTensor': torch.LongTensor
}
