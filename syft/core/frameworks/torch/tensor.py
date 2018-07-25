import json
import re
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

        if self.child is not None:
            self.child.parent = self

        if owner is not None:
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
                self.owner.de_register(x.child)
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
            return None

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

        if ptr_id is not None:
            if(ptr_id == id_at_location):
                raise AttributeError("The PointerTensor and the tensor being pointed to cannot have the same id.")

        else:
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
        raise NotImplementedError('No general ser() function for Syft')

    @staticmethod
    def deser(dct, worker, acquire):
        pat = re.compile('__(.+)__')
        for key, obj in dct.items(): # A trick, we don't really loop
            obj_type = pat.search(key).group(1)
            if utils.is_syft_tensor(obj_type):
                if obj_type == '_LocalTensor':
                    return sy._LocalTensor.deser(obj, worker, acquire)
                elif obj_type == '_PointerTensor':
                    return sy._PointerTensor.deser(obj, worker, acquire)
                else:
                    raise TypeError('SyftTensor', obj_type, 'is not supported so far')


    def __str__(self):
        return "["+type(self).__name__+" - id:" + str(self.id) + " owner:" + str(self.owner.id) + "]"

    def __repr__(self):
        return self.__str__()


class _LocalTensor(_SyftTensor):

    def __init__(self, child, parent, torch_type, owner=None, id=None, skip_register=False):
        super().__init__(child=child, parent=parent, torch_type=torch_type, owner=owner, id=id, skip_register=skip_register)

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
    def deser(msg_obj, worker, acquire):
        if not 'owner' in msg_obj:
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


class _PointerTensor(_SyftTensor):

    def __init__(self, child, parent, torch_type, location=None, id_at_location=None, id=None, owner=None, skip_register=False):
        super().__init__(child=child, parent=parent, torch_type=torch_type, owner=owner, id=id, skip_register=skip_register)
        if location is None:
            raise AttributeError("Pointer must have a location specified")
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

    @staticmethod
    def deser(msg_obj, worker, acquire):
        # If local, we render the object or syft object
        if msg_obj['location'] == worker.id:
            syft_obj = worker.get_obj(msg_obj['id_at_location'])
        else:
            if acquire:  # If there is data transmission, data being here Pointer
                # We acquire the tensor pointer
                previous_pointer = worker.get_pointer_to(msg_obj['owner'], msg_obj['id'])
                if previous_pointer is None:
                    syft_obj = sy._PointerTensor(child=None,
                                                 parent=None,
                                                 torch_type=msg_obj['torch_type'],
                                                 location=msg_obj['location'],
                                                 id_at_location=msg_obj['id_at_location'],
                                                 owner=worker,
                                                 id=msg_obj['id'],
                                                 skip_register=True)
                else:
                    syft_obj = previous_pointer
            else:  # We point at the Pointer
                owner = worker.get_worker(msg_obj['owner'])
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

    def wrap(self): # TODO do it in a smart (and dual?) way
        wrapper = guard[self.torch_type]()
        self.owner.rm_obj(wrapper.child.id)
        wrapper.child = self
        utils.fix_chain_ends(wrapper)
        return wrapper

    def get(self, deregister_ptr=True):
        # Remove this pointer
        if deregister_ptr:
            if self.torch_type == 'syft.Variable':
                self.owner.rm_obj(self.parent.data.child.id)
            self.owner.rm_obj(self.id)

        # if the pointer happens to be pointing to a local object,
        # just return that object (this is an edge case)
        if self.location == self.owner:
            return self.owner._objects[self.id_at_location].child

        # get SyftTensor (Local or Pointer) from remote machine
        tensorvar = self.owner.request_obj(self.id_at_location, self.location)
        utils.assert_has_only_torch_tensorvars(tensorvar)

        syft_tensor = tensorvar.child
        syft_tensor.id = self.id
        if self.torch_type == 'syft.Variable':
            tensorvar.data.child.id = self.parent.data.child.id

        # Register the result
        self.owner.register(syft_tensor)
        if syft_tensor.torch_type == 'syft.Variable':
            self.owner.register(tensorvar.data.child)

        utils.fix_chain_ends(tensorvar)
        return tensorvar

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

        # returns a Tensor object wrapping a SyftTensor
        tensor = self.child.get(deregister_ptr=deregister_ptr)
        utils.assert_has_only_torch_tensorvars(tensor)
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
                self.set_(tensor)
            utils.fix_chain_ends(self)
            utils.assert_is_chain_well_formed(self)

        return self

    def move(self, worker, new_id=None):
        """
        Give the end leaf of the chain to worker,
        just like if the last elmt was send its child
        to worker
        self->alice->obj [worker] => self->alice->worker->obj
        """
        raise NotImplementedError('Move is not supported anymore.')
        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        if new_id is None:
            new_id = random.randint(0,10e10)

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
        obj_type, msg_obj = utils.extract_type_and_obj(msg_obj)
        syft_obj = sy._SyftTensor.deser(msg_obj['child'], worker, acquire)
        data = msg_obj['data']
        # TODO: Find a smart way to skip register and not leaking the info to the local worker
        # This would imply overload differently the __init__ to provide an owner for the child attr.
        tensorvar = eval('sy.' + obj_type)(data)
        worker.hook.local_worker.de_register(tensorvar)
        # This is a special case where we want to get rid of the empty wrapper
        if syft_obj.child is not None and len(data) == 0:
            return syft_obj.child
        tensorvar.child = syft_obj
        syft_obj.parent = tensorvar
        return tensorvar

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

        self.owner.send_obj(self,
                            ptr_id,
                            worker)

        # clears data which could be cached in the wrapper (which is self)
        # which would be confusing for folks
        self.native_set_()

        # set this wrapper's child to be the newly created PointerTensor
        self.child.id = obj_id
        x_ptr = self.child.create_pointer(location=worker, id_at_location=ptr_id, register=True)
        self.child = x_ptr
        x_ptr.parent = self
        self.parent = None

        return self

    def __str__(self):
        if isinstance(self.child, _PointerTensor):
            return type(self).__name__+self.child.__str__()+""
        elif isinstance(self.child, _LocalTensor) and utils.is_tensor_empty(self):
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

        obj_id = self.child.id
        obj_data_id = self.data.child.id

        self.owner.send_obj(self,
                            new_id,
                            worker,
                            new_data_id=new_data_id)


        # clears data which could be cached in the wrapper (which is self)
        # which would be confusing for folks
        self.native_set_()
        self.data.native_set_()

        # set this wrapper's child to be the newly created PointerTensor
        self.child.id = obj_id
        var_ptr = self.child.create_pointer(location=worker, id_at_location=new_id, register=True)
        self.child = var_ptr
        var_ptr.parent = self
        self.parent = None

        # same for data
        self.data.child.id = obj_data_id
        var_data_ptr = self.data.child.create_pointer(location=worker, id_at_location=new_data_id, register=True)
        self.data.child = var_data_ptr
        var_data_ptr.parent = self.data
        self.data.parent = None

        return self

    def get(self, deregister_ptr=True, update_ptr_wrapper=True):

        # returns a Variable object wrapping a SyftTensor
        variable = self.child.get(deregister_ptr=deregister_ptr)
        utils.assert_has_only_torch_tensorvars(variable)
        # this will change the wrapper variable to instead wrap the
        # SyftTensor object that was returned so that any variable that may
        # still exist referencing this pointer will simply call local data instead
        # of sending messages elsewhere, or a closer pointer
        if update_ptr_wrapper:
            self.child = variable.child
            self.data.child = variable.data.child

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
                self.set_(variable)
            utils.fix_chain_ends(self)
            utils.assert_is_chain_well_formed(self)

        return self

    def ser(self, private, as_dict=True):
        key = '__' + type(self).__name__ + '__'
        data = self.data.ser(private)
        tensor_msg = {
            'type': str(self.__class__).split("'")[1],
            'torch_type': 'syft.' + type(self).__name__,
            'data': data,
            'child': self.child.ser(private),
            'requires_grad': self.requires_grad
        }
        if as_dict:
            return {key: tensor_msg}
        else:
            return json.dumps({key: tensor_msg}) + "\n"

    @staticmethod
    def deser(msg_obj, worker, acquire):
        obj_type, msg_obj = utils.extract_type_and_obj(msg_obj)
        var_syft_obj = sy._SyftTensor.deser(msg_obj['child'], worker, acquire)
        # Deser the var.data
        var_data_type, var_data_tensor = utils.extract_type_and_obj(msg_obj['data'])
        if utils.is_tensor(var_data_type):
            var_data =  eval('sy.' + var_data_type).deser(msg_obj['data'], worker, acquire)
            worker.hook.local_worker.de_register(var_data)
        # TODO: Find a smart way to skip register and not leaking the info to the local worker
        # This would imply overload differently the __init__ to provide an owner for the child attr.
        variable = sy.Variable(var_data, requires_grad=msg_obj['requires_grad'])
        worker.hook.local_worker.de_register(variable)

        variable.child = var_syft_obj
        var_syft_obj.parent = variable

        return variable


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
