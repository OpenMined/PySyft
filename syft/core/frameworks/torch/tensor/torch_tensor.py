import json
import random
import torch
import syft as sy
from syft.core.frameworks.torch.tensor import _TorchObject
from syft.core.frameworks.torch.tensor import _PointerTensor
from syft.core.frameworks.torch.tensor import _LocalTensor
from syft.core.frameworks.torch.tensor import _GeneralizedPointerTensor
from syft.core.frameworks.torch import torch_utils


class _TorchTensor(_TorchObject):

    def __str__(self):
        if isinstance(self.child, _PointerTensor):
            return type(self).__name__ + self.child.__str__() + ""
        elif isinstance(self.child, _LocalTensor) and torch_utils.is_tensor_empty(self):
            if (hasattr(self.child, 'child')):
                return self.child.child.native___str__()
            else:
                return "Empty Wrapper:\n" + self.native___str__()
        else:
            if not isinstance(self.child, (sy._LocalTensor, sy._PointerTensor)):
                x_ = type(self)()
                x_.native_set_(self)
                return "[Head of chain]\n" + x_.native___repr__()
            return self.native___str__()

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
        pointers_dict = {}
        for worker in workers:
            pointers_dict[worker] = self.clone().send(worker).child
        return _GeneralizedPointerTensor(pointers_dict).on(self)

    def send(self, worker, ptr_id=None):
        """
        Give the root of the chain held by self to worker
        self->alice->obj [worker] => self->worker->alice->obj

        Args:
            worker: the recipient of the transfer
            ptr_id: the id of the object when sent:
                x.send(bob, 1000)
                will result in bob having the tensor x with id 1000
        """

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
