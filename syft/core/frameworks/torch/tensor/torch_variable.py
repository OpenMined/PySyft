import random
import json
import torch
import syft as sy
from syft.core.frameworks.torch.tensor import _TorchObject
from .... import utils
from syft.core.frameworks.torch import torch_utils


class _TorchVariable(_TorchObject):

    def send(self, worker, new_id=None, new_data_id=None, new_grad_id=None, new_grad_data_id=None):
        """
        Give the root of the chain held by self to worker
        self->alice->obj [worker] => self->worker->alice->obj
        Because there are Variable involved, there are actually 4 chains involved,
        the variable chain, variable.data, variable.grad, variable.grad.data
        """

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

    def ser(self, private, as_dict=True):
        key = '__' + type(self).__name__ + '__'

        tensor_msg = {
            'type': str(self.__class__).split("'")[1],
            'torch_type': 'syft.' + type(self).__name__,
            'data': self.data.ser(private),
            'child': self.child.ser(private),
            'requires_grad': self.requires_grad
        }
        if self.grad is not None:
            tensor_msg['grad'] = self.grad.ser(private)
        elif self.data.dim() > 0:
            # Create a .grad just if there is some data in the tensor (to avoid recursion errors)
            self.init_grad_()
            tensor_msg['grad'] = self.grad.ser(private)

        if as_dict:
            return {key: tensor_msg}
        else:
            return json.dumps({key: tensor_msg}) + "\n"

    @staticmethod
    def deser(msg_obj, worker, acquire):
        obj_type, msg_obj = torch_utils.extract_type_and_obj(msg_obj)
        var_syft_obj = sy._SyftTensor.deser_routing(msg_obj['child'], worker, acquire)

        if var_syft_obj.parent is not None and var_syft_obj.child is not None:
            return var_syft_obj.parent

        # Deser the var.data
        var_data_type, var_data_tensor = torch_utils.extract_type_and_obj(msg_obj['data'])
        if torch_utils.is_tensor(var_data_type):
            var_data = torch.guard['syft.' + var_data_type].deser(msg_obj['data'], worker, acquire)
            worker.hook.local_worker.de_register(var_data)
        else:
            raise TypeError('Data is not a tensor:', var_data_type)

        variable = sy.Variable(var_data, requires_grad=msg_obj['requires_grad'])

        # Deser the var.grad
        if 'grad' in msg_obj:
            var_grad_type, var_grad_tensor = torch_utils.extract_type_and_obj(msg_obj['grad'])
            var_grad = torch.guard['syft.' + var_grad_type].deser(msg_obj['grad'], worker, acquire)
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
