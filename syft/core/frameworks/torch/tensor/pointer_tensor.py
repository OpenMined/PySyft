import logging
import json
from .... import utils
import syft as sy
from syft.core.frameworks.torch import torch_utils
from syft.core.frameworks.torch.tensor import _SyftTensor


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

    def register_pointer(self):
        worker = self.owner
        if(isinstance(self.location, int)):
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

        # Perform the un-wrap
        response, _ = torch_utils.get_child_command(response)

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