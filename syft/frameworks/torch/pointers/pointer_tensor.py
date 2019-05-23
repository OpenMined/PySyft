import syft
import torch
from syft.frameworks.torch.tensors.interpreters import abstract
from syft.exceptions import CannotRequestObjectAttribute
from syft.frameworks.torch import pointers

from typing import List
from typing import Union
from typing import TYPE_CHECKING

# this if statement avoids circular imports between base.py and pointer.py
if TYPE_CHECKING:
    from syft.workers import BaseWorker


class PointerTensor(pointers.ObjectPointer, abstract.AbstractTensor):
    """A pointer to another tensor.

    A PointerTensor forwards all API calls to the remote.PointerTensor objects
    point to tensors (as their name implies). They exist to mimic the entire
    API of a normal tensor, but instead of computing a tensor function locally
    (such as addition, subtraction, etc.) they forward the computation to a
    remote machine as specified by self.location. Specifically, every
    PointerTensor has a tensor located somewhere that it points to (they should
    never exist by themselves). Note that PointerTensor objects can point to
    both torch.Tensor objects AND to other PointerTensor objects. Furthermore,
    the objects being pointed to can be on the same machine or (more commonly)
    on a different one. Note further that a PointerTensor does not know the
    nature how it sends messages to the tensor it points to (whether over
    socket, http, or some other protocol) as that functionality is abstracted
    in the BaseWorker object in self.location.

    Example:

     >>> import syft as sy
     >>> hook = sy.TorchHook()
     >>> bob = sy.VirtualWorker(id="bob")
     >>> x = sy.Tensor([1,2,3,4,5])
     >>> y = sy.Tensor([1,1,1,1,1])
     >>> x_ptr = x.send(bob) # returns a PointerTensor, sends tensor to Bob
     >>> y_ptr = y.send(bob) # returns a PointerTensor, sends tensor to Bob
     >>> # executes command on Bob's machine
     >>> z_ptr = x_ptr + y_ptr
    """

    def __init__(
        self,
        location: "BaseWorker" = None,
        id_at_location: Union[str, int] = None,
        owner: "BaseWorker" = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        shape: torch.Size = None,
        point_to_attr: str = None,
        tags: List[str] = None,
        description: str = None,
    ):
        super().__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=id,
            garbage_collect_data=garbage_collect_data,
            point_to_attr=point_to_attr,
            tags=tags,
            description=description,
        )
        self._shape = shape

    def get_shape(self):
        """Request information about the shape to the remote worker"""
        return self.owner.request_remote_tensor_shape(self)

    @property
    def shape(self):
        """ This method returns the shape of the data being pointed to.
        This shape information SHOULD be cached on self._shape, but
        occasionally this information may not be present. If this is the
        case, then it requests the shape information from the remote object
        directly (which is inefficient and should be avoided).
        """

        if self._shape is None:
            self._shape = self.get_shape()

        return self._shape

    @shape.setter
    def shape(self, new_shape):
        self._shape = new_shape

    def get(self, deregister_ptr: bool = True):
        """Requests the tensor/chain being pointed to, be serialized and return

        Since PointerTensor objects always point to a remote tensor (or chain
        of tensors, where a chain is simply a linked-list of tensors linked via
        their .child attributes), this method will request that the tensor/chain
        being pointed to be serialized and returned from this function.

        Note:
            This will typically mean that the remote object will be
            removed/destroyed. To just bring a copy back to the local worker,
            call .copy() before calling .get().


        Args:

            deregister_ptr (bool, optional): this determines whether to
                deregister this pointer from the pointer's owner during this
                method. This defaults to True because the main reason people use
                this method is to move the tensor from the remote machine to the
                local one, at which time the pointer has no use.

        Returns:
            An AbstractTensor object which is the tensor (or chain) that this
            object used to point to on a remote machine.

        TODO: add param get_copy which doesn't destroy remote if true.
        """

        if self.point_to_attr is not None:

            raise CannotRequestObjectAttribute(
                "You called .get() on a pointer to"
                " a tensor attribute. This is not yet"
                " supported. Call .clone().get() instead."
            )

        # if the pointer happens to be pointing to a local object,
        # just return that object (this is an edge case)
        if self.location == self.owner:
            tensor = self.owner.get_obj(self.id_at_location).child
        else:
            # get tensor from remote machine
            tensor = self.owner.request_obj(self.id_at_location, self.location)

        # Register the result
        assigned_id = self.id_at_location
        self.owner.register_obj(tensor, assigned_id)

        # Remove this pointer by default
        if deregister_ptr:
            self.owner.de_register_obj(self)

        # TODO: remove these 3 lines
        # The fact we have to check this means
        # something else is probably broken
        if tensor.is_wrapper:
            if isinstance(tensor.child, torch.Tensor):
                return tensor.child

        return tensor

    def attr(self, attr_name):
        attr_ptr = syft.PointerTensor(
            id=self.id,
            owner=self.owner,
            location=self.location,
            id_at_location=self.id_at_location,
            point_to_attr=self._create_attr_name_string(attr_name),
        ).wrap()
        self.__setattr__(attr_name, attr_ptr)
        return attr_ptr

    def share(self, *args, **kwargs):
        """
        Send a command to remote worker to additively share a tensor

        Returns:
            A pointer to an AdditiveSharingTensor
        """

        # Send the command
        command = ("share", self, args, kwargs)

        response = self.owner.send_command(self.location, command)

        return response
