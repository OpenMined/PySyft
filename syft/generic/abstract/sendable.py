import syft as sy

from syft.generic.abstract.object import AbstractObject
from syft.serde.syft_serializable import SyftSerializable
from syft.workers.abstract import AbstractWorker


class AbstractSendable(AbstractObject, SyftSerializable):
    """
    This layers functionality for sending objects between workers on top of AbstractObject.
    """

    def serialize(self):  # check serde.py to see how to provide compression schemes
        """Serializes the tensor on which it's called.

        This is the high level convenience function for serializing torch
        tensors. It includes three steps, Simplify, Serialize, and Compress as
        described in serde.py.
        By default serde is compressing using LZ4

        Returns:
            The serialized form of the tensor.
            For example:
                x = torch.Tensor([1,2,3,4,5])
                x.serialize() # returns a serialized object
        """
        return sy.serde.serialize(self)

    def ser(self, *args, **kwargs):
        return self.serialize(*args, **kwargs)

    def create_pointer(
        self,
        location: AbstractWorker = None,
        id_at_location: (str or int) = None,
        owner: AbstractWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
        shape=None,
        **kwargs,
    ):
        pass

    def send(
        self,
        *location,
        inplace: bool = False,
        user: object = None,
        local_autograd: bool = False,
        requires_grad: bool = False,
        preinitialize_grad: bool = False,
        no_wrap: bool = False,
        garbage_collect_data: bool = True,
    ):
        pass

    def send_(self, *location, **kwargs):
        pass

    def get(self):
        """Just a pass through. This is most commonly used when calling .get() on a
        Syft tensor which has a child which is a pointer, an additive shared tensor,
        a multi-pointer, etc."""
        class_attributes = self.get_class_attributes()
        return type(self)(
            **class_attributes,
            owner=self.owner,
            tags=self.tags,
            description=self.description,
            id=self.id,
        ).on(self.child.get())

    def get_(self, *args, **kwargs):
        pass

    def mid_get(self):
        """This method calls .get() on a child pointer and correctly registers the results"""
        child_id = self.id
        tensor = self.get()
        tensor.id = child_id
        self.owner.register_obj(tensor)

    def remote_get(self):
        pass

    def allow(self, user=None) -> bool:
        pass

    def move(self, location: AbstractWorker):
        pass

    def move_(self, location: AbstractWorker):
        pass

    def remote_send(self, location):
        pass
