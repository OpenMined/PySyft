from syft.serde.syft_serializable import SyftSerializable
from syft.generic.abstract.object import AbstractObject


class AbstractSendable(AbstractObject, SyftSerializable):
    """
    This layers functionality for sending objects between workers on top of AbstractObject.
    """

    def send(self, destination):
        return self.owner.send_obj(self, destination)

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

    def mid_get(self):
        """This method calls .get() on a child pointer and correctly registers the results"""

        child_id = self.id
        tensor = self.get()
        tensor.id = child_id
        self.owner.register_obj(tensor)
