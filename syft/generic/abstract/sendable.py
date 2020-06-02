from syft.serde.syft_serializable import SyftSerializable
from syft.generic.abstract.object import AbstractObject


class AbstractSendable(AbstractObject, SyftSerializable):
    """This layers functionality for sending objects between workers on top of AbstractObject.
    """

    def send(self, destination):
        """Send the current object to the worker `destination`.

        Args:
            destination (BaseWorker): The worker where the current object is sent.
 
        Returns:
            An  object of type ObjectPointer pointing to the sent object.
        """

        ptr = self.owner.send(self, destination)

        return ptr
