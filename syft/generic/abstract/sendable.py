from syft.serde.syft_serializable import SyftSerializable
from syft.generic.abstract.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.pointers.object_pointer import ObjectPointer


class AbstractSendable(AbstractObject, SyftSerializable):
    """This layers functionality for sending objects between workers on top of AbstractObject.
    """

    def send(self, destination: BaseWorker) -> ObjectPointer:
        """Send the current object to the worker `destination`.

        Args:
            destination: The worker where the current object is sent.
 
        Returns:
            A pointer to the send object.
        """

        ptr = self.owner.send(self, destination)

        return ptr
