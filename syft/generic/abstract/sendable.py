from syft.generic.abstract.syft_serializable import SyftSerializable
from syft.generic.abstract.object import AbstractObject


class AbstractSendable(AbstractObject, SyftSerializable):
    """
    This layers functionality for sending objects between workers on top of AbstractObject.
    """

    def send(self, destination):
        return self.owner.send_obj(self, destination)
