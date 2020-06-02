from collections import defaultdict

from syft.generic.abstract.object import AbstractObject
from syft.serde.syft_serializable import SyftSerializable


class AbstractSendable(AbstractObject, SyftSerializable):
    """
    This layers functionality for sending objects between workers on top of AbstractObject.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # { worker_id -> set(object_ids) }
        self.remote_refs = defaultdict(set)

    def send(self, destination):
        return self.owner.send_obj(self, destination)

    def _before_create_pointer(self, location, id_at_location, *args, **kwargs):
        self.remote_refs[location] = id_at_location

    @staticmethod
    def _before_create_pointer(sendable, location, id_at_location, *args, **kwargs):
        sendable.remote_refs[location] = id_at_location
