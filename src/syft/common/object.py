from .id import UID
from ..proto import ProtoObjectWithId


class AbstractObjectWithID:
    ""


class ObjectWithId(AbstractObjectWithID):
    def __init__(self, id: UID = None):

        if id is None:
            id = UID()

        self.id = id

    def serialize(self):
        return ProtoObjectWithId(id=self.id.serialize())

    @staticmethod
    def deserialize(proto_obj: ProtoObjectWithId) -> AbstractObjectWithID:
        return ObjectWithId(id=UID.deserialize(proto_obj.id))
