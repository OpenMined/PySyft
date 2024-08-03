# stdlib
import uuid


class ObjectId:
    def __init__(self, id=None) -> None:
        super().__init__()
        if id is None:
            self._id = uuid.uuid1()
        else:
            self._id = uuid.UUID(id)

    def __eq__(self, other):
        return isinstance(other, ObjectId) and other._id == self._id

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._id)

    def __repr__(self) -> str:
        return f"ObjectId({self._id})"

    def __str__(self) -> str:
        return str(self._id)
