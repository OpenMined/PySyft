# stdlib
import uuid


class ObjectId(object):
    def __init__(self, id=None):
        super(ObjectId, self).__init__()
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

    def __repr__(self):
        return "ObjectId({0})".format(self._id)

    def __str__(self):
        return str(self._id)
