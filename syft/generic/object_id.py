from syft.workers.abstract import AbstractWorker


class ObjectId:
    """ ObjectIds are used to uniquely identify PySyft objects.
    """

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, ObjectId):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    @staticmethod
    def simplify(worker: "AbstractWorker", id: "ObjectId") -> tuple:
        return (id.value,)

    @staticmethod
    def detail(worker: "AbstractWorker", simplified_id: tuple) -> "ObjectId":
        (value,) = simplified_id
        return ObjectId(value)
