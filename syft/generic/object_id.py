from syft.workers.abstract import AbstractWorker

from syft_proto.types.syft.v1.id_pb2 import Id as IdPB


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

    @staticmethod
    def bufferize(worker: "AbstractWorker", id: "ObjectId") -> tuple:
        protobuf_id = IdPB()
        if isinstance(id.value, int):
            protobuf_id.id_int = id.value
        elif isinstance(id.value, str):
            protobuf_id.id_str = id.value

        return protobuf_id

    @staticmethod
    def unbufferize(worker: "AbstractWorker", protobuf_id: tuple) -> "ObjectId":
        value = getattr(protobuf_id, protobuf_id.WhichOneof("id"))

        return ObjectId(value)
