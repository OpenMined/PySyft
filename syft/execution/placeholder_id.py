import syft as sy
from syft.workers.abstract import AbstractWorker

from syft.generic.abstract.syft_serializable import SyftSerializable
from syft_proto.execution.v1.placeholder_id_pb2 import PlaceholderId as PlaceholderIdPB


class PlaceholderId(SyftSerializable):
    """
    PlaceholderIds are used to identify which Placeholder tensors should be used
    as the inputs and outputs of Actions.
    """

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, PlaceholderId):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    @staticmethod
    def simplify(worker: "AbstractWorker", id: "PlaceholderId") -> tuple:
        return (id.value,)

    @staticmethod
    def detail(worker: "AbstractWorker", simplified_id: tuple) -> "PlaceholderId":
        (value,) = simplified_id
        return PlaceholderId(value)

    @staticmethod
    def bufferize(worker: "AbstractWorker", id: "PlaceholderId") -> tuple:
        protobuf_id = PlaceholderIdPB()
        sy.serde.protobuf.proto.set_protobuf_id(protobuf_id.id, id.value)

        return protobuf_id

    @staticmethod
    def unbufferize(worker: "AbstractWorker", protobuf_id: tuple) -> "PlaceholderId":
        value = sy.serde.protobuf.proto.get_protobuf_id(protobuf_id.id)

        return PlaceholderId(value)

    @staticmethod
    def get_protobuf_schema() -> PlaceholderIdPB:
        return PlaceholderIdPB
