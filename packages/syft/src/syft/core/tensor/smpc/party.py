# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.serializable import bind_protobuf
from syft.proto.core.tensor.smpc.party_pb2 import Party as Party_PB


@bind_protobuf
class Party(Serializable):
    __slots__ = ("url", "port")
    _DOCKER_HOST: str = "http://docker-host"

    def __init__(self, url: str, port: int) -> None:
        # TODO: This is not used -- it is hardcoded to docker
        # GM: Probably for real life scenario we would need to change this (when we serialize)
        # for a more real life scenario
        self.url = url

        self.port = port

    def _object2proto(self) -> Party_PB:
        return Party_PB(url=self.url, port=self.port)

    @staticmethod
    def _proto2object(proto: Party_PB) -> "Party":
        # TODO: If on the same machine use docker-host - if not real address
        # How to distinguish? (if 127.0.0.1 and localhost we consider using docker-host?)
        res = Party(url=Party._DOCKER_HOST, port=proto.port)
        return res

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Party_PB

    def __hash__(self) -> int:
        res_str = f"{Party._DOCKER_HOST}:{self.port}"
        return hash((Party._DOCKER_HOST, self.port))

    def __eq__(self, other: "Party") -> bool:
        return self.port == other.port
