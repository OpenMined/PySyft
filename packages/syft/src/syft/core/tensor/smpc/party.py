# stdlib
# stdlib
from typing import Any

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ....proto.core.tensor.party_pb2 import Party as Party_PB
from ...common.serde.serializable import serializable


@serializable()
class Party:
    __slots__ = ("url", "port")
    # _DOCKER_HOST: str = "http://docker-host"

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
        res = Party(url=proto.url, port=proto.port)
        return res

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Party_PB

    def __hash__(self) -> int:
        # TODO: Rasswanth, George, Trask this takes into consideration a hashing based on url and port
        # that we login only once
        # Is that sufficient?
        # res_str = f"{self.url}:{self.port}"
        return hash((self.url, self.port))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Party):
            return False

        return self.port == other.port and self.url == other.url
