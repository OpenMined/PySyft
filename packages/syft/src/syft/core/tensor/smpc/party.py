# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.serializable import bind_protobuf
from syft.proto.core.tensor.smpc.party_pb2 import Party as Party_PB


@bind_protobuf
class Party(Serializable):
    __slots__ = ("url", "email", "passwd", "port")

    def __init__(self, url: str, email: str, passwd: str, port: int) -> None:
        self.url = url
        self.email = email
        self.passwd = passwd
        self.port = port

    def _object2proto(self) -> Party_PB:
        return Party_PB(url=self.url, port=self.port)

    @staticmethod
    def _proto2object(proto: Party_PB) -> "Party":
        # TODO: If on the same machine use docker-host - if not real address
        # How to distinguish? (if 127.0.0.1 and localhost we consider using docker-host?)
        res = Party(url="docker-host", email="", passwd="", port=proto.port)
        return res

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Party_PB
