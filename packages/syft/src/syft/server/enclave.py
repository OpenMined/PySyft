# relative
from ..abstract_server import ServerType
from ..serde.serializable import serializable
from .server import Server


@serializable()
class Enclave(Server):
    def post_init(self) -> None:
        self.server_type = ServerType.ENCLAVE
        super().post_init()
