# relative
from ..abstract_server import ServerType
from ..serde.serializable import serializable
from .server import Server


@serializable()
class Gateway(Server):
    def post_init(self) -> None:
        self.server_type = ServerType.GATEWAY
        super().post_init()
