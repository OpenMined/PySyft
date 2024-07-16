# relative
from ..abstract_server import ServerType
from ..serde.serializable import serializable
from .server import Server


@serializable(canonical_name="Gateway", version=1)
class Gateway(Server):
    def post_init(self) -> None:
        self.server_type = ServerType.GATEWAY
        super().post_init()
