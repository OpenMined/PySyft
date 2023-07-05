# relative
from ..serde.serializable import serializable
from .client import SyftClient


@serializable()
class DomainClient(SyftClient):
    def __repr__(self) -> str:
        return f"<DomainClient: {self.name}>"
