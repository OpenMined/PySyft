# relative
from ..serde.serializable import serializable
from .client import SyftClient


@serializable()
class EnclaveClient(SyftClient):
    def __repr__(self) -> str:
        return f"<EnclaveClient: {self.name}>"
