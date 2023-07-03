# relative
from ..serde.serializable import serializable
from .domain import Domain


@serializable(without=["queue_manager"])
class Enclave(Domain):
    pass
