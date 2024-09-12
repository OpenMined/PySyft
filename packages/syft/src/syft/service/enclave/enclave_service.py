# stdlib

# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ..service import AbstractService


@serializable(canonical_name="EnclaveService", version=1)
class EnclaveService(AbstractService):
    def __init__(self, store: DBManager) -> None:
        pass
