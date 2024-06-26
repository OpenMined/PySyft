# stdlib

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ..service import AbstractService


@serializable()
class EnclaveService(AbstractService):
    store: DocumentStore

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
