
from ..service import AbstractService
from ...store.document_store import DocumentStore
from .code_interface import CodeInterfaceStash

class CodeInterfaceService(AbstractService):
    store: DocumentStore
    stash: CodeInterfaceStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = CodeInterfaceStash(store=store)

