# relative
from ..serde.serializable import serializable
from .document_store import StoreClientConfig


@serializable(canonical_name="MongoStoreClientConfig", version=1)
class MongoStoreClientConfig(StoreClientConfig):
    pass
