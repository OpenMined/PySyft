# relative
from ...serde.serializable import serializable
from ...store.db.stash import ObjectStash
from ...types.blob_storage import BlobStorageEntry


@serializable(canonical_name="BlobStorageSQLStash", version=1)
class BlobStorageStash(ObjectStash[BlobStorageEntry]):
    pass
