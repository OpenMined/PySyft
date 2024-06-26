# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from .enclave import EnclaveInstance


@instrument
@serializable()
class EnclaveInstanceStash(BaseUIDStoreStash):
    object_type = EnclaveInstance
    settings: PartitionSettings = PartitionSettings(
        name=EnclaveInstance.__canonical_name__, object_type=EnclaveInstance
    )
