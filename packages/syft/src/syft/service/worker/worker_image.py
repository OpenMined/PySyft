# stdlib
from typing import Optional

# relative
from ...custom_worker.config import WorkerConfig
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from .image_identifier import SyftWorkerImageIdentifier


@serializable()
class SyftWorkerImage(SyftObject):
    __canonical_name__ = "SyftWorkerImage"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["config"]
    __attr_searchable__ = ["config", "image_hash", "created_by"]
    __repr_attrs__ = ["image_identifier", "image_hash", "created_at", "built_at"]

    id: UID
    config: WorkerConfig
    image_identifier: Optional[SyftWorkerImageIdentifier]
    image_hash: Optional[str]
    created_at: DateTime = DateTime.now()
    created_by: SyftVerifyKey
    built_at: Optional[DateTime]
