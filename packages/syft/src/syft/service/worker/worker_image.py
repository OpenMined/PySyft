# stdlib
from typing import Optional

# relative
from ...custom_worker.config import PrebuiltWorkerConfig
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
    __repr_attrs__ = [
        "image_identifier",
        "image_hash",
        "created_at",
        "built_at",
        "config",
    ]

    id: UID
    config: WorkerConfig
    created_by: SyftVerifyKey
    created_at: DateTime = DateTime.now()
    image_identifier: Optional[SyftWorkerImageIdentifier] = None
    image_hash: Optional[str] = None
    built_at: Optional[DateTime] = None

    @property
    def is_built(self) -> bool:
        """Returns True if the image has been built or is prebuilt."""

        return self.built_at is not None or self.is_prebuilt

    @property
    def is_prebuilt(self) -> bool:
        return isinstance(self.config, PrebuiltWorkerConfig)

    @property
    def built_image_tag(self) -> Optional[str]:
        """Returns the full name of the image if it has been built."""

        if self.is_built and self.image_identifier:
            return self.image_identifier.full_name_with_tag
        return None
