# stdlib

# stdlib
from collections.abc import Callable

# relative
from ...custom_worker.config import PrebuiltWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...types.datetime import DateTime
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.uid import UID
from .image_identifier import SyftWorkerImageIdentifier


@serializable()
class SyftWorkerImageV1(SyftObject):
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
    image_identifier: SyftWorkerImageIdentifier | None = None
    image_hash: str | None = None
    built_at: DateTime | None = None

    @property
    def is_prebuilt(self) -> bool:
        return isinstance(self.config, PrebuiltWorkerConfig)


@serializable()
class SyftWorkerImage(SyftObject):
    __canonical_name__ = "SyftWorkerImage"
    __version__ = SYFT_OBJECT_VERSION_2

    __attr_unique__ = ["config_hash"]
    __attr_searchable__ = [
        "config",
        "image_hash",
        "created_by",
        "config_hash",
    ]

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
    image_identifier: SyftWorkerImageIdentifier | None = None
    image_hash: str | None = None
    built_at: DateTime | None = None

    @property
    def config_hash(self) -> str:
        return self.config.hash()

    @property
    def is_built(self) -> bool:
        """Returns True if the image has been built or is prebuilt."""

        return self.built_at is not None or self.is_prebuilt

    @property
    def is_prebuilt(self) -> bool:
        return isinstance(self.config, PrebuiltWorkerConfig)

    @property
    def built_image_tag(self) -> str | None:
        """Returns the full name of the image if it has been built."""

        if self.is_built and self.image_identifier:
            return self.image_identifier.full_name_with_tag
        return None


@migrate(SyftWorkerImageV1, SyftWorkerImage)
def migrate_syft_worker_image_v1_to_v2() -> list[Callable]:
    return []  # no migrations needed at data level, only unique and searchable attributes changed
