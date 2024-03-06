# stdlib
import copy
from typing import Any
from typing import ClassVar
from typing import Type

# third party
from typing_extensions import Self

# relative
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject


class SyncableSyftObject(SyftObject):
    __canonical_name__ = "SyncableSyftObject"
    __version__ = SYFT_OBJECT_VERSION_1
    # mapping of private attributes and their mock values
    __private_sync_attrs__: ClassVar[dict[str, any]] = {}

    from_private_sync: bool = False

    @classmethod
    def _has_private_sync_attrs(cls: Type[Self]) -> bool:
        return len(cls.__private_sync_attrs__) > 0

    def create_shareable_sync_copy(self, mock: bool) -> Self:
        update = {}
        if mock:
            if self._has_private_sync_attrs():
                update |= copy.deepcopy(self.__private_sync_attrs__)
            update["from_private_sync"] = True
        return self.model_copy(update=update, deep=True)

    def get_sync_dependencies(self, api: Any = None) -> list[SyftObject]:
        return []
