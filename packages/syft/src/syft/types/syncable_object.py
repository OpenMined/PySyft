# stdlib
import copy
from typing import Any
from typing import ClassVar

# third party
from typing_extensions import Self

# relative
from ..service.context import AuthedServiceContext
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .uid import UID


class SyncableSyftObject(SyftObject):
    __canonical_name__ = "SyncableSyftObject"
    __version__ = SYFT_OBJECT_VERSION_1
    # mapping of private attributes and their mock values
    __private_sync_attr_mocks__: ClassVar[dict[str, any]] = {}

    @classmethod
    def _has_private_sync_attrs(cls: type[Self]) -> bool:
        return len(cls.__private_sync_attr_mocks__) > 0

    def create_shareable_sync_copy(self, mock: bool) -> Self:
        update: dict[str, Any] = {}
        if mock and self._has_private_sync_attrs():
            update |= copy.deepcopy(self.__private_sync_attr_mocks__)
        return self.model_copy(update=update, deep=True)

    def get_sync_dependencies(self, context: AuthedServiceContext) -> list[UID]:
        return []
