# stdlib
from typing import Any
from typing import Optional
from typing import Type

# third party
from typing_extensions import Self

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID


@serializable(recursive_serde=True)
class LinkedObject(SyftObject):
    __canonical_name__ = "LinkedObject"
    __version__ = SYFT_OBJECT_VERSION_1

    node_uid: UID
    service_type: Type[Any]
    object_type: Type[SyftObject]
    object_uid: UID

    def __str__(self) -> str:
        return f"<{self.object_type}: {self.object_uid}@<Node: {self.node_uid}>"

    @property
    def resolve(self) -> SyftObject:
        # relative
        from .api import APIRegistry

        api = APIRegistry.api_for(node_uid=self.node_uid)
        return api.services.messages.resolve_object(self)

    @classmethod
    def from_obj(
        cls,
        obj: SyftObject,
        service_type: Optional[Type[Any]] = None,
        node_uid: Optional[UID] = None,
    ) -> Self:
        if service_type is None:
            # relative
            from .service import TYPE_TO_SERVICE

            service_type = TYPE_TO_SERVICE[type(obj)]

        object_uid = getattr(obj, "id", None)
        if object_uid is None:
            raise Exception(f"{cls} Requires an object UID")

        if node_uid is None:
            node_uid = getattr(obj, "node_id", None)
            if node_uid is None:
                raise Exception(f"{cls} Requires an object UID")

        return LinkedObject(
            node_uid=node_uid,
            service_type=service_type,
            object_type=type(obj),
            object_uid=object_uid,
        )

    @staticmethod
    def from_uid(
        object_uid: UID,
        object_type: Type[SyftObject],
        service_type: Type[Any],
        node_uid: UID,
    ) -> Self:
        return LinkedObject(
            node_uid=node_uid,
            service_type=service_type,
            object_type=object_type,
            object_uid=object_uid,
        )
