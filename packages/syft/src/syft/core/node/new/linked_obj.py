# stdlib
from typing import Any
from typing import Optional
from typing import Type
from typing import Union

# third party
from typing_extensions import Self

# relative
from .context import NodeServiceContext
from .response import SyftError
from .response import SyftSuccess
from .serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .uid import UID


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

    def resolve_with_context(self, context: NodeServiceContext) -> Any:
        return context.node.get_service(self.service_type).resolve_link(
            context=context, linked_obj=self
        )

    def update_with_context(
        self, context: NodeServiceContext, obj: Any
    ) -> Union[SyftSuccess, SyftError]:
        result = context.node.get_service(self.service_type).stash.update(obj)
        if result.is_ok():
            return result

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
            node_uid = getattr(obj, "node_uid", None)
            if node_uid is None:
                raise Exception(f"{cls} Requires an object UID")

        return LinkedObject(
            node_uid=node_uid,
            service_type=service_type,
            object_type=type(obj),
            object_uid=object_uid,
        )

    @classmethod
    def with_context(
        cls,
        obj: SyftObject,
        context: NodeServiceContext,
        object_uid: Optional[UID] = None,
        service_type: Optional[Type[Any]] = None,
    ) -> Self:
        if service_type is None:
            # relative
            from .service import TYPE_TO_SERVICE

            service_type = TYPE_TO_SERVICE[type(obj)]

        if object_uid is None and hasattr(obj, "id"):
            object_uid = getattr(obj, "id", None)
        if object_uid is None:
            raise Exception(f"{cls} Requires an object UID")

        node_uid = context.node.id

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
