# stdlib
from typing import Any

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..service.context import AuthedServiceContext
from ..service.context import ChangeContext
from ..service.context import NodeServiceContext
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.syft_object import SYFT_OBJECT_VERSION_2
from ..types.syft_object import SyftObject
from ..types.uid import UID


@serializable()
class LinkedObject(SyftObject):
    __canonical_name__ = "LinkedObject"
    __version__ = SYFT_OBJECT_VERSION_2

    node_uid: UID
    service_type: type[Any]
    object_type: type[SyftObject]
    object_uid: UID

    __exclude_sync_diff_attrs__ = ["node_uid"]

    def __str__(self) -> str:
        resolved_obj_type = (
            type(self.resolve) if self.object_type is None else self.object_type
        )
        return f"{resolved_obj_type.__name__}: {self.object_uid} @ Node {self.node_uid}"

    @property
    def resolve(self) -> SyftObject:
        # relative
        from ..client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
        if api is None:
            raise ValueError(f"api is None. You must login to {self.node_uid}")

        return api.services.notifications.resolve_object(self)

    def resolve_with_context(self, context: NodeServiceContext) -> Any:
        if context.node is None:
            raise ValueError(f"context {context}'s node is None")
        return context.node.get_service(self.service_type).resolve_link(
            context=context, linked_obj=self
        )

    def update_with_context(
        self, context: NodeServiceContext | ChangeContext | Any, obj: Any
    ) -> SyftSuccess | SyftError:
        if isinstance(context, AuthedServiceContext):
            credentials = context.credentials
        elif isinstance(context, ChangeContext):
            credentials = context.approving_user_credentials
        else:
            return SyftError(message="wrong context passed")
        if context.node is None:
            return SyftError(message=f"context {context}'s node is None")
        service = context.node.get_service(self.service_type)
        if hasattr(service, "stash"):
            result = service.stash.update(credentials, obj)
        else:
            return SyftError(message=f"service {service} does not have a stash")
        return result

    @classmethod
    def from_obj(
        cls,
        obj: SyftObject | type[SyftObject],
        service_type: type[Any] | None = None,
        node_uid: UID | None = None,
    ) -> Self:
        if service_type is None:
            # relative
            from ..service.action.action_object import ActionObject
            from ..service.action.action_service import ActionService
            from ..service.service import TYPE_TO_SERVICE

            if isinstance(obj, ActionObject):
                service_type = ActionService
            else:
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
            syft_client_verify_key=obj.syft_client_verify_key,
        )

    @classmethod
    def with_context(
        cls,
        obj: SyftObject,
        context: NodeServiceContext,
        object_uid: UID | None = None,
        service_type: type[Any] | None = None,
    ) -> Self:
        if service_type is None:
            # relative
            from ..service.service import TYPE_TO_SERVICE

            service_type = TYPE_TO_SERVICE[type(obj)]

        if object_uid is None and hasattr(obj, "id"):
            object_uid = getattr(obj, "id", None)
        if object_uid is None:
            raise Exception(f"{cls} Requires an object UID")

        if context.node is None:
            raise ValueError(f"context {context}'s node is None")
        node_uid = context.node.id

        return LinkedObject(
            node_uid=node_uid,
            service_type=service_type,
            object_type=type(obj),
            object_uid=object_uid,
        )

    @classmethod
    def from_uid(
        cls,
        object_uid: UID,
        object_type: type[SyftObject],
        service_type: type[Any],
        node_uid: UID,
    ) -> Self:
        return cls(
            node_uid=node_uid,
            service_type=service_type,
            object_type=object_type,
            object_uid=object_uid,
        )
