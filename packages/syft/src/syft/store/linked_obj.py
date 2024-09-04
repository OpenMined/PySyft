# stdlib
import logging
from typing import Any

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..service.context import AuthedServiceContext
from ..service.context import ChangeContext
from ..service.context import ServerServiceContext
from ..service.response import SyftSuccess
from ..types.errors import SyftException
from ..types.result import as_result
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID

logger = logging.getLogger(__name__)


@serializable()
class LinkedObject(SyftObject):
    __canonical_name__ = "LinkedObject"
    __version__ = SYFT_OBJECT_VERSION_1

    server_uid: UID
    service_type: type[Any]
    object_type: type[SyftObject]
    object_uid: UID

    _resolve_cache: SyftObject | None = None

    __exclude_sync_diff_attrs__ = ["server_uid"]

    def __str__(self) -> str:
        resolved_obj_type = (
            type(self.resolve) if self.object_type is None else self.object_type
        )
        return f"{resolved_obj_type.__name__}: {self.object_uid} @ Server {self.server_uid}"

    @property
    def resolve(self) -> SyftObject:
        api = None
        try:
            # relative
            api = self.get_api()  # raises
            resolve: SyftObject = api.services.notifications.resolve_object(self)
            self._resolve_cache = resolve
            return resolve
        except Exception as e:
            logger.error(">>> Failed to resolve object", type(api), e)
            raise e

    @as_result(SyftException)
    def resolve_with_context(self, context: ServerServiceContext) -> Any:
        if context.server is None:
            raise ValueError(f"context {context}'s server is None")
        return (
            context.server.get_service(self.service_type)
            .resolve_link(context=context, linked_obj=self)
            .unwrap()
        )

    def update_with_context(
        self, context: ServerServiceContext | ChangeContext | Any, obj: Any
    ) -> SyftSuccess:
        if isinstance(context, AuthedServiceContext):
            credentials = context.credentials
        elif isinstance(context, ChangeContext):
            credentials = context.approving_user_credentials
        else:
            raise SyftException(public_message="wrong context passed")
        if context.server is None:
            raise SyftException(public_message=f"context {context}'s server is None")
        service = context.server.get_service(self.service_type)
        if hasattr(service, "stash"):
            result = service.stash.update(credentials, obj)
        else:
            raise SyftException(
                public_message=f"service {service} does not have a stash"
            )
        return result

    @classmethod
    def from_obj(
        cls,
        obj: SyftObject | type[SyftObject],
        service_type: type[Any] | None = None,
        server_uid: UID | None = None,
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

        if server_uid is None:
            server_uid = getattr(obj, "server_uid", None)
            if server_uid is None:
                raise Exception(f"{cls} Requires an object UID")

        return LinkedObject(
            server_uid=server_uid,
            service_type=service_type,
            object_type=type(obj),
            object_uid=object_uid,
            syft_client_verify_key=obj.syft_client_verify_key,
        )

    @classmethod
    def with_context(
        cls,
        obj: SyftObject,
        context: ServerServiceContext,
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

        if context.server is None:
            raise ValueError(f"context {context}'s server is None")
        server_uid = context.server.id

        return LinkedObject(
            server_uid=server_uid,
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
        server_uid: UID,
    ) -> Self:
        return cls(
            server_uid=server_uid,
            service_type=service_type,
            object_type=object_type,
            object_uid=object_uid,
        )
