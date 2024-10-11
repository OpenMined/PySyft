# stdlib
import logging
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import Union
from typing import get_args

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
from ..types.syft_object import SyftObjectVersioned
from ..types.uid import UID

T = TypeVar("T", bound=SyftObject)
logger = logging.getLogger(__name__)


@serializable()
class LinkedObject(SyftObjectVersioned, Generic[T]):
    __canonical_name__ = "LinkedObject"
    __version__ = SYFT_OBJECT_VERSION_1

    server_uid: UID
    service_type: type[Any]
    object_type: type[T]
    object_uid: UID

    _resolve_cache: SyftObject | None = None

    __exclude_sync_diff_attrs__ = ["server_uid"]

    def __str__(self) -> str:
        resolved_obj_type = (
            type(self.resolve) if self.object_type is None else self.object_type
        )
        return f"{resolved_obj_type.__name__}: {self.object_uid} @ Server {self.server_uid}"

    @classmethod
    def get_generic_type(cls: type[Self]) -> type[T]:
        args = cls.__pydantic_generic_metadata__["args"]
        if len(args) != 1:
            raise ValueError(
                "Cannot infer LinkedObject type, generic argument not provided"
            )
        return args[0]  # type: ignore

    @property
    def resolve(self) -> SyftObject:
        return self._resolve()

    def _resolve(self, load_cached: bool = False) -> SyftObject:
        api = None
        if load_cached and self._resolve_cache is not None:
            return self._resolve_cache
        try:
            # relative
            api = self.get_api()  # raises
            resolve: SyftObject = api.services.notifications.resolve_object(self)
            self._resolve_cache = resolve
            return resolve
        except Exception as e:
            logger.error(">>> Failed to resolve object", type(api), e)
            raise e

    def resolve_dynamic(
        self, context: ServerServiceContext | None, load_cached: bool = False
    ) -> SyftObject:
        if context is not None:
            return self.resolve_with_context(context, load_cached).unwrap()
        else:
            return self._resolve(load_cached)

    @as_result(SyftException)
    def resolve_with_context(
        self, context: ServerServiceContext, load_cached: bool = False
    ) -> Any:
        if load_cached and self._resolve_cache is not None:
            return self._resolve_cache
        if context.server is None:
            raise ValueError(f"context {context}'s server is None")
        res = (
            context.server.get_service(self.service_type)
            .resolve_link(context=context, linked_obj=self)
            .unwrap()
        )
        self._resolve_cache = res
        return res

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
            result = service.stash.update(credentials, obj).unwrap()
        else:
            raise SyftException(
                public_message=f"service {service} does not have a stash"
            )
        return result

    @classmethod
    def from_obj(
        cls,
        obj: T | type[T],
        service_type: type[Any] | None = None,
        server_uid: UID | None = None,
    ) -> "LinkedObject[T]":  # type: ignore
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

        return LinkedObject[type(obj)](  # type: ignore
            server_uid=server_uid,
            service_type=service_type,
            object_type=type(obj),
            object_uid=object_uid,
            syft_client_verify_key=obj.syft_client_verify_key,
        )

    @classmethod
    def with_context(
        cls,
        obj: T,
        context: ServerServiceContext,
        object_uid: UID | None = None,
        service_type: type[Any] | None = None,
    ) -> "LinkedObject[T]":
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

        return LinkedObject[type(obj)](  # type: ignore
            server_uid=server_uid,
            service_type=service_type,
            object_type=type(obj),
            object_uid=object_uid,
        )

    @classmethod
    def from_uid(
        cls,
        object_uid: UID,
        object_type: type[T],
        service_type: type[Any],
        server_uid: UID,
    ) -> "LinkedObject[T]":
        return cls[object_type](  # type: ignore
            server_uid=server_uid,
            service_type=service_type,
            object_type=object_type,
            object_uid=object_uid,
        )


def _unwrap_optional(type_: Any) -> Any:
    try:
        if type_ | None == type_:
            args = get_args(type_)
            return Union[tuple(arg for arg in args if arg != type(None))]  # noqa
        return type_
    except Exception:
        return type_


def _annotation_issubclass(type_: Any, cls: type) -> bool:
    try:
        return issubclass(type_, cls)
    except Exception:
        return False


def _resolve_syftobject_forward_refs(raise_errors: bool = False) -> None:
    # relative
    from ..types.syft_object_registry import SyftObjectRegistry

    type_names = [
        t.__name__ for t in SyftObjectRegistry.__type_to_canonical_name__.keys()
    ]
    if len(type_names) != len(set(type_names)):
        raise ValueError(
            "Duplicate names in SyftObjectRegistry, cannot resolve forward references"
        )

    types_namespace = {
        k.__name__: k for k in SyftObjectRegistry.__type_to_canonical_name__.keys()
    }
    syft_objects = [v for v in types_namespace.values() if issubclass(v, SyftObject)]

    for so in syft_objects:
        so.model_rebuild(raise_errors=raise_errors, _types_namespace=types_namespace)


def find_unannotated_linked_objects() -> None:
    # Utility method to find LinkedObjects that are not annotated with a generic type

    # relative
    from ..types.syft_object_registry import SyftObjectRegistry

    # Need to resolve forward references to find LinkedObjects
    _resolve_syftobject_forward_refs()

    annotated = []
    unannotated = []

    for cls in SyftObjectRegistry.__type_to_canonical_name__.keys():
        if not issubclass(cls, SyftObject):
            continue

        for name, field in cls.model_fields.items():
            type_ = _unwrap_optional(field.annotation)
            if _annotation_issubclass(type_, LinkedObject):
                try:
                    type_.get_generic_type()
                    annotated.append((cls, name))
                except Exception:
                    unannotated.append((cls, name))

    print("Annotated LinkedObjects:")
    for cls, name in annotated:
        print(f"{cls.__name__}.{name}")

    print("\n\nUnannotated LinkedObjects:")
    for cls, name in unannotated:
        print(f"{cls.__name__}.{name}")
