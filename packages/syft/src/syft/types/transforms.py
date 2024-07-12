# stdlib
from collections.abc import Callable
from typing import Any

# third party
from pydantic import EmailStr
from typing_extensions import Self

# relative
from ..abstract_server import AbstractServer
from ..server.credentials import SyftVerifyKey
from ..service.context import AuthedServiceContext
from ..service.context import ServerServiceContext
from .server_url import ServerURL
from .syft_object import Context
from .syft_object import SyftBaseObject
from .syft_object_registry import SyftObjectRegistry
from .uid import UID


class NotNone:
    pass


class TransformContext(Context):
    output: dict[str, Any] | None = None
    server: AbstractServer | None = None
    credentials: SyftVerifyKey | None = None
    obj: Any | None = None

    @classmethod
    def from_context(cls, obj: Any, context: Context | None = None) -> Self:
        t_context = cls()
        t_context.obj = obj
        try:
            t_context.output = obj.to_dict(exclude_empty=True)
        except Exception:
            t_context.output = dict(obj)
        if context is None:
            return t_context
        if hasattr(context, "credentials"):
            t_context.credentials = context.credentials
        if hasattr(context, "server"):
            t_context.server = context.server
        return t_context

    def to_server_context(self) -> ServerServiceContext:
        if self.credentials:
            return AuthedServiceContext(
                server=self.server, credentials=self.credentials
            )
        if self.server:
            return ServerServiceContext(server=self.server)
        return Context()


def geteitherattr(
    _self: Any, output: dict, key: str, default: Any = NotNone
) -> Any | None:
    if key in output:
        return output[key]
    if default == NotNone:
        return getattr(_self, key)
    return getattr(_self, key, default)


def make_set_default(key: str, value: Any) -> Callable:
    def set_default(context: TransformContext) -> TransformContext:
        if context.output and not geteitherattr(context.obj, context.output, key, None):
            context.output[key] = value
        return context

    return set_default


def drop(list_keys: list[str]) -> Callable:
    def drop_keys(context: TransformContext) -> TransformContext:
        if context.output:
            for key in list_keys:
                if key in context.output:
                    del context.output[key]
        return context

    return drop_keys


def rename(old_key: str, new_key: str) -> Callable:
    def drop_keys(context: TransformContext) -> TransformContext:
        if context.output:
            context.output[new_key] = geteitherattr(
                context.obj, context.output, old_key
            )
            if old_key in context.output:
                del context.output[old_key]
        return context

    return drop_keys


def keep(list_keys: list[str]) -> Callable:
    def drop_keys(context: TransformContext) -> TransformContext:
        if context.output is None:
            return context

        for key in list_keys:
            if key not in context.output:
                context.output[key] = getattr(context.obj, key, None)

        keys = list(context.output.keys())

        for key in keys:
            if key not in list_keys and key in context.output:
                del context.output[key]

        return context

    return drop_keys


def convert_types(
    list_keys: list[str], types: type | list[type]
) -> Callable[[TransformContext], TransformContext]:
    if not isinstance(types, list):
        types = [types] * len(list_keys)

    if isinstance(types, list) and len(types) != len(list_keys):
        raise Exception("convert types lists must be the same length")

    def run_convert_types(context: TransformContext) -> TransformContext:
        if context.output:
            for key, _type in zip(list_keys, types):
                context.output[key] = _type(
                    geteitherattr(context.obj, context.output, key)
                )
        return context

    return run_convert_types


def generate_id(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context
    if "id" not in context.output or not isinstance(context.output["id"], UID):
        context.output["id"] = UID()
    return context


def generate_action_object_id(context: TransformContext) -> TransformContext:
    if context.output is None:
        return context
    if "action_object_id" not in context.output or not isinstance(
        context.output["action_object_id"], UID
    ):
        context.output["action_object_id"] = UID()
    return context


def validate_url(context: TransformContext) -> TransformContext:
    if context.output and context.output["url"] is not None:
        context.output["url"] = ServerURL.from_url(context.output["url"]).url_no_port
    return context


def validate_email(context: TransformContext) -> TransformContext:
    if context.output and context.output["email"] is not None:
        EmailStr._validate(context.output["email"])
    return context


def str_url_to_server_url(context: TransformContext) -> TransformContext:
    if context.output:
        url = context.output.get("url", None)
        if url is not None and isinstance(url, str):
            context.output["url"] = ServerURL.from_url(str)
    return context


def add_credentials_for_key(key: str) -> Callable:
    def add_credentials(context: TransformContext) -> TransformContext:
        if context.output is not None:
            context.output[key] = context.credentials
        return context

    return add_credentials


def add_server_uid_for_key(key: str) -> Callable:
    def add_server_uid(context: TransformContext) -> TransformContext:
        if context.output is not None and context.server is not None:
            context.output[key] = context.server.id
        return context

    return add_server_uid


def generate_transform_wrapper(
    klass_from: type, klass_to: type, transforms: list[Callable]
) -> Callable:
    def wrapper(
        self: klass_from,
        context: TransformContext | ServerServiceContext | None = None,
    ) -> klass_to:
        t_context = TransformContext.from_context(obj=self, context=context)
        for transform in transforms:
            t_context = transform(t_context)
        return klass_to(**t_context.output)

    return wrapper


def validate_klass_and_version(
    klass_from: type | str,
    klass_to: type | str,
    version_from: int | None = None,
    version_to: int | None = None,
) -> tuple[str, int | None, str, int | None]:
    if not isinstance(klass_from, type | str):
        raise NotImplementedError(
            "Arguments to `klass_from` should be either of `Type` or `str` type."
        )

    if isinstance(klass_from, str):
        klass_from_str = klass_from
    elif issubclass(klass_from, SyftBaseObject):
        klass_from_str = klass_from.__canonical_name__
        version_from = klass_from.__version__
    else:
        klass_from_str = klass_from.__name__
        version_from = None

    if not isinstance(klass_to, type | str):
        raise NotImplementedError(
            "Arguments to `klass_to` should be either of `Type` or `str` type."
        )

    if isinstance(klass_to, str):
        klass_to_str = klass_to
    elif issubclass(klass_to, SyftBaseObject):
        klass_to_str = klass_to.__canonical_name__
        version_to = klass_to.__version__
    else:
        klass_to_str = klass_to.__name__
        version_to = None

    return klass_from_str, version_from, klass_to_str, version_to


def transform_method(
    klass_from: type | str,
    klass_to: type | str,
    version_from: int | None = None,
    version_to: int | None = None,
) -> Callable:
    (
        klass_from_str,
        version_from,
        klass_to_str,
        version_to,
    ) = validate_klass_and_version(
        klass_from=klass_from,
        version_from=version_from,
        klass_to=klass_to,
        version_to=version_to,
    )

    def decorator(function: Callable) -> Callable:
        SyftObjectRegistry.add_transform(
            klass_from=klass_from_str,
            version_from=version_from,
            klass_to=klass_to_str,
            version_to=version_to,
            method=function,
        )

        return function

    return decorator


def transform(
    klass_from: type | str,
    klass_to: type | str,
    version_from: int | None = None,
    version_to: int | None = None,
) -> Callable:
    (
        klass_from_str,
        version_from,
        klass_to_str,
        version_to,
    ) = validate_klass_and_version(
        klass_from=klass_from,
        version_from=version_from,
        klass_to=klass_to,
        version_to=version_to,
    )

    def decorator(function: Callable) -> Callable:
        transforms = function()

        wrapper = generate_transform_wrapper(
            klass_from=klass_from, klass_to=klass_to, transforms=transforms
        )

        SyftObjectRegistry.add_transform(
            klass_from=klass_from_str,
            version_from=version_from,
            klass_to=klass_to_str,
            version_to=version_to,
            method=wrapper,
        )

        return function

    return decorator
