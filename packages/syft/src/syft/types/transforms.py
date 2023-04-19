# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from pydantic import EmailStr
from typing_extensions import Self

# relative
from ..abstract_node import AbstractNode
from ..node.credentials import SyftVerifyKey
from ..service.context import AuthedServiceContext
from ..service.context import NodeServiceContext
from .grid_url import GridURL
from .syft_object import Context
from .syft_object import SyftBaseObject
from .syft_object import SyftObjectRegistry
from .uid import UID


class NotNone:
    pass


class TransformContext(Context):
    output: Optional[Dict[str, Any]]
    node: Optional[AbstractNode]
    credentials: Optional[SyftVerifyKey]
    obj: Optional[Any]

    @staticmethod
    def from_context(obj: Any, context: Optional[Context] = None) -> Self:
        t_context = TransformContext()
        t_context.obj = obj
        t_context.output = dict(obj)
        if hasattr(context, "credentials"):
            t_context.credentials = context.credentials
        if hasattr(context, "node"):
            t_context.node = context.node
        return t_context

    def to_node_context(self) -> NodeServiceContext:
        if self.credentials:
            return AuthedServiceContext(node=self.node, credentials=self.credentials)
        if self.node:
            return NodeServiceContext(node=self.node)
        return Context()


def geteitherattr(
    _self: Any, output: Dict, key: str, default: Any = NotNone
) -> Optional[Any]:
    if key in output:
        return output[key]
    if default == NotNone:
        return getattr(_self, key)
    return getattr(_self, key, default)


def make_set_default(key: str, value: Any) -> Callable:
    def set_default(context: TransformContext) -> TransformContext:
        if not geteitherattr(context.obj, context.output, key, None):
            context.output[key] = value
        return context

    return set_default


def drop(list_keys: List[str]) -> Callable:
    def drop_keys(context: TransformContext) -> TransformContext:
        for key in list_keys:
            if key in context.output:
                del context.output[key]
        return context

    return drop_keys


def rename(old_key: str, new_key: str) -> Callable:
    def drop_keys(context: TransformContext) -> TransformContext:
        context.output[new_key] = geteitherattr(context.obj, context.output, old_key)
        if old_key in context.output:
            del context.output[old_key]
        return context

    return drop_keys


def keep(list_keys: List[str]) -> Callable:
    def drop_keys(context: TransformContext) -> TransformContext:
        for key in list_keys:
            if key not in context.output:
                context.output[key] = getattr(context.obj, key, None)

        keys = list(context.output.keys())

        for key in keys:
            if key not in list_keys and key in context.output:
                del context.output[key]

        return context

    return drop_keys


def convert_types(list_keys: List[str], types: Union[type, List[type]]) -> Callable:
    if not isinstance(types, list):
        types = [types] * len(list_keys)

    if isinstance(types, list) and len(types) != len(list_keys):
        raise Exception("convert types lists must be the same length")

    def run_convert_types(context: TransformContext) -> TransformContext:
        for key, _type in zip(list_keys, types):
            context.output[key] = _type(geteitherattr(context.obj, context.output, key))
        return context

    return run_convert_types


def generate_id(context: TransformContext) -> TransformContext:
    if "id" not in context.output or not isinstance(context.output["id"], UID):
        context.output["id"] = UID()
    return context


def validate_url(context: TransformContext) -> TransformContext:
    if context.output["url"] is not None:
        context.output["url"] = GridURL.from_url(context.output["url"]).url_no_port
    return context


def validate_email(context: TransformContext) -> TransformContext:
    if context.output["email"] is not None:
        context.output["email"] = EmailStr(context.output["email"])
        EmailStr.validate(context.output["email"])
    return context


def add_credentials_for_key(key: str) -> Callable:
    def add_credentials(context: TransformContext) -> TransformContext:
        context.output[key] = context.credentials
        return context

    return add_credentials


def add_node_uid_for_key(key: str) -> Callable:
    def add_node_uid(context: TransformContext) -> TransformContext:
        context.output[key] = context.node.id
        return context

    return add_node_uid


def generate_transform_wrapper(
    klass_from: type, klass_to: type, transforms: List[Callable]
) -> Callable:
    def wrapper(
        self: klass_from,
        context: Optional[Union[TransformContext, NodeServiceContext]] = None,
    ) -> klass_to:
        t_context = TransformContext.from_context(obj=self, context=context)
        for transform in transforms:
            t_context = transform(t_context)
        return klass_to(**t_context.output)

    return wrapper


def validate_klass_and_version(
    klass_from: Union[Type, str],
    klass_to: Union[Type, str],
    version_from: Optional[int] = None,
    version_to: Optional[int] = None,
):
    if not isinstance(klass_from, (type, str)):
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

    if not isinstance(klass_to, (type, str)):
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
    klass_from: Union[Type, str],
    klass_to: Union[Type, str],
    version_from: Optional[int] = None,
    version_to: Optional[int] = None,
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

    def decorator(function: Callable):
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
    klass_from: Union[type, str],
    klass_to: Union[type, str],
    version_from: Optional[int] = None,
    version_to: Optional[int] = None,
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

    def decorator(function: Callable):
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
