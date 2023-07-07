# stdlib
from collections import defaultdict
from copy import deepcopy
import inspect
from inspect import Parameter
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

# third party
from result import Ok
from result import OkErr

# relative
from ..abstract_node import AbstractNode
from ..node.credentials import SyftVerifyKey
from ..serde.lib_permissions import CMPCRUDPermission
from ..serde.lib_permissions import CMPPermission
from ..serde.lib_service_registry import CMPBase
from ..serde.lib_service_registry import CMPClass
from ..serde.lib_service_registry import CMPFunction
from ..serde.lib_service_registry import action_execute_registry_libs
from ..serde.serializable import serializable
from ..serde.signature import Signature
from ..serde.signature import signature_remove_context
from ..serde.signature import signature_remove_self
from ..store.linked_obj import LinkedObject
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftObject
from ..types.syft_object import attach_attribute_to_syft_object
from ..types.uid import UID
from .context import AuthedServiceContext
from .context import ChangeContext
from .response import SyftError
from .user.user_roles import DATA_OWNER_ROLE_LEVEL
from .user.user_roles import ServiceRole

TYPE_TO_SERVICE = {}
SERVICE_TO_TYPES = defaultdict(set)


class AbstractService:
    node: AbstractNode
    node_uid: UID

    def resolve_link(
        self, context: AuthedServiceContext, linked_obj: LinkedObject
    ) -> Union[Any, SyftError]:
        if isinstance(context, AuthedServiceContext):
            credentials = context.credentials
        elif isinstance(context, ChangeContext):
            credentials = context.approving_user_credentials
        else:
            return SyftError(message="wrong context passed")

        obj = self.stash.get_by_uid(credentials, uid=linked_obj.object_uid)
        if isinstance(obj, OkErr) and obj.is_ok():
            obj = obj.ok()
        if hasattr(obj, "node_uid"):
            obj.node_uid = context.node.id
        if not isinstance(obj, OkErr):
            obj = Ok(obj)
        return obj


@serializable()
class BaseConfig(SyftBaseObject):
    public_path: str
    private_path: str
    public_name: str
    method_name: str
    doc_string: Optional[str]
    signature: Optional[Signature]
    is_from_lib: bool = False


@serializable()
class ServiceConfig(BaseConfig):
    permissions: List
    roles: List[ServiceRole]

    def has_permission(self, user_service_role: ServiceRole):
        return user_service_role in self.roles


@serializable()
class LibConfig(BaseConfig):
    permissions: Set[CMPPermission]

    def has_permission(self, credentials: SyftVerifyKey):
        # TODO: implement user level permissions
        for p in self.permissions:
            if p.permission_string == CMPCRUDPermission.ALL_EXECUTE.name:
                return True
            if p.permission_string == CMPCRUDPermission.NONE_EXECUTE.name:
                return False
        return False


class ServiceConfigRegistry:
    __service_config_registry__: Dict[str, ServiceConfig] = {}
    # __public_to_private_path_map__: Dict[str, str] = {}

    @classmethod
    def register(cls, config: ServiceConfig) -> None:
        if not cls.path_exists(config.public_path):
            cls.__service_config_registry__[config.public_path] = config
            # cls.__public_to_private_path_map__[config.public_path] = config.private_path

    @classmethod
    def get_registered_configs(cls) -> Dict[str, ServiceConfig]:
        return cls.__service_config_registry__

    @classmethod
    def path_exists(cls, path: str):
        return path in cls.__service_config_registry__


class LibConfigRegistry:
    __service_config_registry__: Dict[str, ServiceConfig] = {}

    @classmethod
    def register(cls, config: ServiceConfig) -> None:
        if not cls.path_exists(config.public_path):
            cls.__service_config_registry__[config.public_path] = config

    @classmethod
    def get_registered_configs(cls) -> Dict[str, ServiceConfig]:
        return cls.__service_config_registry__

    @classmethod
    def path_exists(cls, path: str):
        return path in cls.__service_config_registry__


class UserLibConfigRegistry:
    def __init__(self, service_config_registry: Dict[str, LibConfig]):
        self.__service_config_registry__: Dict[str, LibConfig] = service_config_registry

    @classmethod
    def from_user(cls, credentials: SyftVerifyKey):
        return cls(
            {
                k: lib_config
                for k, lib_config in LibConfigRegistry.get_registered_configs().items()
                if lib_config.has_permission(credentials)
            }
        )

    def __contains__(self, path: str):
        return path in self.__service_config_registry__

    def private_path_for(self, public_path: str) -> str:
        return self.__service_config_registry__[public_path].private_path

    def get_registered_configs(self) -> Dict[str, LibConfig]:
        return self.__service_config_registry__


class UserServiceConfigRegistry:
    def __init__(self, service_config_registry: Dict[str, ServiceConfig]):
        self.__service_config_registry__: Dict[
            str, ServiceConfig
        ] = service_config_registry

    @classmethod
    def from_role(cls, user_service_role: ServiceRole):
        return cls(
            {
                k: service_config
                for k, service_config in ServiceConfigRegistry.get_registered_configs().items()
                if service_config.has_permission(user_service_role)
            }
        )

    def __contains__(self, path: str):
        return path in self.__service_config_registry__

    def private_path_for(self, public_path: str) -> str:
        return self.__service_config_registry__[public_path].private_path

    def get_registered_configs(self) -> Dict[str, ServiceConfig]:
        return self.__service_config_registry__


def register_lib_obj(lib_obj: CMPBase):
    signature = lib_obj.signature
    path = lib_obj.absolute_path
    func_name = lib_obj.name

    if signature is not None:
        if path != "numpy.source":
            lib_config = LibConfig(
                public_path=str(path),
                private_path=str(path),
                public_name=str(func_name),
                method_name=str(func_name),
                doc_string=str(lib_obj.__doc__),
                signature=signature,
                permissions=set([lib_obj.permissions]),
                is_from_lib=True,
            )

            LibConfigRegistry.register(lib_config)


# hacky, prevent circular imports
for lib_obj in action_execute_registry_libs.flatten():
    # # for functions
    # func_name = func.__name__
    # # for classes
    # func_name = path.split(".")[-1]
    if isinstance(lib_obj, CMPFunction) or isinstance(lib_obj, CMPClass):
        register_lib_obj(lib_obj)


def deconstruct_param(param: inspect.Parameter) -> Dict[str, Any]:
    # Gets the init signature form pydantic object
    param_type = param.annotation
    if not hasattr(param_type, "__signature__"):
        raise Exception(
            f"Type {param_type} needs __signature__. Or code changed to support backup init"
        )
    signature = param_type.__signature__
    sub_mapping = {}
    for k, v in signature.parameters.items():
        sub_mapping[k] = v
    return sub_mapping


def types_for_autosplat(signature: Signature, autosplat: List[str]) -> Dict[str, type]:
    autosplat_types = {}
    for k, v in signature.parameters.items():
        if k in autosplat:
            autosplat_types[k] = v.annotation
    return autosplat_types


def reconstruct_args_kwargs(
    signature: Signature,
    autosplat: List[str],
    args: Tuple[Any, ...],
    kwargs: Dict[Any, str],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    autosplat_types = types_for_autosplat(signature=signature, autosplat=autosplat)

    autosplat_objs = {}
    for autosplat_key, autosplat_type in autosplat_types.items():
        init_kwargs = {}
        keys = autosplat_type.__fields__.keys()
        for key in keys:
            if key in kwargs:
                init_kwargs[key] = kwargs.pop(key)
        autosplat_objs[autosplat_key] = autosplat_type(**init_kwargs)

    final_kwargs = {}
    for param_key, param in signature.parameters.items():
        if param_key in kwargs:
            final_kwargs[param_key] = kwargs[param_key]
        elif param_key in autosplat_objs:
            final_kwargs[param_key] = autosplat_objs[param_key]
        elif not isinstance(param.default, type(Parameter.empty)):
            final_kwargs[param_key] = param.default
        else:
            raise Exception(f"Missing {param_key} not in kwargs.")
    return (args, final_kwargs)


def expand_signature(signature: Signature, autosplat: List[str]) -> Signature:
    new_mapping = {}
    for k, v in signature.parameters.items():
        if k in autosplat:
            sub_mapping = deconstruct_param(v)
            for s, t in sub_mapping.items():
                new_t_kwargs = {
                    "annotation": t.annotation,
                    "name": t.name,
                    "default": t.default,
                    "kind": Parameter.POSITIONAL_OR_KEYWORD,
                }
                new_t = Parameter(**new_t_kwargs)
                new_mapping[s] = new_t
        else:
            new_mapping[k] = v

    # Reorder the parameter based on if they have default value or not
    new_params = sorted(
        new_mapping.values(),
        key=lambda param: param.default is param.empty,
        reverse=True,
    )

    return Signature(
        **{
            "parameters": new_params,
            "return_annotation": signature.return_annotation,
        }
    )


def service_method(
    name: Optional[str] = None,
    path: Optional[str] = None,
    roles: Optional[List[ServiceRole]] = None,
    autosplat: Optional[List[str]] = None,
):
    if roles is None or len(roles) == 0:
        # TODO: this is dangerous, we probably want to be more conservative
        roles = DATA_OWNER_ROLE_LEVEL

    def wrapper(func):
        func_name = func.__name__
        class_name = func.__qualname__.split(".")[-2]
        _path = class_name + "." + func_name
        signature = inspect.signature(func)
        signature = signature_remove_self(signature)
        signature = signature_remove_context(signature)

        input_signature = deepcopy(signature)

        def _decorator(self, *args, **kwargs):
            if autosplat is not None and len(autosplat) > 0:
                args, kwargs = reconstruct_args_kwargs(
                    signature=input_signature,
                    autosplat=autosplat,
                    args=args,
                    kwargs=kwargs,
                )
            result = func(self, *args, **kwargs)
            context = kwargs.get("context", None)
            context = args[0] if context is None else context
            attrs_to_attach = {
                "syft_node_location": context.node.id,
                "syft_client_verify_key": context.credentials,
            }
            return attach_attribute_to_syft_object(
                result=result, attr_dict=attrs_to_attach
            )

        if autosplat is not None and len(autosplat) > 0:
            signature = expand_signature(signature=input_signature, autosplat=autosplat)

        config = ServiceConfig(
            public_path=_path if path is None else path,
            private_path=_path,
            public_name=("public_" + func_name) if name is None else name,
            method_name=func_name,
            doc_string=func.__doc__,
            signature=signature,
            roles=roles,
            permissions=["Guest"],
        )
        ServiceConfigRegistry.register(config)

        _decorator.__name__ = func.__name__
        _decorator.__qualname__ = func.__qualname__
        return _decorator

    return wrapper


class SyftServiceRegistry:
    __service_registry__: Dict[str, Callable] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "__canonical_name__") and hasattr(cls, "__version__"):
            mapping_string = f"{cls.__canonical_name__}_{cls.__version__}"
            cls.__object_version_registry__[mapping_string] = cls

    @classmethod
    def versioned_class(cls, name: str, version: int) -> Optional[Type["SyftObject"]]:
        mapping_string = f"{name}_{version}"
        if mapping_string not in cls.__object_version_registry__:
            return None
        return cls.__object_version_registry__[mapping_string]

    @classmethod
    def add_transform(
        cls,
        klass_from: str,
        version_from: int,
        klass_to: str,
        version_to: int,
        method: Callable,
    ) -> None:
        mapping_string = f"{klass_from}_{version_from}_x_{klass_to}_{version_to}"
        cls.__object_transform_registry__[mapping_string] = method

    @classmethod
    def get_transform(
        cls, type_from: Type["SyftObject"], type_to: Type["SyftObject"]
    ) -> Callable:
        klass_from = type_from.__canonical_name__
        version_from = type_from.__version__
        klass_to = type_to.__canonical_name__
        version_to = type_to.__version__
        mapping_string = f"{klass_from}_{version_from}_x_{klass_to}_{version_to}"
        return cls.__object_transform_registry__[mapping_string]
