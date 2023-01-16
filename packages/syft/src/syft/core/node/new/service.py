# stdlib
import inspect
from typing import Dict
from typing import List
from typing import Optional

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .signature import Signature


class AbstractNode:
    id: UID


class AbstractService:
    node: AbstractNode
    node_uid: UID


@serializable(recursive_serde=True)
class ServiceConfig(SyftObject):
    public_path: str
    private_path: str
    public_name: str
    method_name: str
    doc_string: Optional[str]
    signature: Signature
    permissions: List


class ServiceConfigRegistry:
    __service_config_registry__: Dict[str, ServiceConfig] = {}
    __public_to_private_path_map__: Dict[str, str] = {}

    @classmethod
    def register(cls, config: ServiceConfig) -> None:
        if not cls.path_exists(config.public_path):
            cls.__service_config_registry__[config.public_path] = config
            cls.__public_to_private_path_map__[config.public_path] = config.private_path

    @classmethod
    def private_path_for(cls, public_path: str) -> str:
        return cls.__public_to_private_path_map__[public_path]

    @classmethod
    def get_registered_configs(cls) -> Dict[str, ServiceConfig]:
        return cls.__service_config_registry__

    @classmethod
    def path_exists(cls, path: str):
        return path in cls.__service_config_registry__


def service_method(
    name: Optional[str] = None,
    path: Optional[str] = None,
):
    def wrapper(func):
        func_name = func.__name__
        class_name = func.__qualname__.split(".")[-2]
        _path = class_name + "." + func_name

        config = ServiceConfig(
            public_path=_path if path is None else path,
            private_path=_path,
            public_name=("public_" + func_name) if name is None else name,
            method_name=func_name,
            doc_string=func.__doc__,
            signature=inspect.signature(func),
            permissions=["Guest"],
        )
        ServiceConfigRegistry.register(config)

        def _decorator(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        return _decorator

    return wrapper
