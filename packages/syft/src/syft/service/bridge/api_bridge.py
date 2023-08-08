# stdlib
from inspect import Parameter
from inspect import Signature
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

# third party
from openapi3 import OpenAPI
from openapi3.paths import Operation
from openapi3.schemas import TYPE_LOOKUP
from pydantic import create_model
import requests
from typing_extensions import Self

# relative
from ...serde.recursive_primitives import recursive_serde_register
from ...serde.serializable import serializable
from ...types.grid_url import GridURL
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...util.constants import DEFAULT_TIMEOUT


def consume_api(url: str, base_url: Optional[str] = None) -> OpenAPI:
    grid_url = GridURL.from_url(url)
    d = requests.get(grid_url, timeout=DEFAULT_TIMEOUT)
    x = d.json()
    if not base_url:
        base_url = GridURL.from_url(grid_url.base_url)
    else:
        base_url = GridURL.from_url(url)
    x["servers"] = [{"url": str(base_url)}]
    api = OpenAPI(x, use_session=True)
    return api, base_url


@serializable(without=["type_"])
class SerdeType(SyftObject):
    __canonical_name__ = "SerdeType"
    __version__ = SYFT_OBJECT_VERSION_1

    fqn: str
    attributes: List[str]
    types: List[Type]
    type_: Optional[Type] = None

    @property
    def _parts(self) -> List[str]:
        return self.fqn.split(".")

    @property
    def _class_name(self) -> str:
        return self._parts.pop()

    @property
    def _module(self) -> str:
        parts = self._parts
        _ = parts.pop()
        return ".".join(parts)

    def create_type(self) -> type:
        if not self.type_:
            args = [self._class_name]
            kwargs = {}
            for name, type_ in zip(self.attributes, self.types):
                kwargs[name] = (type_, None)
            serde_type = create_model(*args, **kwargs)
            serde_type.__module__ = self._module
            self.type_ = serde_type
        return self.type_

    def register_type(self) -> None:
        type_ = self.create_type()
        recursive_serde_register(type_)


@serializable()
class APIBridge(SyftObject):
    # version
    __canonical_name__ = "APIBridge"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["title"]
    __attr_unique__ = ["title"]

    title: str
    version: str
    openapi_spec_url: GridURL
    base_url: GridURL
    openapi: OpenAPI

    def from_url(url: str, base_url: Optional[str] = None) -> Self:
        grid_url = GridURL.from_url(url)
        api, base_url = consume_api(grid_url, base_url)
        return APIBridge(
            title=api.info.title,
            version=api.info.version,
            openapi_spec_url=grid_url,
            base_url=base_url,
            openapi=api,
        )

    @property
    def api_name(self) -> str:
        # return a variable name which is valid python
        return re.sub(
            "^[0-9]+", "", re.sub("[^0-9a-zA-Z]+", "_", self.title.lower()).strip("_")
        )

    def get_serde_types(self) -> List[SerdeType]:
        serde_types = []
        for _, schema in self.openapi.components.schemas.items():
            schema_type = schema.get_type()
            fqn = schema_type.__module__ + "." + schema_type.__name__
            attributes = []
            types = []
            for attribute in schema.properties:
                param_type = schema.properties[attribute].type
                python_type = TYPE_LOOKUP[param_type]
                attributes.append(attribute)
                types.append(python_type)
            st = SerdeType(fqn=fqn, attributes=attributes, types=types)
            serde_types.append(st)
        return serde_types

    def register_serde_types(self) -> None:
        for serde_type in self.get_serde_types():
            serde_type.register_type()

    @classmethod
    def op_to_signature(cls, method: Operation) -> Signature:
        parameters = []
        for param in method.parameters:
            name = param.name
            required = param.required
            param_type = param.schema.type
            python_type = TYPE_LOOKUP[param_type]
            if not required:
                python_type = Optional[python_type]
            param = Parameter(
                name=name, kind=Parameter.KEYWORD_ONLY, annotation=python_type
            )
            parameters.append(param)

        if method.requestBody:
            if "application/json" in method.requestBody.content:
                schema = method.requestBody.content["application/json"].schema._proxy
                title = schema.title.lower()
                param = Parameter(
                    name=title, kind=Parameter.KEYWORD_ONLY, annotation=schema
                )
                parameters.append(param)
        return Signature(parameters=parameters)

    @classmethod
    def kwargs_to_parameters(
        cls, method: Operation, kwargs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        parameters = {}
        data = {}

        for param in method.parameters:
            name = param.name
            if name in kwargs:
                parameters[name] = kwargs[name]

        if method.requestBody:
            if "application/json" in method.requestBody.content:
                schema = method.requestBody.content["application/json"].schema._proxy
                title = schema.title.lower()
                if title in kwargs:
                    data = dict(kwargs[title])

        return parameters, data
