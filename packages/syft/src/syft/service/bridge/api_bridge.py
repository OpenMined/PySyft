# stdlib
from inspect import Parameter
from inspect import Signature
import re
from typing import Optional

# third party
from openapi3 import OpenAPI
from openapi3.paths import Operation
from openapi3.schemas import TYPE_LOOKUP
import requests
from typing_extensions import Self

# relative
from ...serde.serializable import serializable
from ...types.grid_url import GridURL
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject


def consume_api(url: str, base_url: Optional[str] = None) -> OpenAPI:
    grid_url = GridURL.from_url(url)
    d = requests.get(grid_url)
    x = d.json()
    if not base_url:
        base_url = GridURL.from_url(grid_url.base_url)
    else:
        base_url = GridURL.from_url(url)
    x["servers"] = [{"url": str(base_url)}]
    api = OpenAPI(x)
    return api, base_url


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
