# future
from __future__ import annotations

# stdlib
from types import NoneType
from typing import Any

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..context import AuthedServiceContext


@serializable()
class CustomEndpointActionObject(SyftObject):
    __canonical_name__ = "CustomEndpointActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    endpoint_id: UID
    context: AuthedServiceContext | None = None

    def add_context(self, context: AuthedServiceContext) -> CustomEndpointActionObject:
        self.context = context
        return self

    def __call__(self, *args, **kwargs) -> Any:
        if self.context is None:
            raise Exception("No context provided to CustomEndpointActionObject")
        endpoint_service = self.context.node.get_service("apiservice")
        return endpoint_service.execute_endpoint_by_id(
            context=self.context, endpoint_uid=self.endpoint_id, *args, **kwargs
        )
