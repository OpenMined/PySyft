# future
from __future__ import annotations

# stdlib
from enum import Enum
from enum import auto
from typing import Any

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..context import AuthedServiceContext


class EXECUTION_MODE(Enum):
    CALL = auto()
    MOCK = auto()
    PRIVATE = auto()


@serializable()
class CustomEndpointActionObject(SyftObject):
    __canonical_name__ = "CustomEndpointActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    endpoint_id: UID
    context: AuthedServiceContext | None = None

    def add_context(self, context: AuthedServiceContext) -> CustomEndpointActionObject:
        self.context = context
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__call_function(  #  type: ignore[misc]
            *args,
            **kwargs,
            call_mode=EXECUTION_MODE.CALL,
        )

    def mock(self, *args: Any, **kwargs: Any) -> Any:
        return self.__call_function(  #  type: ignore[misc]
            *args,
            **kwargs,
            call_mode=EXECUTION_MODE.MOCK,
        )

    def private(self, *args: Any, **kwargs: Any) -> Any:
        return self.__call_function(  #  type: ignore[misc]
            *args,
            **kwargs,
            call_mode=EXECUTION_MODE.PRIVATE,
        )

    def __call_function(
        self, call_mode: EXECUTION_MODE, *args: Any, **kwargs: Any
    ) -> Any:
        self.context = self.__check_context()
        endpoint_service = self.context.node.get_service("apiservice")

        if call_mode == EXECUTION_MODE.MOCK:
            __endpoint_mode = endpoint_service.execute_server_side_endpoint_mock_by_id
        elif call_mode == EXECUTION_MODE.PRIVATE:
            __endpoint_mode = (
                endpoint_service.execute_service_side_endpoint_private_by_id
            )
        else:
            __endpoint_mode = endpoint_service.execute_server_side_endpoint_by_id

        return __endpoint_mode(
            *args, context=self.context, endpoint_uid=self.endpoint_id, **kwargs
        )

    def __check_context(self) -> AuthedServiceContext:
        if self.context is None:
            raise Exception("No context provided to CustomEndpointActionObject")
        return self.context
