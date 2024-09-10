# future
from __future__ import annotations

# stdlib
from collections.abc import Callable
from enum import Enum
from enum import auto
from typing import Any

# relative
from ...serde.serializable import serializable
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ...types.uid import UID
from ..context import AuthedServiceContext


class EXECUTION_MODE(Enum):
    CALL = auto()
    MOCK = auto()
    PRIVATE = auto()


@serializable()
class CustomEndpointActionObjectV1(SyftObject):
    __canonical_name__ = "CustomEndpointActionObject"
    __version__ = SYFT_OBJECT_VERSION_1

    endpoint_id: UID
    context: AuthedServiceContext | None = None


@serializable()
class CustomEndpointActionObject(SyftObject):
    __canonical_name__ = "CustomEndpointActionObject"
    __version__ = SYFT_OBJECT_VERSION_2

    endpoint_id: UID
    context: AuthedServiceContext | None = None
    log_id: UID | None = None

    def add_context(
        self, context: AuthedServiceContext, log_id: UID | None = None
    ) -> CustomEndpointActionObject:
        self.context = context
        self.log_id = log_id
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
        if call_mode == EXECUTION_MODE.MOCK:
            __endpoint_mode = (
                self.context.server.services.api.execute_server_side_endpoint_mock_by_id
            )
        elif call_mode == EXECUTION_MODE.PRIVATE:
            __endpoint_mode = self.context.server.services.api.execute_service_side_endpoint_private_by_id
        else:
            __endpoint_mode = (
                self.context.server.services.api.execute_server_side_endpoint_by_id
            )

        return __endpoint_mode(  #  type: ignore[misc]
            *args,
            context=self.context,
            endpoint_uid=self.endpoint_id,
            log_id=self.log_id,
            **kwargs,
        ).unwrap()

    def __check_context(self) -> AuthedServiceContext:
        if self.context is None:
            raise Exception("No context provided to CustomEndpointActionObject")
        return self.context


@migrate(CustomEndpointActionObjectV1, CustomEndpointActionObject)
def migrate_custom_endpoint_v1_to_v2() -> list[Callable]:
    return [make_set_default("log_id", None)]


@migrate(CustomEndpointActionObject, CustomEndpointActionObjectV1)
def migrate_custom_endpoint_v2_to_v1() -> list[Callable]:
    return [drop(["log_id"])]
