# stdlib
import json
from typing import Any
from typing import Optional
from typing import Union

# relative
from ...client.connection import NodeConnection
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ..response import SyftError
from .vpn import BaseVPNClient
from .vpn import VPNRoutes


class HeadScaleAuthToken(SyftObject):
    __canonical_name__ = "HeadScaleAuthToken"
    __version__ = SYFT_OBJECT_VERSION_1

    namespace: str
    key: str


class HeadScaleRoutes(VPNRoutes):
    GENERATE_KEY = "/commands/generate_key"
    LIST_NODES = "/commands/list_nodes"


@serializable()
class HeadScaleClient(BaseVPNClient):
    connection: NodeConnection
    api_key: str
    _token: Optional[str]

    def __init__(self, connection: NodeConnection, api_key: str) -> None:
        self.connection = connection
        self.api_key = api_key

    @property
    def route(self) -> Any:
        return self.connection.route

    def generate_token(
        self,
    ) -> Union[HeadScaleAuthToken, SyftError]:
        result = self.connection.send_command(
            path=self.route.GENERATE_KEY.value,
            api_key=self.api_key,
        )

        if result.is_err():
            return SyftError(message=result.err())

        command_report = result.ok()

        result = self.connection.resolve_report(
            api_key=self.api_key, report=command_report
        )

        if result.is_err():
            return SyftError(message=result.err())

        command_result = result.ok()

        if command_result.error:
            return SyftError(message=result.error)

        result = json.loads(command_result.report)

        return HeadScaleAuthToken(**result)
