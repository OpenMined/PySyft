# stdlib
import json
from typing import Optional
from typing import Union

# relative
from ...client.connection import NodeConnection
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..response import SyftError
from .vpn import BaseVPNClient
from .vpn import VPNRoutes


@serializable()
class HeadscaleAuthToken(SyftObject):
    __canonical_name__ = "HeadscaleAuthToken"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    namespace: str
    key: str


class HeadscaleRoutes(VPNRoutes):
    GENERATE_KEY = "/commands/generate_key"
    LIST_NODES = "/commands/list_nodes"


@serializable()
class HeadscaleClient(BaseVPNClient):
    connection: NodeConnection
    api_key: str

    def __init__(self, connection: NodeConnection, api_key: str) -> None:
        self.connection = connection
        self.api_key = api_key

    def generate_token(
        self,
    ) -> Union[HeadscaleAuthToken, SyftError]:
        result = self.connection.send_command(
            path=self.connection.routes.GENERATE_KEY.value,
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
        return HeadscaleAuthToken(
            key=result["key"],
            namespace=result["namespace"],
        )
