# stdlib
from enum import Enum
import json
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

# third party
from pydantic import validator
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from result import Err
from result import Result
from urllib3 import Retry

# relative
from ...client.client import upgrade_tls
from ...client.connection import NodeConnection
from ...serde.serializable import serializable
from ...types.grid_url import GridURL
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...util.util import verify_tls
from ..response import SyftError

DEFAULT_TIMEOUT = 5  # in seconds


class VPNRoutes(Enum):
    pass


class CommandStatus(Enum):
    RUNNING = "running"


class CommandReport(SyftObject):
    key: str
    result_url: str
    status: CommandStatus

    @validator("status", pre=True)
    def validate_status(cls, v: Union[str, CommandStatus]) -> CommandStatus:
        if type(v) is not CommandStatus:
            value = CommandStatus[str(v)]
        else:
            value = v

        return value


class CommandResult(PartialSyftObject):
    report: str
    key: str
    start_time: float
    end_time: float
    process_time: float
    returncode: int
    error: Optional[str]


class HeadScaleAuthToken(SyftObject):
    __canonical_name__ = "HeadScaleAuthToken"
    __version__ = SYFT_OBJECT_VERSION_1

    namespace: str
    key: str


class HeadScaleRoutes(VPNRoutes):
    GENERATE_KEY = "/commands/generate_key"
    LIST_NODES = "/commands/list_nodes"


class BaseVPNClient:
    pass


class VPNClientConnection(NodeConnection):
    __canonical_name__ = "VPNClientConnection"
    __version__ = SYFT_OBJECT_VERSION_1

    url: GridURL
    session_cache: Optional[Session]
    routes: Type[VPNRoutes]

    def __init__(self, url: Union[GridURL, str]) -> None:
        url = GridURL.from_url(url)
        super().__init__(url=url)

    @property
    def session(self) -> Session:
        if self.session_cache is None:
            session = requests.Session()
            retry = Retry(total=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self.session_cache = session
        return self.session_cache

    def _make_get(
        self,
        path: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        url = self.url.with_path(path)
        response = self.session.get(
            str(url),
            verify=verify_tls(),
            proxies={},
            params=params,
            headers=headers,
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response.content

    def _make_post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        url = self.url.with_path(path)
        response = self.session.post(
            str(url),
            verify=verify_tls(),
            json=json,
            proxies={},
            data=data,
            headers=headers,
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response.content

    def send_command(
        self,
        path: str,
        api_key: str,
        command_args: dict = {},
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Result[CommandReport, str]:
        command_args.update({"timeout": timeout, "force_unique_key": True})
        headers = {"X-STACK-API-KEY": api_key}
        result = self.connection._make_post(
            path=path,
            json=command_args,
            headers=headers,
        )
        json_result = json.loads(result)
        return CommandReport(**json_result)

    def resolve_report(
        self, api_key: str, report: CommandReport
    ) -> Result[CommandResult, str]:
        headers = {"X-STACK-API-KEY": api_key}
        if report.status is not CommandStatus.RUNNING:
            return Err(f"Task in not running. Current status: {report.status}")

        result = self._make_get(path=report.result_url, headers=headers)
        result_json = json.loads(result)
        return CommandResult(**result_json)


@serializable()
class HeadScaleClient(BaseVPNClient):
    connection: Type[NodeConnection]
    api_key: str
    _token: Optional[str]

    def __init__(self, connection: Type[NodeConnection], api_key: str) -> None:
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
