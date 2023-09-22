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
from result import Ok
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
from ...util.constants import DEFAULT_TIMEOUT
from ...util.util import verify_tls


@serializable()
class VPNRoutes(Enum):
    pass


@serializable()
class BaseVPNClient:
    pass


@serializable()
class CommandStatus(Enum):
    RUNNING = "running"


@serializable()
class CommandReport(SyftObject):
    __canonical_name__ = "CommandReport"
    __version__ = SYFT_OBJECT_VERSION_1

    key: str
    path: str
    status: CommandStatus

    @validator("status", pre=True)
    def validate_status(cls, v: Union[str, CommandStatus]) -> CommandStatus:
        if type(v) is not CommandStatus:
            value = CommandStatus[str(v).upper()]
        else:
            value = v

        return value


@serializable()
class CommandResult(PartialSyftObject):
    __canonical_name__ = "CommandResult"
    __version__ = SYFT_OBJECT_VERSION_1

    report: str
    key: str
    start_time: float
    end_time: float
    process_time: float
    returncode: int
    error: Optional[str]


@serializable()
class VPNClientConnection(NodeConnection):
    __canonical_name__ = "VPNClientConnection"
    __version__ = SYFT_OBJECT_VERSION_1

    url: GridURL
    session_cache: Optional[Session]
    routes: Type[VPNRoutes]

    @validator("url", pre=True)
    def __make_grid_url(cls, url: Union[GridURL, str]) -> GridURL:
        return GridURL.from_url(url)

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
        if response.status_code not in [200, 201, 202]:
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
        command_args: Optional[dict] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Result[CommandReport, str]:
        if command_args is None:
            command_args = {}
        command_args.update({"timeout": timeout, "force_unique_key": True})
        headers = {"X-STACK-API-KEY": api_key}
        result = self._make_post(
            path=path,
            json=command_args,
            headers=headers,
        )
        json_result = json.loads(result)

        json_result["path"] = path

        CommandReport(**json_result)
        return Ok(CommandReport(**json_result))

    def resolve_report(
        self, api_key: str, report: CommandReport
    ) -> Result[CommandResult, str]:
        headers = {"X-STACK-API-KEY": api_key}
        if report.status is not CommandStatus.RUNNING:
            return Err(f"Task in not running. Current status: {report.status}")

        # make get request
        result = self._make_get(
            path=report.path,
            headers=headers,
            params={"key": report.key, "wait": "true"},
        )

        try:
            result_json = json.loads(result.decode())
        except Exception as e:
            return Err(
                f"Failed making get request. Path: {report.path}. Error: {e}",
            )

        command_result = CommandResult(**result_json)

        return Ok(command_result)
