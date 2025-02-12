import datetime
import sys
from platform import platform

import httpx
from httpx._models import Response
from pydantic import AnyHttpUrl, BaseModel, Field

from syftbox import __version__
from syftbox.client.env import syftbox_env
from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.http import SYFTBOX_HEADERS


class ErrorReport(BaseModel):
    client_config: SyftClientConfig
    client_syftbox_version: str = __version__
    python_version: str = sys.version
    platform: str = platform()
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    env: dict = Field(default=syftbox_env.model_dump())

    @classmethod
    def from_client_config(cls, client_config: SyftClientConfig) -> "ErrorReport":
        return cls(
            client_config=client_config,
            server_version=try_get_server_version(client_config.server_url),
        )


def make_error_report(client_config: SyftClientConfig) -> ErrorReport:
    return ErrorReport.from_client_config(client_config)


def try_get_server_version(server_url: AnyHttpUrl) -> Response:
    try:
        # do not use the server_client here, as it may not be in bad state
        return httpx.get(
            f"{server_url}/info?error_report=1",
            headers=SYFTBOX_HEADERS,
        ).json()["version"]
    except Exception:
        return None
