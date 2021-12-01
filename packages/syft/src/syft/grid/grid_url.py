# future
from __future__ import annotations

# stdlib
import copy
from typing import Optional
from typing import Union
from urllib.parse import urlparse

# third party
import requests

# relative
from ..core.common.serde.serializable import serializable
from ..util import verify_tls


@serializable(recursive_serde=True)
class GridURL:
    __attr_allowlist__ = ["protocol", "host_or_ip", "port", "path"]

    @staticmethod
    def from_url(url: Union[str, GridURL]) -> GridURL:
        if isinstance(url, GridURL):
            return url
        try:
            # urlparse doesnt handle no protocol properly
            if "://" not in url:
                url = "http://" + url
            parts = urlparse(url)
            host_or_ip_parts = parts.netloc.split(":")
            # netloc is host:port
            port = 80
            if len(host_or_ip_parts) > 1:
                port = int(host_or_ip_parts[1])
            host_or_ip = host_or_ip_parts[0]
            return GridURL(
                host_or_ip=host_or_ip, path=parts.path, port=port, protocol=parts.scheme
            )
        except Exception as e:
            print(f"Failed to convert url: {url} to GridURL. {e}")
            raise e

    def __init__(
        self,
        protocol: str = "http",
        host_or_ip: str = "localhost",
        port: Optional[int] = 80,
        path: str = "",
    ) -> None:
        # in case a preferred port is listed but its not clear if an alternative
        # port was included in the supplied host_or_ip:port combo passed in earlier
        if ":" in host_or_ip:
            sub_grid_url: GridURL = GridURL.from_url(host_or_ip)
            host_or_ip = str(sub_grid_url.host_or_ip)  # type: ignore
            port = int(sub_grid_url.port)  # type: ignore
            protocol = str(sub_grid_url.protocol)  # type: ignore
            path = str(sub_grid_url.path)  # type: ignore
        elif port is None:
            port = 80

        self.host_or_ip = host_or_ip
        self.path = path
        self.port = port
        self.protocol = protocol

    def with_path(self, path: str) -> GridURL:
        dupe = copy.copy(self)
        dupe.path = path
        return dupe

    def as_docker_host(self) -> GridURL:
        if self.host_or_ip != "localhost":
            return self
        return GridURL(
            protocol=self.protocol,
            host_or_ip="docker-host",
            port=self.port,
            path=self.path,
        )

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.path}"

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host_or_ip}:{self.port}"

    def to_tls(self) -> GridURL:
        if self.protocol == "https":
            return self

        # TODO: only ignore ssl in dev mode
        r = requests.get(
            self.base_url, verify=verify_tls()
        )  # ignore ssl cert if its fake
        new_base_url = r.url
        if new_base_url.endswith("/"):
            new_base_url = new_base_url[0:-1]
        return GridURL.from_url(url=f"{new_base_url}{self.path}")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.url}>"

    def __str__(self) -> str:
        return self.url

    def __hash__(self) -> int:
        return hash(self.__str__())
