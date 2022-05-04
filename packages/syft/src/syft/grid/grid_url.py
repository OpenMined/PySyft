# future
from __future__ import annotations

# stdlib
import copy
import os
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
    __attr_allowlist__ = ["protocol", "host_or_ip", "port", "path", "query"]

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
                host_or_ip=host_or_ip,
                path=parts.path,
                port=port,
                protocol=parts.scheme,
                query=getattr(parts, "query", ""),
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
        query: str = "",
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
        self.query = query

    def with_path(self, path: str) -> GridURL:
        dupe = copy.copy(self)
        dupe.path = path
        return dupe

    def as_container_host(self, container_host: Optional[str] = None) -> GridURL:
        if self.host_or_ip not in ["localhost", "docker-host", "host.k3d.internal"]:
            return self

        if container_host is None:
            # TODO: we could move config.py to syft and then the Settings singleton
            # could be importable in all parts of the code
            container_host = os.getenv("CONTAINER_HOST", "docker")

        hostname = "docker-host" if container_host == "docker" else "host.k3d.internal"

        return GridURL(
            protocol=self.protocol,
            host_or_ip=hostname,
            port=self.port,
            path=self.path,
        )

    @property
    def query_string(self) -> str:
        query_string = ""
        if len(self.query) > 0:
            query_string = f"?{self.query}"
        return query_string

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.path}{self.query_string}"

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host_or_ip}:{self.port}"

    @property
    def url_path(self) -> str:
        return f"{self.path}{self.query_string}"

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
        return GridURL.from_url(url=f"{new_base_url}{self.path}{self.query_string}")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.url}>"

    def __str__(self) -> str:
        return self.url

    def __hash__(self) -> int:
        return hash(self.__str__())

    def copy(self) -> GridURL:
        return GridURL.from_url(self.url)
