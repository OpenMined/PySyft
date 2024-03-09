# stdlib
from enum import Enum
import os
import subprocess  # nosec
import sys
from typing import Any
from typing import Callable
from typing import Tuple
from typing import Union
from urllib.parse import urlparse

# relative
from .dummynum import DummyNum


class NodeSideType(str, Enum):
    LOW_SIDE = "low"
    HIGH_SIDE = "high"

    def __str__(self) -> str:
        # Use values when transforming NodeType to str
        return self.value


class ImportFromSyft:
    @staticmethod
    def import_syft_error() -> Callable:
        try:
            # syft absolute
            from syft.service.response import SyftError
        except Exception:
            SyftError = DummyNum

        return SyftError

    @staticmethod
    def import_stage_protocol_changes() -> Callable:
        try:
            # syft absolute
            from syft.protocol.data_protocol import stage_protocol_changes
        except Exception:

            def stage_protocol_changes(*args: Any, **kwargs: Any) -> None:
                pass

        return stage_protocol_changes

    @staticmethod
    def import_node_type() -> Callable:
        try:
            # syft absolute
            from syft.abstract_node import NodeType
        except Exception:
            NodeType = DummyNum

        return NodeType


def from_url(url: str) -> Tuple[str, str, int, str, Union[Any, str]]:
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
        return (
            host_or_ip,
            parts.path,
            port,
            parts.scheme,
            getattr(parts, "query", ""),
        )
    except Exception as e:
        print(f"Failed to convert url: {url} to GridURL. {e}")
        raise e


def fix_windows_virtualenv_api(cls: type) -> None:
    # fix bug in windows
    def _python_rpath(self: Any) -> str:
        """The relative path (from environment root) to python."""
        # Windows virtualenv installation installs pip to the [Ss]cripts
        # folder. Here's a simple check to support:
        if sys.platform == "win32":
            # fix here https://github.com/sjkingo/virtualenv-api/issues/47
            return os.path.join(self.path, "Scripts", "python.exe")
        return os.path.join("bin", "python")

    cls._python_rpath = property(_python_rpath)


def shell(command: str) -> str:
    try:
        output = subprocess.check_output(  # nosec
            command, shell=True, stderr=subprocess.STDOUT
        )
    except Exception:
        output = b""
    return output.decode("utf-8")
