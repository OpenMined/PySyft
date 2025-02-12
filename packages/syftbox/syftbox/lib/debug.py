import os
import platform
import shutil
import sys
from pathlib import PurePath
from typing import Any, Optional, Union
from venv import logger

import psutil

from syftbox import __version__
from syftbox.app.manager import list_app
from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.exceptions import ClientConfigException
from syftbox.lib.platform import OS_ARCH, OS_NAME, OS_VERSION
from syftbox.lib.types import PathLike
from syftbox.lib.workspace import SyftWorkspace


def debug_report(config_path: Optional[PathLike] = None) -> dict:
    client_config: Optional[Union[SyftClientConfig, dict]] = None
    apps = []
    app_dir = None
    try:
        client_config = SyftClientConfig.load(config_path)
        workspace = SyftWorkspace(client_config.data_dir)
        result = list_app(workspace)
        app_dir = result.apps_dir
        apps = [app.name for app in result.apps]
        client_config = client_config.as_dict(exclude={"token", "access_token"})
    except ClientConfigException:
        pass
    except Exception as e:
        logger.exception("Error loading client config", e)
        pass

    syftbox_path = shutil.which("syftbox")

    return {
        "system": {
            "os": {
                "name": OS_NAME,
                "version": OS_VERSION,
                "arch": OS_ARCH,
            },
            "python": {
                "version": platform.python_version(),
                "binary_location": sys.executable,
            },
            "resources": {
                "cpus": psutil.cpu_count(logical=True),
                "architecture": platform.machine(),
                "ram": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            },
        },
        "syftbox": {
            "version": __version__,
            "command": syftbox_path or "syftbox executable not found in PATH",
            "client_config_path": config_path,
            "client_config": client_config,
            "apps_dir": app_dir,
            "apps": apps,
        },
        "syftbox_env": {key: value for key, value in os.environ.items() if key.startswith("SYFT")},
    }


def debug_report_yaml(config_path: Optional[PathLike] = None) -> str:
    import yaml
    from pydantic_core import Url

    def str_representer(dumper: yaml.Dumper, val: Any) -> yaml.ScalarNode:
        return dumper.represent_scalar("tag:yaml.org,2002:str", str(val))

    # Register the custom representers
    yaml.add_multi_representer(PurePath, str_representer)
    yaml.add_representer(Url, str_representer)

    report = debug_report(config_path)
    return yaml.dump(report, default_flow_style=False, sort_keys=False)
