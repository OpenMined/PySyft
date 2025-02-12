import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from syftbox.app.install import InstallResult, install
from syftbox.lib.workspace import SyftWorkspace


@dataclass
class InstalledApps:
    apps_dir: Path
    apps: List[Path]


def install_app(workspace: SyftWorkspace, repository: str, branch: str = "main") -> InstallResult:
    return install(workspace.apps, repository, branch)


def list_app(workspace: SyftWorkspace) -> InstalledApps:
    apps = []
    if workspace.apps.exists() and workspace.apps.is_dir():
        apps = sorted([app for app in workspace.apps.iterdir() if app.is_dir()])
    return InstalledApps(workspace.apps, apps)


def uninstall_app(app_name: str, workspace: SyftWorkspace) -> Optional[Path]:
    app_dir = Path(workspace.apps, app_name)
    # first check for symlink
    if app_dir.exists() and app_dir.is_symlink():
        app_dir.unlink()
        return app_dir
    elif app_dir.exists() and app_dir.is_dir():
        shutil.rmtree(app_dir)
        return app_dir
    else:
        return None


def update_app(ws: SyftWorkspace) -> None:
    pass
