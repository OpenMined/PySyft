import asyncio
import json
import platform
import shutil
from pathlib import Path
from types import TracebackType

import uvicorn
from httpx import BaseTransport
from loguru import logger
from pid import PidFile, PidFileAlreadyLockedError, PidFileAlreadyRunningError
from typing_extensions import Optional, Type

from syftbox import __version__
from syftbox.client.api import create_api
from syftbox.client.base import PluginManagerInterface, SyftBoxContextInterface
from syftbox.client.env import syftbox_env
from syftbox.client.exceptions import SyftBoxAlreadyRunning
from syftbox.client.logger import setup_logger
from syftbox.client.plugin_manager import PluginManager
from syftbox.client.server_client import SyftBoxClient
from syftbox.client.utils import error_reporting, file_manager, macos
from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.datasite import create_datasite
from syftbox.lib.exceptions import SyftBoxException
from syftbox.lib.ignore import IGNORE_FILENAME
from syftbox.lib.platform import OS_NAME, OS_VERSION, PYTHON_VERSION
from syftbox.lib.workspace import SyftWorkspace

SCRIPT_DIR = Path(__file__).parent
ASSETS_FOLDER = SCRIPT_DIR.parent / "assets"
ICON_FOLDER = ASSETS_FOLDER / "icon"
METADATA_FILENAME = ".metadata.json"


class SyftBoxRunner:
    """The local SyftBox instance.

    This is the main SyftBox instance that handles workspace data, server
    communication, and local API services. Only one instance can run
    for a given workspace directory.

    Warning:
        This class should not be imported directly by sub-systems.
        Use the provided interfaces and context objects instead.

    Raises:
        SyftBoxAlreadyRunning: If another client is already running for the same workspace
        Exception: If the client fails to start due to any reason
    """

    def __init__(
        self,
        config: SyftClientConfig,
        log_level: str = "INFO",
        server_transport: Optional[BaseTransport] = None,
        **kwargs: dict,
    ) -> None:
        self.config = config
        self.log_level = log_level

        self.workspace = SyftWorkspace(self.config.data_dir)
        self.pid = PidFile(pidname="syftbox.pid", piddir=self.workspace.data_dir)
        self.client = SyftBoxClient.from_config(self.config, transport=server_transport)

        # create a single client context shared across components
        self.__ctx = SyftBoxContext(
            self.config,
            self.workspace,
            client=self.client,
            plugins=None,
        )
        self.plugins = PluginManager(self.__ctx, sync_manager=None, app_runner=None, **kwargs)
        # make plugins available to the context
        self.__ctx.plugins = self.plugins

        # kwargs for making customization/unit testing easier
        # this will be replaced with a sophisticated plugin system
        self.__local_server: uvicorn.Server = None

    @property
    def is_registered(self) -> bool:
        """Check if the current user is registered with the server"""
        return bool(self.config.token)

    @property
    def datasite(self) -> Path:
        """The datasite of the current user"""
        return self.workspace.datasites / self.config.email

    @property
    def public_dir(self) -> Path:
        """The public directory in the datasite of the current user"""
        return self.datasite / "public"

    @property
    def context(self) -> "SyftBoxContext":
        return self.__ctx

    def start(self) -> None:
        try:
            self.pid.create()
        except PidFileAlreadyLockedError:
            raise SyftBoxAlreadyRunning(f"Another instance of SyftBox is running on {self.config.data_dir}")
        self.create_metadata_file()

        logger.info("Started SyftBox")

        self.config.save()  # commit config changes (like migration) to disk after PID is created
        self.workspace.mkdirs()  # create the workspace directories
        self.register_self()  # register the email with the server
        self.init_datasite()  # init the datasite on local machine

        # start plugins/components
        self.plugins.start()
        return self.__run_local_server()

    @property
    def metadata_path(self) -> Path:
        return self.workspace.data_dir / METADATA_FILENAME

    def create_metadata_file(self) -> None:
        metadata_json = self.config.model_dump(mode="json")
        metadata_json["version"] = __version__
        self.metadata_path.write_text(json.dumps(metadata_json, indent=2))

    def shutdown(self) -> None:
        if self.__local_server:
            _result = asyncio.run(self.__local_server.shutdown())

        self.plugins.stop()

        self.pid.close()
        logger.info("SyftBox shutdown complete")

    def check_pidfile(self) -> str:
        """Check if another instance of SyftBox is running"""

        try:
            return self.pid.check()
        except PidFileAlreadyRunningError:
            raise SyftBoxAlreadyRunning(f"Another instance of SyftBox is running on {self.config.data_dir}")

    def init_datasite(self) -> None:
        if self.datasite.exists():
            return
        create_datasite(self.context)

    def register_self(self) -> None:
        """Register the user's email with the SyftBox cache server"""
        if self.is_registered:
            return
        try:
            token = self.client.register(self.config.email)
            # TODO + FIXME - once we have JWT, we should not store token in config!
            # ideally in OS keychain (using keyring) or
            # in a separate location under self.workspace.plugins
            self.config.token = str(token)
            self.config.save()
            logger.info("Email registration successful")
        except Exception as e:
            raise SyftBoxException(f"Failed to register with the server - {e}") from e

    def __run_local_server(self) -> None:
        logger.info(f"Starting local server on {self.config.client_url}")
        app = create_api(self.__ctx)
        self.__local_server = uvicorn.Server(
            config=uvicorn.Config(
                app=app,
                host=self.config.client_url.host,
                port=self.config.client_url.port,
                log_level=self.log_level.lower(),
            )
        )
        return self.__local_server.run()

    # utils
    def open_datasites_dir(self) -> None:
        file_manager.open_dir(str(self.workspace.datasites))

    def copy_icons(self) -> None:
        self.workspace.mkdirs()
        if platform.system() == "Darwin":
            macos.copy_icon_file(ICON_FOLDER, self.workspace.data_dir)

    def log_system_info(self) -> None:
        self.client.log_analytics_event(
            event_name="system_info",
            os_name=OS_NAME,
            os_version=OS_VERSION,
            syftbox_version=__version__,
            python_version=PYTHON_VERSION,
        )

    def __enter__(self) -> "SyftBoxRunner":
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        self.shutdown()


class SyftBoxContext(SyftBoxContextInterface):
    """
    Provides a light-weight context object for sub-systems to interact with.
    It will be instantiated by LocalSyftBox, but sub-systems can freely pass it around.
    """

    def __init__(
        self,
        config: SyftClientConfig,
        workspace: SyftWorkspace,
        client: SyftBoxClient,
        plugins: Optional[PluginManagerInterface],
    ):
        self.config = config
        self.workspace = workspace
        self.client = client
        self.plugins = plugins

    @property
    def email(self) -> str:
        return self.config.email

    @property
    def my_datasite(self) -> Path:
        return self.workspace.datasites / self.config.email

    @property
    def all_datasites(self) -> list[str]:
        """List all datasites in the workspace"""
        return [d.name for d in self.workspace.datasites.iterdir() if (d.is_dir() and "@" in d.name)]

    def __repr__(self) -> str:
        return f"SyftBoxContext<{self.config.email}, {self.config.data_dir.as_posix()}>"


def run_apps_to_api_migration(new_ws: SyftWorkspace) -> None:
    old_sync_folder = new_ws.data_dir
    old_apps_dir = old_sync_folder / "apps"
    new_apps_dir = new_ws.apps

    if old_apps_dir.exists():
        logger.info(f"Migrating directory apps —> {new_apps_dir.relative_to(new_ws.data_dir)}...")
        if new_apps_dir.exists():
            shutil.rmtree(new_apps_dir)
        shutil.move(str(old_apps_dir), str(new_apps_dir))


def run_migration(config: SyftClientConfig, migrate_datasite: bool = True) -> None:
    # first run config migration
    config.migrate()

    # then run workspace migration
    new_ws = SyftWorkspace(config.data_dir)

    # migrate workspace/apps to workspace/apis
    run_apps_to_api_migration(new_ws)

    # check for old dir structure and migrate to new
    # data_dir == sync_folder
    old_sync_folder = new_ws.data_dir
    old_datasite_path = Path(old_sync_folder, config.email)

    if not migrate_datasite:
        return

    # Option 2: if syftbox folder has old structure, migrate to new
    if old_datasite_path.exists():
        logger.info("Migrating to new datasite structure")
        new_ws.mkdirs()

        # create the datasites directory & move all under it
        for dir in old_sync_folder.glob("*@*"):
            shutil.move(str(dir), str(new_ws.datasites))

        # move syftignore file
        old_ignore_file = old_sync_folder / IGNORE_FILENAME
        if old_ignore_file.exists():
            shutil.move(str(old_ignore_file), str(new_ws.datasites / IGNORE_FILENAME))

        # move old sync state file
        old_sync_state = old_sync_folder / ".syft" / "local_syncstate.json"
        if old_sync_state.exists():
            shutil.move(str(old_sync_state), str(new_ws.plugins / "local_syncstate.json"))
        if old_sync_state.parent.exists():
            shutil.rmtree(str(old_sync_state.parent))


def run_syftbox(
    client_config: SyftClientConfig,
    open_dir: bool = False,
    log_level: str = "INFO",
    migrate_datasite: bool = True,
) -> int:
    """Run the SyftBox client"""
    syftbox_instance = None

    setup_logger(log_level, log_dir=client_config.data_dir / "logs")

    error_config = error_reporting.make_error_report(client_config)
    logger.info(
        f"Client metadata\n{error_config.model_dump_json(indent=2, exclude={'client_config': {'access_token'}})}"
    )

    # a flag to disable icons
    # GitHub CI needs to zip sync dir in tests and fails when it encounters Icon\r files
    if syftbox_env.DISABLE_ICONS:
        logger.debug("Directory icons are disabled")

    try:
        syftbox_instance = SyftBoxRunner(client_config, log_level=log_level)
        # we don't want to run migration if another instance of client is already running
        if syftbox_instance.check_pidfile():
            run_migration(client_config, migrate_datasite=migrate_datasite)
        if not syftbox_env.DISABLE_ICONS:
            syftbox_instance.copy_icons()
        if open_dir:
            syftbox_instance.open_datasites_dir()
        syftbox_instance.log_system_info()
        syftbox_instance.start()
    except SyftBoxAlreadyRunning as e:
        logger.error(e)
        return -1
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down the client")
    except Exception as e:
        logger.exception("Unhandled exception when starting the client", e)
        return -2
    finally:
        if syftbox_instance is not None:
            syftbox_instance.shutdown()
    return 0
