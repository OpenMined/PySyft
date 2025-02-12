"""
SyftBox Client Shim for apps and external dependencies


NOTE: this will likely get refactored as it's own SDK.
But we need it to maintain compatibility with apps
"""

from pathlib import Path

from pydantic import EmailStr
from typing_extensions import Optional, Self

from syftbox.lib.client_config import SyftClientConfig
from syftbox.lib.types import PathLike, to_path
from syftbox.lib.workspace import SyftWorkspace

# this just makes it a bit clear what the default is for the api_data() method
CURRENT_API_REQUEST_NAME = None
MY_DATASITE = None


class Client:
    """
    Client shim for SyftBox Apps

    Minimal set of properties and methods exposed to the apps.
    """

    def __init__(self, conf: SyftClientConfig):
        self.config = conf
        self.workspace = SyftWorkspace(self.config.data_dir)

    @property
    def email(self) -> EmailStr:
        """Email of the current user"""
        return self.config.email

    @property
    def config_path(self) -> Path:
        """Path to the config of the current user"""
        return self.config.path

    @property
    def my_datasite(self) -> Path:
        """Path to the datasite of the current user"""
        return self.workspace.datasites / self.config.email

    @property
    def datasites(self) -> Path:
        """Path to the datasites folder"""
        return self.workspace.datasites

    @property
    def sync_folder(self) -> Path:
        """Deprecated property use `client.datasites` instead"""
        return self.workspace.datasites

    @property
    def datasite_path(self) -> Path:
        """Deprecated property. Use `client.my_datasite` instead"""
        return self.workspace.datasites / self.config.email

    @classmethod
    def load(cls, filepath: Optional[PathLike] = None) -> Self:
        """
        Load the client configuration from the given file path or env var or default location
        Raises: ClientConfigException
        """
        return cls(conf=SyftClientConfig.load(filepath))

    @property
    def api_request_name(self) -> str:
        """Returns the name of root directory of the API request calling this property.

        Use this property instead of hardcoding your API request's directory name,
        as SyftBox may dynamically change it to prevent conflicts.
        """
        # The below works coz we set the cwd to the app's path before executing run.sh (see find_and_run_script method)
        api_path = Path.cwd()
        api_name = api_path.name
        return api_name

    def api_data(
        self, api_request_name: Optional[str] = CURRENT_API_REQUEST_NAME, datasite: Optional[str] = MY_DATASITE
    ) -> Path:
        """
        Gets the filesystem path to an application's API data directory for a specific datasite.

        Args:
            api_request_name (Optional[str], default=CURRENT_API_REQUEST_NAME): The name of the API request
            whose API data path is needed.
                If None, defaults to the name of the API request from which this method is being called.
            datasite (Optional[str], default=MY_DATASITE): The datasite's email.
                If None, defaults to the current user's configured email.

        Returns:
            Path: A filesystem path pointing to '<workspace>/datasites/<datasite>/api_data/<api_request_name>'.
        """
        api_request_name = api_request_name or self.api_request_name
        datasite = datasite or self.config.email
        return self.workspace.datasites / datasite / "api_data" / api_request_name

    def makedirs(self, *paths: PathLike) -> None:
        """Create directories"""

        for path in paths:
            to_path(path).mkdir(parents=True, exist_ok=True)
