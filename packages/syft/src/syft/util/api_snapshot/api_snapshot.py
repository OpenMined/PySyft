# stdlib
from collections import OrderedDict
from functools import cache
import hashlib
from inspect import Signature
import json
import os
from pathlib import Path

# relative
from ...service.service import ServiceConfigRegistry
from ...service.warnings import APIEndpointWarning
from ..util import get_root_data_path
from .json_diff import json_diff

API_DUMP_JSON_FILENAME = "syft_api_dump.json"
API_DIFF_JSON_FILENAME = "syft_api_diff.json"


def api_snapshot_dir() -> Path:
    """A helper function to get the path of the API snapshot directory."""
    return Path(os.path.abspath(str(Path(__file__).parent)))


class SyftAPISnapshot:
    def __init__(self, filename: str = API_DUMP_JSON_FILENAME) -> None:
        """
        Initialize the SyftAPISnapshot object.

        Args:
            filename (str): The name of the JSON file to load the API snapshot from.
                            Defaults to API_DUMP_JSON_FILENAME.
        """
        self.file_path = api_snapshot_dir() / filename
        self.history = self.load_map()
        self.state = self.build_map()

    @staticmethod
    def extract_service_name(path: str) -> str:
        """
        Extract the service name from the given path.

        Args:
            path (str): The path of the service.

        Returns:
            str: The extracted service name.
        """
        return path.split(".")[0].capitalize()

    @staticmethod
    def extract_arguments(signature: Signature) -> dict:
        """
        Extract the arguments from the given signature.

        Args:
            signature (Signature): The signature object.

        Returns:
            dict: The extracted arguments as a dictionary.
        """
        signature_kwargs = {
            f"{v.name}": f"{v.annotation}" for k, v in signature.parameters.items()
        }
        return OrderedDict(sorted(signature_kwargs.items()))

    @staticmethod
    def get_role_level(roles: list) -> str:
        """
        Get the role level from the given list of roles.

        Args:
            roles (list): The list of roles.

        Returns:
            str: The role level.
        """
        return sorted(roles)[0].name + "_ROLE_LEVEL"

    @staticmethod
    def extract_warning_info(warning: APIEndpointWarning) -> dict | str:
        """
        Extract the warning information from the given APIEndpointWarning object.

        Args:
            warning (APIEndpointWarning): The APIEndpointWarning object.

        Returns:
            dict: The extracted warning information.
        """
        if not warning:
            return ""

        return {
            "name": f"{warning.__class__.__name__}",
            "confirmation": warning.confirmation,
            "enabled": warning.enabled,
        }

    @staticmethod
    def generate_hash(api_map: OrderedDict) -> str:
        """
        Generate a hash for the given API map.

        Args:
            api_map (OrderedDict): The API map.

        Returns:
            str: The generated hash.
        """
        return hashlib.sha256(json.dumps(api_map).encode()).hexdigest()

    def load_map(self) -> OrderedDict:
        """
        Load the API map from the JSON file.

        Returns:
            OrderedDict: The loaded API map.
        """

        if not self.file_path.exists():
            return OrderedDict()

        return OrderedDict(json.loads(self.file_path.read_text()))

    def build_map(self) -> OrderedDict:
        """
        Build the API map.

        Returns:
            OrderedDict: The built API map.
        """
        api_details = OrderedDict()
        for (
            _,
            service_config,
        ) in ServiceConfigRegistry.__service_config_registry__.items():
            service_name = self.extract_service_name(service_config.private_path)
            warning = service_config.warning
            signature = service_config.signature
            role_level = self.get_role_level(service_config.roles)
            api_detail = {
                "public_path": service_config.public_path,
                "RBAC_permission": f"{role_level}",
                "signature": self.extract_arguments(service_config.signature),
                "return_type": f"{signature.return_annotation}" if signature else "",
                "warning": self.extract_warning_info(warning),
                # "unwrap_on_success": getattrservice_config.unwrap_on_success,
            }
            api_detail["hash"] = self.generate_hash(api_detail)
            api_details[f"{service_name}.{service_config.public_path}"] = OrderedDict(
                api_detail
            )
        return api_details

    def save_as_json(self) -> None:
        """
        Save the API map as a JSON file.
        """
        self.file_path.write_text(json.dumps(self.state, indent=2))

    def calc_diff(self, save: bool = False) -> dict:
        """
        Calculate the difference between the current API snapshot and the previous one.

        Args:
            save (bool): Whether to save the difference as a JSON file. Defaults to False.

        Returns:
            dict: The difference between the API snapshots.
        """
        diff = json_diff(self.history, self.state)
        if save:
            diff_file_path = get_root_data_path() / API_DIFF_JSON_FILENAME
            diff_file_path.write_text(json.dumps(diff, indent=2))

        return diff


@cache
def get_api_snapshot() -> SyftAPISnapshot:
    """
    Retrieves the API snapshot.
    """
    snapshot = SyftAPISnapshot(filename=API_DUMP_JSON_FILENAME)
    return snapshot


def take_api_snapshot() -> SyftAPISnapshot:
    """
    Takes a snapshot of the API and saves it as a JSON file.
    """
    snapshot = get_api_snapshot()
    snapshot.save_as_json()
    return snapshot


def show_api_diff() -> dict:
    """
    Calculates the difference between the current API snapshot and the previous one,
    saves it as a JSON file, and returns the difference.
    """
    snapshot = get_api_snapshot()

    diff = snapshot.calc_diff(save=True)
    print("Generated API diff file at: ", get_root_data_path() / API_DIFF_JSON_FILENAME)
    return diff
