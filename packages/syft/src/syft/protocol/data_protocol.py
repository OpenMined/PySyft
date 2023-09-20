# stdlib
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from typing import Dict

# relative
from ..types.syft_object import SyftMigrationRegistry
from ..util.util import get_env
from ..util.util import str_to_bool


def get_dev_mode() -> bool:
    return str_to_bool(get_env("DEV_MODE", "False"))


PROTOCOL_STATE_FILENAME = "protocol_state.json"
PROTOCOL_STATE_FILENAME_DEV = "protocol_state_dev.json"


def data_protocol_file_name():
    return PROTOCOL_STATE_FILENAME_DEV if get_dev_mode() else PROTOCOL_STATE_FILENAME


def data_protocol_dir():
    return os.path.abspath(str(Path(__file__).parent))


def make_hash_sha256(obj_to_hash: Any) -> str:
    def make_hashable(obj):
        if isinstance(obj, (tuple, list)):
            return tuple(make_hashable(e) for e in obj)

        if isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))

        if isinstance(obj, (set, frozenset)):
            return tuple(sorted(make_hashable(e) for e in obj))

        return obj

    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(obj_to_hash)).encode())
    return base64.b64encode(hasher.digest()).decode()


class DataProtocol:
    def __init__(self, filename: str) -> None:
        self.file_path = Path(data_protocol_dir()) / filename
        self.state = self.read_state()

    def calc_latest_object_versions(self):
        object_latest_version_map = {}
        object_versions = iter(
            SyftMigrationRegistry.__migration_version_registry__.items()
        )
        for canonical_name, available_versions in object_versions:
            object_latest_version_map[canonical_name] = list(available_versions)

        return object_latest_version_map

    def read_state(self) -> Dict:
        return json.loads(self.file_path.read_text())

    def save_state(self):
        self.file_path.write_text(json.dumps(self.state))

    def find_deleted_versions(
        self,
        current_object_version_map: Dict,
        new_object_version_map: Dict,
    ):
        deleted_object_classes = set(current_object_version_map) - set(
            new_object_version_map
        )

        deleted_versions_map = {}

        for canonical_name, new_versions in new_object_version_map.items():
            current_versions = current_object_version_map.get(canonical_name)
            if current_versions is None:
                continue

            deleted_versions = list(set(current_versions) - set(new_versions))
            deleted_versions_map[canonical_name] = deleted_versions

        return deleted_object_classes, deleted_versions_map

    def compute_supported_protocol_states(
        self,
        current_protocol_version: int,
        new_object_version_map: Dict,
    ):
        current_protocol_state = self.state[current_protocol_version]
        deleted_object_classes, deleted_versions_map = self.find_deleted_versions(
            current_protocol_state,
            new_object_version_map=new_object_version_map,
        )

        for _, protocol_state in self.state.items():
            object_versions = protocol_state["object_versions"]
            if protocol_state["supported"]:
                continue

            # Check if any object class is deleted,
            # then mark the protocol as not supported.
            is_unsupported = any(
                object_class in object_versions
                for object_class in deleted_object_classes
            )
            if is_unsupported:
                protocol_state["supported"] = False
                continue

            for object_class, supported_versions in deleted_versions_map.items():
                available_versions = object_versions.get(object_class, [])
                unsupported_versions_present = set(available_versions).intersection(
                    supported_versions
                )
                if unsupported_versions_present:
                    is_unsupported = True
                    break

            if is_unsupported:
                protocol_state["supported"] = False

    @property
    def state_defined(self):
        return len(self.state) > 0


def upgrade_protocol():
    data_protocol = DataProtocol(filename=data_protocol_file_name())

    object_version_map = data_protocol.calc_latest_object_versions()
    new_protocol_hash = make_hash_sha256(object_version_map)

    if not data_protocol.state_defined:
        new_protocol_version = 1
    else:
        # Find the current version
        current_protocol_version = sorted(
            data_protocol.state.keys(),
            reverse=True,
        )[0]

        new_protocol_version = current_protocol_version + 1

        current_protocol_state = data_protocol.state[current_protocol_version]
        if current_protocol_state["hash"] == new_protocol_hash:
            print("No change in schema. Skipping upgrade.")
            return

        data_protocol.compute_supported_protocol_states(
            current_protocol_version=current_protocol_version,
            new_object_version_map=object_version_map,
        )

    data_protocol.state[new_protocol_version] = {
        "object_versions": object_version_map,
        "hash": new_protocol_hash,
        "supported": True,
    }
    data_protocol.save_state()
