# stdlib
import hashlib
import json
import os
from pathlib import Path
from typing import Dict
from typing import Type

# relative
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftMigrationRegistry
from ..util.util import get_env
from ..util.util import index_syft_by_module_name
from ..util.util import str_to_bool


def get_dev_mode() -> bool:
    return str_to_bool(get_env("DEV_MODE", "False"))


PROTOCOL_STATE_FILENAME = "protocol_state.json"
PROTOCOL_STATE_FILENAME_DEV = "protocol_state_dev.json"


def data_protocol_file_name():
    return PROTOCOL_STATE_FILENAME_DEV if get_dev_mode() else PROTOCOL_STATE_FILENAME


def data_protocol_dir():
    return os.path.abspath(str(Path(__file__).parent))


class InConsistentVersionException(Exception):
    pass


class DataProtocol:
    def __init__(self, filename: str) -> None:
        self.file_path = Path(data_protocol_dir()) / filename
        self.state = self.read_state()

    @staticmethod
    def _calculate_object_hash(klass: Type[SyftBaseObject]) -> str:
        field_data = {
            field_name: repr(model_field.annotation)
            for field_name, model_field in klass.__fields__.items()
        }
        obj_meta_info = {
            "canonical_name": klass.__canonical_name__,
            "version": klass.__version__,
            "unique_keys": getattr(klass, "__attr_unique__", []),
            "field_data": field_data,
        }

        return hashlib.sha256(json.dumps(obj_meta_info).encode()).hexdigest()

    def calc_latest_object_versions(self):
        object_latest_version_map = {}
        migration_registry = SyftMigrationRegistry.__migration_version_registry__
        for canonical_name in migration_registry:
            available_versions = migration_registry[canonical_name]
            version_obj_hash_map = {}
            for object_version, fqn in available_versions.items():
                object_klass = index_syft_by_module_name(fqn)
                object_hash = self._calculate_object_hash(object_klass)
                version_obj_hash_map[object_version] = object_hash
            object_latest_version_map[canonical_name] = version_obj_hash_map

        return object_latest_version_map

    def read_state(self) -> Dict:
        return json.loads(self.file_path.read_text())

    def save_state(self):
        self.file_path.write_text(json.dumps(self.state))

    def find_deleted_versions(
        self,
        current_object_to_version_map: Dict,
        new_object_to_version_map: Dict,
    ):
        deleted_object_classes = set(current_object_to_version_map).difference(
            new_object_to_version_map.keys()
        )

        deleted_versions_map = {}

        for canonical_name, new_versions in new_object_to_version_map.items():
            current_versions = current_object_to_version_map.get(
                canonical_name,
                None,
            )
            if current_versions is None:
                continue

            deleted_versions = list(set(current_versions).difference(new_versions))
            deleted_versions_map[canonical_name] = deleted_versions

        return deleted_object_classes, deleted_versions_map

    def recompute_supported_states(
        self,
        current_protocol_version: int,
        new_object_to_version_map: Dict,
    ):
        current_protocol_state = self.state[current_protocol_version]
        deleted_object_classes, deleted_versions_map = self.find_deleted_versions(
            current_protocol_state,
            new_object_to_version_map=new_object_to_version_map,
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

    @property
    def latest_version(self):
        return max(self.state.keys())

    @staticmethod
    def _hash_to_sha256(obj_dict: Dict) -> str:
        return hashlib.sha256(json.dumps(obj_dict).encode()).hexdigest()

    def upgrade(self):
        object_to_version_map = self.calc_latest_object_versions()
        new_protocol_hash = self._hash_to_sha256(object_to_version_map)

        if not self.state_defined:
            new_protocol_version = 1
        else:
            # Find the current version
            current_protocol_version = self.latest_version

            new_protocol_version = int(current_protocol_version) + 1

            current_protocol_state = self.state[current_protocol_version]
            if current_protocol_state["hash"] == new_protocol_hash:
                print("No change in schema. Skipping upgrade.")
                return

            self.recompute_supported_states(
                current_protocol_version=current_protocol_version,
                new_object_to_version_map=object_to_version_map,
            )

        self.state[new_protocol_version] = {
            "object_versions": object_to_version_map,
            "hash": new_protocol_hash,
            "supported": True,
        }
        self.save_state()

    def validate_current_state(self) -> bool:
        current_object_version_map = self.state[self.latest_version]["object_versions"]
        inconsistent_versions = []

        migration_registry = SyftMigrationRegistry.__migration_version_registry__
        for canonical_name in migration_registry:
            available_versions = migration_registry[canonical_name]
            curr_version_hash_map = current_object_version_map.get(canonical_name, {})
            for object_version, fqn in available_versions.items():
                object_klass = index_syft_by_module_name(fqn)
                object_hash = self._calculate_object_hash(object_klass)
                if curr_version_hash_map.get(str(object_version), None) != object_hash:
                    inconsistent_versions.append((canonical_name, object_version))

        if len(inconsistent_versions) > 0:
            raise InConsistentVersionException(
                f"Version update is required for the following objects.\n {inconsistent_versions}"
            )

        return True

    @property
    def supported_protocols(self):
        """Returns a list of protocol numbers that are marked as supported."""
        return [
            int(protocol_version)
            for protocol_version, protocol_state in self.state.items()
            if protocol_state["supported"]
        ]


def upgrade_protocol():
    data_protocol = DataProtocol(filename=data_protocol_file_name())
    data_protocol.upgrade()


def validate_protocol():
    pass
