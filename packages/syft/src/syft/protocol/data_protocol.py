# stdlib
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type

# third party
from result import Result

# relative
from ..serde.recursive import TYPE_BANK
from ..service.response import SyftError
from ..service.response import SyftException
from ..service.response import SyftSuccess
from ..types.syft_object import SyftBaseObject


def natural_key(key: int | str) -> list[int]:
    """Define key for natural ordering of strings."""
    if isinstance(key, int):
        key = str(key)
    return [int(s) if s.isdigit() else s for s in re.split("(\d+)", key)]


def sort_dict_naturally(d: dict) -> dict:
    """Sort dictionary by keys in natural order."""
    return {k: d[k] for k in sorted(d.keys(), key=natural_key)}


def protocol_state_builder(protocol_dict: dict, stop_key: Optional[str] = None) -> dict:
    sorted_dict = sort_dict_naturally(protocol_dict)
    state_dict = defaultdict(dict)
    for k, _v in sorted_dict.items():
        # stop early
        if stop_key == k:
            return state_dict
        object_versions = sorted_dict[k]["object_versions"]
        for canonical_name, versions in object_versions.items():
            for version, object_metadata in versions.items():
                action = object_metadata["action"]
                version = object_metadata["version"]
                hash_str = object_metadata["hash"]
                state_versions = state_dict[canonical_name]
                if action == "add" and (
                    str(version) in state_versions.keys()
                    or hash_str in state_versions.values()
                ):
                    raise Exception(
                        f"Can't add {object_metadata} already in state {versions}"
                    )
                elif action == "remove" and (
                    str(version) not in state_versions.keys()
                    or hash_str not in state_versions.values()
                ):
                    raise Exception(
                        f"Can't remove {object_metadata} missing from state {versions}"
                    )
                if action == "add":
                    state_dict[canonical_name][str(version)] = hash_str
                elif action == "remove":
                    del state_dict[canonical_name][str(version)]
    return state_dict


PROTOCOL_STATE_FILENAME = "protocol_version.json"


def data_protocol_file_name():
    return PROTOCOL_STATE_FILENAME


def data_protocol_dir():
    return os.path.abspath(str(Path(__file__).parent))


class InConsistentVersionException(Exception):
    pass


def diff_state(state: dict) -> dict:
    object_diff = defaultdict(dict)
    compare_dict = defaultdict(dict)
    for k in TYPE_BANK:
        (
            nonrecursive,
            serialize,
            deserialize,
            attribute_list,
            exclude_attrs_list,
            serde_overrides,
            hash_exclude_attrs,
            cls,
            attribute_types,
            version,
        ) = TYPE_BANK[k]
        if issubclass(cls, SyftBaseObject):
            canonical_name = cls.__canonical_name__
            hash_str = DataProtocol._calculate_object_hash(cls)

            # build this up for later
            compare_dict[canonical_name][version] = hash_str

            if canonical_name not in state:
                # new object so its an add
                object_diff[canonical_name][str(version)] = {}
                object_diff[canonical_name][str(version)]["version"] = version
                object_diff[canonical_name][str(version)]["hash"] = hash_str
                object_diff[canonical_name][str(version)]["action"] = "add"
                continue

            versions = state[canonical_name]
            if str(version) in versions.keys() and versions[str(version)] == hash_str:
                # already there so do nothing
                continue
            elif str(version) in versions.keys():
                raise Exception(
                    f"{canonical_name} {cls} version {version} hash has changed. "
                    + f"{hash_str} not in {versions.values()}. "
                    + "You probably need to bump the version number."
                )
            else:
                # new object so its an add
                object_diff[canonical_name][str(version)] = {}
                object_diff[canonical_name][str(version)]["version"] = version
                object_diff[canonical_name][str(version)]["hash"] = hash_str
                object_diff[canonical_name][str(version)]["action"] = "add"
                continue

    # now check for remove actions
    for canonical_name in state:
        for version, hash_str in state[canonical_name].items():
            if canonical_name not in compare_dict:
                # missing so its a remove
                object_diff[canonical_name][str(version)] = {}
                object_diff[canonical_name][str(version)]["version"] = version
                object_diff[canonical_name][str(version)]["hash"] = hash_str
                object_diff[canonical_name][str(version)]["action"] = "remove"
                continue
            versions = compare_dict[canonical_name]
            if str(version) in versions.keys():
                # missing so its a remove
                object_diff[canonical_name][str(version)] = {}
                object_diff[canonical_name][str(version)]["version"] = version
                object_diff[canonical_name][str(version)]["hash"] = hash_str
                object_diff[canonical_name][str(version)]["action"] = "remove"
                continue
    return object_diff


class DataProtocol:
    def __init__(self, filename: str) -> None:
        self.file_path = Path(data_protocol_dir()) / filename
        self.protocol_history = self.read_history()
        self.state = self.build_state()

    @staticmethod
    def _calculate_object_hash(klass: Type[SyftBaseObject]) -> str:
        # TODO: this depends on what is marked as serde
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

    # def calc_latest_object_versions(self):
    #     object_latest_version_map = {}
    #     migration_registry = SyftMigrationRegistry.__migration_version_registry__
    #     for canonical_name in migration_registry:
    #         available_versions = migration_registry[canonical_name]
    #         version_obj_hash_map = {}
    #         for object_version, fqn in available_versions.items():
    #             object_klass = index_syft_by_module_name(fqn)
    #             object_hash = self._calculate_object_hash(object_klass)
    #             version_obj_hash_map[object_version] = object_hash
    #         object_latest_version_map[canonical_name] = version_obj_hash_map

    #     return object_latest_version_map

    def read_history(self) -> Dict:
        return json.loads(self.file_path.read_text())

    def save_history(self, history: dict) -> None:
        self.file_path.write_text(json.dumps(history, indent=2))

    # def find_deleted_versions(
    #     self,
    #     current_object_to_version_map: Dict,
    #     new_object_to_version_map: Dict,
    # ):
    #     deleted_object_classes = set(current_object_to_version_map).difference(
    #         new_object_to_version_map.keys()
    #     )

    #     deleted_versions_map = {}

    #     for canonical_name, new_versions in new_object_to_version_map.items():
    #         current_versions = current_object_to_version_map.get(
    #             canonical_name,
    #             None,
    #         )
    #         if current_versions is None:
    #             continue

    #         deleted_versions = list(set(current_versions).difference(new_versions))
    #         deleted_versions_map[canonical_name] = deleted_versions

    #     return deleted_object_classes, deleted_versions_map

    # def recompute_supported_states(
    #     self,
    #     current_protocol_version: int,
    #     new_object_to_version_map: Dict,
    # ):
    #     current_protocol_state = self.state[str(current_protocol_version)]
    #     deleted_object_classes, deleted_versions_map = self.find_deleted_versions(
    #         current_protocol_state,
    #         new_object_to_version_map=new_object_to_version_map,
    #     )

    #     for _, protocol_state in self.state.items():
    #         object_versions = protocol_state["object_versions"]
    #         if protocol_state["supported"]:
    #             continue

    #         # Check if any object class is deleted,
    #         # then mark the protocol as not supported.
    #         is_unsupported = any(
    #             object_class in object_versions
    #             for object_class in deleted_object_classes
    #         )
    #         if is_unsupported:
    #             protocol_state["supported"] = False
    #             continue

    #         for object_class, supported_versions in deleted_versions_map.items():
    #             available_versions = object_versions.get(object_class, [])
    #             unsupported_versions_present = set(available_versions).intersection(
    #                 supported_versions
    #             )
    #             if unsupported_versions_present:
    #                 is_unsupported = True
    #                 break

    #         if is_unsupported:
    #             protocol_state["supported"] = False

    # @property
    # def state_defined(self):
    #     return len(self.state) > 0

    # @property
    # def latest_version(self):
    #     return int(max(self.state.keys()))

    @staticmethod
    def _hash_to_sha256(obj_dict: Dict) -> str:
        return hashlib.sha256(json.dumps(obj_dict).encode()).hexdigest()

    def build_state(self) -> dict:
        return protocol_state_builder(self.protocol_history)

    def diff(self, state: dict) -> dict:
        return diff_state(state)

    def upgrade(self) -> Result[SyftSuccess, SyftError]:
        state = self.build_state()
        print(">>> got state", state)
        diff = self.diff(state)
        print(">>> got diff", diff)
        current_history = self.protocol_history
        if "dev" not in current_history:
            current_history["dev"] = {}
            current_history["dev"]["object_versions"] = {}
        object_versions = current_history["dev"]["object_versions"]
        for canonical_name, versions in diff.items():
            for version, version_metadata in versions.items():
                if canonical_name not in object_versions:
                    object_versions[canonical_name] = {}
                object_versions[canonical_name][version] = version_metadata

        current_history["dev"]["object_versions"] = object_versions
        self.save_history(current_history)

    # def bump_version(self) -> Result[SyftSuccess, SyftError]:
    #     state = self.build_state()
    #     print(">>> got state", state)
    #     diff = self.diff(state)
    #     print(">>> got diff", diff)
    #     current_history = self.protocol_history
    #     if "dev" not in current_history:
    #         current_history["dev"] = {}
    #         current_history["dev"]["object_versions"] = {}
    #     object_versions = current_history["dev"]["object_versions"]
    #     for canonical_name, versions in diff.items():
    #         for version, version_metadata in versions.items():
    #             if canonical_name not in object_versions:
    #                 object_versions[canonical_name] = {}
    #             object_versions[canonical_name][version] = version_metadata

    #     current_history["dev"]["object_versions"] = object_versions
    #     self.save_history(current_history)

    # def upgrade(self):
    #     object_to_version_map = self.calc_latest_object_versions()
    #     new_protocol_hash = self._hash_to_sha256(object_to_version_map)

    #     if not self.state_defined:
    #         new_protocol_version = 1
    #     else:
    #         # Find the current version
    #         current_protocol_version = self.latest_version

    #         new_protocol_version = int(current_protocol_version) + 1

    #         current_protocol_state = self.state[str(current_protocol_version)]
    #         if current_protocol_state["hash"] == new_protocol_hash:
    #             print("No change in schema. Skipping upgrade.")
    #             return

    #         self.recompute_supported_states(
    #             current_protocol_version=current_protocol_version,
    #             new_object_to_version_map=object_to_version_map,
    #         )

    #     self.state[new_protocol_version] = {
    #         "object_versions": object_to_version_map,
    #         "hash": new_protocol_hash,
    #         "supported": True,
    #     }
    #     self.save_state()
    #     return SyftSuccess(message="Protocol Updated")

    # def validate_current_state(self) -> bool:
    #     current_object_version_map = self.state[self.latest_version]["object_versions"]
    #     inconsistent_versions = []

    #     migration_registry = SyftMigrationRegistry.__migration_version_registry__
    #     for canonical_name in migration_registry:
    #         available_versions = migration_registry[canonical_name]
    #         curr_version_hash_map = current_object_version_map.get(canonical_name, {})
    #         for object_version, fqn in available_versions.items():
    #             object_klass = index_syft_by_module_name(fqn)
    #             object_hash = self._calculate_object_hash(object_klass)
    #             if curr_version_hash_map.get(str(object_version), None) != object_hash:
    #                 inconsistent_versions.append((canonical_name, object_version))

    #     if len(inconsistent_versions) > 0:
    #         raise InConsistentVersionException(
    #             f"Version update is required for the following objects.\n {inconsistent_versions}"
    #         )

    #     return True

    # @property
    # def supported_protocols(self) -> List[int]:
    #     """Returns a list of protocol numbers that are marked as supported."""
    #     return [
    #         int(protocol_version)
    #         for protocol_version, protocol_state in self.state.items()
    #         if str_to_bool(protocol_state["supported"])
    #     ]

    # def get_object_versions(self, protocol: Union[int, str]) -> List:
    #     return self.state[str(protocol)]["object_versions"]


def get_data_protocol():
    return DataProtocol(filename=data_protocol_file_name())


def upgrade_protocol():
    data_protocol = get_data_protocol()
    data_protocol.upgrade()


def migrate_args_and_kwargs(
    args: Tuple,
    kwargs: Dict,
    to_protocol: Optional[int] = None,
    to_latest_protocol: bool = False,
) -> Tuple[Tuple, Dict]:
    """Migrate args and kwargs to latest version for given protocol.

    If `to_protocol` is None, then migrate to latest protocol version.

    """
    data_protocol = get_data_protocol()

    if to_protocol is None:
        to_protocol = data_protocol.latest_version if to_latest_protocol else None

    if to_protocol is None:
        raise SyftException(message="Protocol version missing.")

    # If latest protocol being used is equal to the protocol to be migrate
    # then skip migration of the object
    if to_protocol == data_protocol.latest_version:
        return args, kwargs

    object_versions = data_protocol.get_object_versions(protocol=to_protocol)

    migrated_kwargs, migrated_args = {}, []

    for param_name, param_val in kwargs.items():
        if isinstance(param_val, SyftBaseObject):
            current_version = int(param_val.__version__)
            migrate_to_version = int(max(object_versions[param_val.__canonical_name__]))
            if current_version > migrate_to_version:  # downgrade
                versions = range(current_version - 1, migrate_to_version - 1, -1)
            else:  # upgrade
                versions = range(current_version + 1, migrate_to_version + 1)
            for version in versions:
                param_val = param_val.migrate_to(version)
        migrated_kwargs[param_name] = param_val

    for arg in args:
        if isinstance(arg, SyftBaseObject):
            current_version = int(arg.__version__)
            migrate_to_version = int(max(object_versions[arg.__canonical_name__]))
            if current_version > migrate_to_version:  # downgrade
                versions = range(current_version - 1, migrate_to_version - 1, -1)
            else:  # upgrade
                versions = range(current_version + 1, migrate_to_version + 1)
            for version in versions:
                arg = arg.migrate_to(version)

        migrated_args.append(arg)

    return tuple(migrated_args), migrated_kwargs
