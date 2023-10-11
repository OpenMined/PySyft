# stdlib
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

# third party
from result import OkErr
from result import Result

# relative
from ..serde.recursive import TYPE_BANK
from ..service.response import SyftError
from ..service.response import SyftException
from ..service.response import SyftSuccess
from ..types.syft_object import SyftBaseObject

PROTOCOL_STATE_FILENAME = "protocol_version.json"
PROTOCOL_TYPE = Union[str, int]


def natural_key(key: PROTOCOL_TYPE) -> list[int]:
    """Define key for natural ordering of strings."""
    if isinstance(key, int):
        key = str(key)
    return [int(s) if s.isdigit() else s for s in re.split("(\d+)", key)]


def sort_dict_naturally(d: dict) -> dict:
    """Sort dictionary by keys in natural order."""
    return {k: d[k] for k in sorted(d.keys(), key=natural_key)}


def data_protocol_file_name():
    return PROTOCOL_STATE_FILENAME


def data_protocol_dir():
    return os.path.abspath(str(Path(__file__).parent))


class DataProtocol:
    def __init__(self, filename: str) -> None:
        self.file_path = Path(data_protocol_dir()) / filename
        self.load_state()

    def load_state(self) -> None:
        self.protocol_history = self.read_history()
        self.state = self.build_state()
        self.diff, self.current = self.diff_state(self.state)
        self.protocol_support = self.calculate_supported_protocols()

    @staticmethod
    def _calculate_object_hash(klass: Type[SyftBaseObject]) -> str:
        # TODO: this depends on what is marked as serde
        field_name_keys = sorted(klass.__fields__.keys())
        field_data = {
            field_name: repr(klass.__fields__[field_name].annotation)
            for field_name in field_name_keys
        }
        obj_meta_info = {
            "canonical_name": klass.__canonical_name__,
            "version": klass.__version__,
            "unique_keys": getattr(klass, "__attr_unique__", []),
            "field_data": field_data,
        }

        return hashlib.sha256(json.dumps(obj_meta_info).encode()).hexdigest()

    def read_history(self) -> Dict:
        return json.loads(self.file_path.read_text())

    def save_history(self, history: dict) -> None:
        self.file_path.write_text(json.dumps(history, indent=2) + "\n")

    @property
    def latest_version(self) -> PROTOCOL_TYPE:
        sorted_versions = sorted(self.protocol_history.keys(), key=natural_key)
        if len(sorted_versions) > 0:
            return sorted_versions[-1] if self.has_dev else int(sorted_versions[-1])
        return "dev"

    @staticmethod
    def _hash_to_sha256(obj_dict: Dict) -> str:
        return hashlib.sha256(json.dumps(obj_dict).encode()).hexdigest()

    def build_state(self, stop_key: Optional[str] = None) -> dict:
        sorted_dict = sort_dict_naturally(self.protocol_history)
        state_dict = defaultdict(dict)
        for k, _v in sorted_dict.items():
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
                            f"Can't remove {object_metadata} missing from state {versions} for object {canonical_name}."
                        )
                    if action == "add":
                        state_dict[canonical_name][str(version)] = hash_str
                    elif action == "remove":
                        del state_dict[canonical_name][str(version)]
            # stop early
            if stop_key == k:
                return state_dict
        return state_dict

    def diff_state(self, state: dict) -> tuple[dict, dict]:
        compare_dict = defaultdict(dict)  # what versions are in the latest code
        object_diff = defaultdict(dict)  # diff in latest code with saved json
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
                compare_dict[canonical_name][str(version)] = hash_str

                if canonical_name not in state:
                    # new object so its an add
                    object_diff[canonical_name][str(version)] = {}
                    object_diff[canonical_name][str(version)]["version"] = version
                    object_diff[canonical_name][str(version)]["hash"] = hash_str
                    object_diff[canonical_name][str(version)]["action"] = "add"
                    continue

                versions = state[canonical_name]
                if (
                    str(version) in versions.keys()
                    and versions[str(version)] == hash_str
                ):
                    # already there so do nothing
                    continue
                elif str(version) in versions.keys():
                    raise Exception(
                        f"{canonical_name} for class {cls.__name__} fqn {cls} "
                        + f"version {version} hash has changed. "
                        + f"{hash_str} not in {versions.values()}. "
                        + "Is a unique __canonical_name__ for this subclass missing? "
                        + "If the class has changed you will need to bump the version number."
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
                if str(version) not in versions.keys():
                    # missing so its a remove
                    object_diff[canonical_name][str(version)] = {}
                    object_diff[canonical_name][str(version)]["version"] = version
                    object_diff[canonical_name][str(version)]["hash"] = hash_str
                    object_diff[canonical_name][str(version)]["action"] = "remove"
                    continue
        return object_diff, compare_dict

    def stage_protocol_changes(self) -> Result[SyftSuccess, SyftError]:
        change_count = 0
        current_history = self.protocol_history
        if "dev" not in current_history:
            current_history["dev"] = {}
            current_history["dev"]["object_versions"] = {}
        object_versions = current_history["dev"]["object_versions"]
        for canonical_name, versions in self.diff.items():
            for version, version_metadata in versions.items():
                if canonical_name not in object_versions:
                    object_versions[canonical_name] = {}
                change_count += 1
                object_versions[canonical_name][version] = version_metadata

        current_history["dev"]["object_versions"] = object_versions

        # trim empty dev
        if len(current_history["dev"]["object_versions"]) == 0:
            del current_history["dev"]

        self.save_history(current_history)
        self.load_state()
        return SyftSuccess(message=f"{change_count} Protocol Updates Staged to dev")

    def bump_protocol_version(self) -> Result[SyftSuccess, SyftError]:
        if len(self.diff):
            raise Exception(
                "You can't bump the protocol version with unstaged changes."
            )

        keys = self.protocol_history.keys()
        if "dev" not in keys:
            raise Exception(
                "You can't bump the protocol if there are no staged changes."
            )

        highest_protocol = 0
        for k in self.protocol_history.keys():
            if k == "dev":
                continue
            highest_protocol = max(highest_protocol, int(k))

        next_highest_protocol = highest_protocol + 1
        self.protocol_history[str(next_highest_protocol)] = self.protocol_history["dev"]
        del self.protocol_history["dev"]
        self.save_history(self.protocol_history)
        self.load_state()
        return SyftSuccess(message=f"Protocol Updated to {next_highest_protocol}")

    @property
    def supported_protocols(self) -> list[Union[int, str]]:
        """Returns a list of protocol numbers that are marked as supported."""
        supported = []
        for version, is_supported in self.protocol_support.items():
            if is_supported:
                if version != "dev":
                    version = int(version)
                supported.append(version)
        return supported

    def calculate_supported_protocols(self) -> dict:
        protocol_supported = {}
        # go through each historical protocol version
        for v, version_data in self.protocol_history.items():
            # we assume its supported until we prove otherwise
            protocol_supported[v] = True
            # iterate through each object
            for canonical_name, _ in version_data["object_versions"].items():
                if canonical_name not in self.state:
                    protocol_supported[v] = False
                    break
        return protocol_supported

    def get_object_versions(self, protocol: Union[int, str]) -> list:
        return self.protocol_history[str(protocol)]["object_versions"]

    @property
    def has_dev(self) -> bool:
        if "dev" in self.protocol_history.keys():
            return True
        return False


def get_data_protocol():
    return DataProtocol(filename=data_protocol_file_name())


def stage_protocol_changes() -> Result[SyftSuccess, SyftError]:
    data_protocol = get_data_protocol()
    return data_protocol.stage_protocol_changes()


def bump_protocol_version() -> Result[SyftSuccess, SyftError]:
    data_protocol = get_data_protocol()
    return data_protocol.bump_protocol_version()


def debox_arg_and_migrate(arg: Any, protocol_state: dict):
    """Debox the argument based on whether it is iterable or single entity."""
    box_to_result_type = None

    if type(arg) in OkErr:
        box_to_result_type = type(arg)
        arg = arg.value

    single_entity = False
    is_tuple = isinstance(arg, tuple)

    if isinstance(arg, (list, tuple)):
        iterable_keys = range(len(arg))
        arg = list(arg)
    elif isinstance(arg, dict):
        iterable_keys = arg.keys()
    else:
        iterable_keys = range(1)
        arg = [arg]
        single_entity = True

    for key in iterable_keys:
        _object = arg[key]
        if isinstance(_object, SyftBaseObject):
            current_version = int(_object.__version__)
            migrate_to_version = int(max(protocol_state[_object.__canonical_name__]))
            if current_version > migrate_to_version:  # downgrade
                versions = range(current_version - 1, migrate_to_version - 1, -1)
            else:  # upgrade
                versions = range(current_version + 1, migrate_to_version + 1)
            for version in versions:
                _object = _object.migrate_to(version)
        arg[key] = _object

    wrapped_arg = arg[0] if single_entity else arg
    wrapped_arg = tuple(wrapped_arg) if is_tuple else wrapped_arg
    if box_to_result_type is not None:
        wrapped_arg = box_to_result_type(wrapped_arg)

    return wrapped_arg


def migrate_args_and_kwargs(
    args: Tuple,
    kwargs: Dict,
    to_protocol: Optional[PROTOCOL_TYPE] = None,
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

    protocol_state = data_protocol.build_state(stop_key=str(to_protocol))

    migrated_kwargs, migrated_args = {}, []

    for param_name, param_val in kwargs.items():
        migrated_val = debox_arg_and_migrate(
            arg=param_val,
            protocol_state=protocol_state,
        )
        migrated_kwargs[param_name] = migrated_val

    for arg in args:
        migrated_val = debox_arg_and_migrate(
            arg=arg,
            protocol_state=protocol_state,
        )
        migrated_args.append(migrated_val)

    return tuple(migrated_args), migrated_kwargs
