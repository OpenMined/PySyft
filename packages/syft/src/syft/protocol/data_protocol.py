# stdlib
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from functools import cache
import hashlib
import json
from operator import itemgetter
import os
from pathlib import Path
import re
from types import UnionType
import typing
from typing import Any
import warnings

# third party
from packaging.version import parse

# syft absolute
from syft.types.result import Err
from syft.types.result import Ok
from syft.util.util import get_dev_mode

# relative
from .. import __version__
from ..service.response import SyftSuccess
from ..types.dicttuple import DictTuple
from ..types.errors import SyftException
from ..types.syft_object import SyftBaseObject
from ..types.syft_object_registry import SyftObjectRegistry

PROTOCOL_STATE_FILENAME = "protocol_version.json"
PROTOCOL_TYPE = str | int

IGNORE_TYPES = [
    "mock_type",
    "MockWrapper",
    "base_stash_mock_object_type",
    "MockObjectFromSyftBaseObj",
    "MockObjectToSyftBaseObj",
]


def natural_key(key: PROTOCOL_TYPE) -> list[int | str | Any]:
    """Define key for natural ordering of strings."""
    if isinstance(key, int):
        key = str(key)
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", key)]


def sort_dict_naturally(d: dict) -> dict:
    """Sort dictionary by keys in natural order."""
    return {k: d[k] for k in sorted(d.keys(), key=natural_key)}


def data_protocol_file_name() -> str:
    return PROTOCOL_STATE_FILENAME


def data_protocol_dir() -> Path:
    return Path(os.path.abspath(str(Path(__file__).parent)))


def protocol_release_dir() -> Path:
    return data_protocol_dir() / "releases"


def handle_union_type_klass_name(type_klass_name: str) -> str:
    if type_klass_name == typing.Union.__name__:
        return UnionType.__name__
    return type_klass_name


def handle_annotation_repr_(annotation: type) -> str:
    """Handle typing representation."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    def get_annotation_repr_for_arg(arg: type) -> str:
        if hasattr(arg, "__canonical_name__"):
            return arg.__canonical_name__
        return getattr(arg, "__name__", str(arg))

    if origin and args:
        args_repr = ", ".join(get_annotation_repr_for_arg(arg) for arg in args)
        origin_repr = getattr(origin, "__name__", str(origin))

        # Handle typing.Union and types.UnionType
        origin_repr = handle_union_type_klass_name(origin_repr)
        return f"{origin_repr}: [{args_repr}]"
    elif args:
        args_repr = ", ".join(
            getattr(arg, "__name__", str(arg)) for arg in sorted(args)
        )
        return args_repr
    else:
        return repr(annotation)


class DataProtocol:
    def __init__(self, filename: str, raise_exception: bool = False) -> None:
        self.file_path = data_protocol_dir() / filename
        self.raise_exception = raise_exception
        self.load_state()

    def load_state(self) -> None:
        self.protocol_history = self.read_history()
        self.state = self.build_state()
        self.diff, self.current = self.diff_state(self.state)
        self.protocol_support = self.calculate_supported_protocols()

    @staticmethod
    def _calculate_object_hash(klass: type[SyftBaseObject]) -> str:
        # TODO: this depends on what is marked as serde

        # Rebuild the model to ensure that the fields are up to date
        # and any ForwardRef are resolved
        klass.model_rebuild()
        field_data = {
            field: handle_annotation_repr_(field_info.rebuild_annotation())
            for field, field_info in sorted(
                klass.model_fields.items(), key=itemgetter(0)
            )
        }
        obj_meta_info = {
            "canonical_name": klass.__canonical_name__,
            "version": klass.__version__,
            "unique_keys": getattr(klass, "__attr_unique__", []),
            "field_data": field_data,
        }

        return hashlib.sha256(json.dumps(obj_meta_info).encode()).hexdigest()

    @staticmethod
    def read_json(file_path: Path) -> dict:
        try:
            return json.loads(file_path.read_text())
        except Exception:
            return {}

    def read_history(self) -> dict:
        protocol_history = self.read_json(self.file_path)

        for version in protocol_history.keys():
            if version == "dev":
                continue
            release_version_path = (
                protocol_release_dir() / protocol_history[version]["release_name"]
            )
            released_version = self.read_json(file_path=release_version_path)
            protocol_history[version] = released_version.get(version, {})

        return protocol_history

    def save_history(self, history: dict) -> None:
        if os.path.exists(protocol_release_dir()):
            for file_path in protocol_release_dir().iterdir():
                for version in self.read_json(file_path):
                    # Skip adding file if the version is not part of the history
                    if version not in history.keys():
                        continue
                    history[version] = {"release_name": file_path.name}
        self.file_path.write_text(json.dumps(history, indent=2) + "\n")

    @property
    def latest_version(self) -> PROTOCOL_TYPE:
        sorted_versions = sorted(self.protocol_history.keys(), key=natural_key)
        if len(sorted_versions) > 0:
            return sorted_versions[-1] if self.has_dev else int(sorted_versions[-1])
        return "dev"

    @staticmethod
    def _hash_to_sha256(obj_dict: dict) -> str:
        return hashlib.sha256(json.dumps(obj_dict).encode()).hexdigest()

    def build_state(self, stop_key: str | None = None) -> dict:
        sorted_dict = sort_dict_naturally(self.protocol_history)
        state_dict: dict = defaultdict(dict)
        for protocol_number in sorted_dict:
            object_versions = sorted_dict[protocol_number]["object_versions"]
            for canonical_name, versions in object_versions.items():
                for version, object_metadata in versions.items():
                    action = object_metadata["action"]
                    version = object_metadata["version"]
                    hash_str = object_metadata["hash"]
                    state_versions = state_dict[canonical_name]
                    state_version_hashes = [val[0] for val in state_versions.values()]
                    if action == "add" and (
                        str(version) in state_versions.keys()
                        or hash_str in state_version_hashes
                    ):
                        raise Exception(
                            f"Can't add {object_metadata} for protocol {protocol_number} already in state {versions}"
                        )
                    if action == "remove" and (
                        str(version) not in state_versions.keys()
                        and hash_str not in state_version_hashes
                    ):
                        raise Exception(
                            f"Can't remove {object_metadata} missing from state {versions} for object {canonical_name}."
                        )
                    if action == "add":
                        state_dict[canonical_name][str(version)] = (
                            hash_str,
                            protocol_number,
                        )
                    elif action == "remove":
                        del state_dict[canonical_name][str(version)]
            # stop early
            if stop_key == protocol_number:
                return state_dict
        return state_dict

    @staticmethod
    def obj_json(version: str | int, _hash: str, action: str = "add") -> dict:
        return {
            "version": int(version),
            "hash": _hash,
            "action": action,
        }

    def diff_state(self, state: dict) -> tuple[dict, dict]:
        compare_dict: dict = defaultdict(dict)  # what versions are in the latest code
        object_diff: dict = defaultdict(dict)  # diff in latest code with saved json
        all_serde_propeties = [
            serde_properties
            for version_dict in SyftObjectRegistry.__object_serialization_registry__.values()
            for serde_properties in version_dict.values()
        ]
        for serde_properties in all_serde_propeties:
            cls, version = serde_properties[7], serde_properties[9]
            if issubclass(cls, SyftBaseObject):
                canonical_name = cls.__canonical_name__
                if canonical_name in IGNORE_TYPES:
                    continue

                hash_str = DataProtocol._calculate_object_hash(cls)

                # build this up for later
                compare_dict[canonical_name][str(version)] = hash_str

                if canonical_name not in state:
                    # new object so its an add
                    obj_to_add = self.obj_json(int(version), hash_str)
                    object_diff[canonical_name][str(version)] = obj_to_add
                    continue

                versions = state[canonical_name]
                if (
                    str(version) in versions.keys()
                    and versions[str(version)][0] == hash_str
                ):
                    # already there so do nothing
                    continue
                elif str(version) in versions.keys():
                    is_protocol_dev = versions[str(version)][1] == "dev"
                    if is_protocol_dev:
                        # force overwrite existing object so its an add
                        obj_to_add = self.obj_json(int(version), hash_str)
                        object_diff[canonical_name][str(version)] = obj_to_add
                        continue
                    error_msg = f"""{canonical_name} for class {cls.__name__} fqn {cls}\
version {version} hash has changed. {hash_str} not in {versions.values()}. \
Is a unique __canonical_name__ for this subclass missing?
If the class has changed you will need to define a new class with the changes, \
with same __canonical_name__ and bump the __version__ number. {cls.model_fields}
"""

                    if get_dev_mode() or self.raise_exception:
                        raise Exception(error_msg)
                    else:
                        warnings.warn(error_msg, stacklevel=1, category=UserWarning)
                        break
                else:
                    # new object so its an add
                    obj_to_add = self.obj_json(int(version), hash_str)
                    object_diff[canonical_name][str(version)] = obj_to_add
                    continue

        # now check for remove actions
        for canonical_name in state:
            for version, (hash_str, _) in state[canonical_name].items():
                if canonical_name not in compare_dict:
                    # missing so its a remove
                    obj_to_remove = self.obj_json(int(version), hash_str, "remove")
                    object_diff[canonical_name][str(version)] = obj_to_remove
                    continue
                versions = compare_dict[canonical_name]
                if str(version) not in versions.keys():
                    # missing so its a remove
                    obj_to_remove = self.obj_json(int(version), hash_str, "remove")
                    object_diff[canonical_name][str(version)] = obj_to_remove
                    continue
        return object_diff, compare_dict

    def stage_protocol_changes(self) -> SyftSuccess:
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
                action = version_metadata["action"]

                # Allow removal of class that only been staged to dev
                if (
                    action == "remove"
                    and str(version) in object_versions[canonical_name]
                ):
                    # Delete the whole class if only single version exists
                    if len(object_versions[canonical_name]) == 1:
                        del object_versions[canonical_name]
                    else:
                        # In case of multiple versions of the class only delete the selected
                        del object_versions[canonical_name][str(version)]

                else:  # Add or overwrite existing data in dev
                    object_versions[canonical_name][str(version)] = version_metadata

            # Sort the version dict
            object_versions[canonical_name] = sort_dict_naturally(
                object_versions.get(canonical_name, {})
            )

        current_history["dev"]["object_versions"] = object_versions

        # trim empty dev
        if len(current_history["dev"]["object_versions"]) == 0:
            del current_history["dev"]

        self.save_history(current_history)
        self.load_state()
        return SyftSuccess(message=f"{change_count} Protocol Updates Staged to dev")

    def bump_protocol_version(self) -> SyftSuccess:
        if len(self.diff):
            raise SyftException(
                public_message="You can't bump the protocol version with unstaged changes."
            )

        keys = self.protocol_history.keys()
        if "dev" not in keys:
            self.validate_release()
            print("You can't bump the protocol if there are no staged changes.")
            raise SyftException(
                public_message="Failed to bump version as there are no staged changes."
            )

        highest_protocol = 0
        for k in self.protocol_history.keys():
            if k == "dev":
                continue
            highest_protocol = max(highest_protocol, int(k))

        next_highest_protocol = highest_protocol + 1
        self.protocol_history[str(next_highest_protocol)] = self.protocol_history["dev"]
        self.freeze_release(self.protocol_history, str(next_highest_protocol))
        del self.protocol_history["dev"]
        self.save_history(self.protocol_history)
        self.load_state()
        return SyftSuccess(message=f"Protocol Updated to {next_highest_protocol}")

    @staticmethod
    def freeze_release(protocol_history: dict, latest_protocol: str) -> None:
        """Freezes latest release as a separate release file."""

        # Get release history
        release_history = protocol_history[latest_protocol]

        # Create new file for the version
        syft_version = parse(__version__)
        release_file_name = f"{syft_version.public}.json"
        release_file = protocol_release_dir() / release_file_name

        # Save the new released version
        release_file.write_text(
            json.dumps({latest_protocol: release_history}, indent=2)
        )

    def validate_release(self) -> None:
        """Validate if latest release name is consistent with syft version"""
        # Read the protocol history
        protocol_history = self.read_json(self.file_path)
        sorted_protocol_versions = sorted(protocol_history.keys(), key=natural_key)

        # Grab the latest protocol
        latest_protocol = (
            sorted_protocol_versions[-1] if len(sorted_protocol_versions) > 0 else None
        )

        # Skip validation if latest protocol is dev
        if latest_protocol is None or latest_protocol == "dev":
            return

        # Get filename of the latest protocol
        release_name = protocol_history[latest_protocol]["release_name"]
        # Extract syft version from release name
        protocol_syft_version = parse(release_name.split(".json")[0])
        current_syft_version = parse(__version__)

        # If base syft version in latest protocol version is not same as current syft version
        # Skip updating the release name
        if protocol_syft_version.base_version != current_syft_version.base_version:
            return

        # Update release name to latest beta, stable or post based on current syft version
        print(
            f"Current release {release_name} will be updated to {current_syft_version}"
        )

        # Get latest protocol file path
        latest_protocol_fp: Path = protocol_release_dir() / release_name

        # New protocol file path
        new_protocol_file_path = (
            protocol_release_dir() / f"{current_syft_version.public}.json"
        )

        # Update older file path to newer file path
        latest_protocol_fp.rename(new_protocol_file_path)
        protocol_history[latest_protocol]["release_name"] = (
            f"{current_syft_version}.json"
        )

        # Save history
        self.file_path.write_text(json.dumps(protocol_history, indent=2) + "\n")

        # Reload protocol
        self.read_history()

    def revert_latest_protocol(self) -> None:
        """Revert latest protocol changes to dev"""

        # Get current protocol history
        protocol_history = self.read_json(self.file_path)

        # Get latest released protocol
        sorted_protocol_versions = sorted(protocol_history.keys(), key=natural_key)
        latest_protocol = (
            sorted_protocol_versions[-1] if len(sorted_protocol_versions) > 0 else None
        )

        # If current protocol is dev, skip revert
        if latest_protocol is None or latest_protocol == "dev":
            raise SyftException(
                public_message="Revert skipped !! Already running dev protocol."
            )

        # Read the current released protocol
        release_name = protocol_history[latest_protocol]["release_name"]
        protocol_file_path: Path = protocol_release_dir() / release_name

        released_protocol = self.read_json(protocol_file_path)
        protocol_history["dev"] = released_protocol[latest_protocol]

        # Delete the current released protocol
        protocol_history.pop(latest_protocol)
        protocol_file_path.unlink()

        # Save history
        self.save_history(protocol_history)
        self.load_state()

    def check_protocol(self) -> SyftSuccess:
        if len(self.diff) != 0:
            raise SyftException(public_message="Protocol Changes Unstaged")
        else:
            return SyftSuccess(message="Protocol Stable")

    def check_or_stage_protocol(self) -> SyftSuccess:
        if not self.check_protocol():
            self.stage_protocol_changes()
        result = self.check_protocol()
        return result

    @property
    def supported_protocols(self) -> list[int | str]:
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
            for canonical_name in version_data["object_versions"].keys():
                if canonical_name not in self.state:
                    protocol_supported[v] = False
                    break
        return protocol_supported

    def get_object_versions(self, protocol: int | str) -> list:
        return self.protocol_history[str(protocol)]["object_versions"]

    @property
    def has_dev(self) -> bool:
        if "dev" in self.protocol_history.keys():
            return True
        return False

    def reset_dev_protocol(self) -> None:
        if self.has_dev:
            del self.protocol_history["dev"]
            self.save_history(self.protocol_history)


def get_data_protocol(raise_exception: bool = False) -> DataProtocol:
    return _get_data_protocol(
        filename=data_protocol_file_name(),
        raise_exception=raise_exception,
    )


@cache
def _get_data_protocol(filename: str, raise_exception: bool = False) -> DataProtocol:
    return DataProtocol(
        filename=filename,
        raise_exception=raise_exception,
    )


def stage_protocol_changes() -> SyftSuccess:
    data_protocol = get_data_protocol(raise_exception=True)
    return data_protocol.stage_protocol_changes()


def bump_protocol_version() -> SyftSuccess:
    data_protocol = get_data_protocol(raise_exception=True)
    return data_protocol.bump_protocol_version()


def check_or_stage_protocol() -> SyftSuccess:
    data_protocol = get_data_protocol()
    return data_protocol.check_or_stage_protocol()


def debox_arg_and_migrate(arg: Any, protocol_state: dict) -> Any:
    """Debox the argument based on whether it is iterable or single entity."""
    constructor = None
    extra_args = []

    single_entity = False

    if isinstance(arg, Ok) or isinstance(arg, Err):
        constructor = type(arg)
        arg = arg.value

    if isinstance(arg, MutableMapping):
        iterable_keys: Iterable = arg.keys()
    elif isinstance(arg, MutableSequence):
        iterable_keys = range(len(arg))
    elif isinstance(arg, tuple):
        iterable_keys = range(len(arg))
        constructor = type(arg)
        if isinstance(arg, DictTuple):
            extra_args.append(arg.keys())
        arg = list(arg)
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
    if constructor is not None:
        wrapped_arg = constructor(wrapped_arg, *extra_args)

    return wrapped_arg


def migrate_args_and_kwargs(
    args: tuple,
    kwargs: dict,
    to_protocol: PROTOCOL_TYPE | None = None,
    to_latest_protocol: bool = False,
) -> tuple[tuple, dict]:
    """Migrate args and kwargs to latest version for given protocol.

    If `to_protocol` is None, then migrate to latest protocol version.

    """
    data_protocol = get_data_protocol()

    if to_protocol is None:
        to_protocol = data_protocol.latest_version if to_latest_protocol else None

    if to_protocol is None:
        raise SyftException("Protocol version missing.")

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
