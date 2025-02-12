import json
import re
import sqlite3
import traceback
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import wcmatch
import yaml
from loguru import logger
from pydantic import BaseModel, model_validator
from wcmatch.glob import globmatch

from syftbox.lib.constants import PERM_FILE
from syftbox.lib.lib import SyftBoxContext
from syftbox.lib.types import PathLike
from syftbox.server.models.sync_models import RelativePath


# TODO "Client" naming for SDK is confusing, it is a context for the syftbox lib
# util
def issubpath(path1: RelativePath, path2: RelativePath) -> bool:
    return path1 in path2.parents


class PermissionType(Enum):
    CREATE = 1
    READ = 2
    WRITE = 3
    ADMIN = 4


class PermissionParsingError(Exception):
    pass


class PermissionRule(BaseModel):
    dir_path: RelativePath  # where does this permfile live
    path: str  # what paths does it apply to (e.g. **/*.txt)
    user: str  # can be *,
    allow: bool = True
    permissions: List[PermissionType]  # read/write/create/admin
    priority: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PermissionRule):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    @property
    def permfile_path(self) -> Path:
        return self.dir_path / PERM_FILE

    @property
    def depth(self) -> int:
        return len(self.permfile_path.parts)

        # write model validator that accepts either a single string or a list of strings as permissions when initializing

    @model_validator(mode="before")
    @classmethod
    def validate_permissions(cls, values: dict) -> dict:
        # check if values only contains keys that are in the model
        invalid_keys = set(values.keys()) - (set(cls.model_fields.keys()) | set(["type"]))
        if len(invalid_keys) > 0:
            raise PermissionParsingError(
                f"rule yaml contains invalid keys {invalid_keys}, only {cls.model_fields.keys()} are allowed"
            )

        # add that if the type value is "disallow" we set allow to false
        if values.get("type") == "disallow":
            values["allow"] = False

        # if path refers to a location higher in the directory tree than the current file, raise an error
        path = values.get("path")
        if path and path.startswith("../"):
            raise PermissionParsingError(
                f"path {path} refers to a location higher in the directory tree than the current file"
            )

        # if user is not a valid email, or *, raise an error
        email = values.get("user", "")
        is_valid_email = re.match(r"[^@]+@[^@]+", email or "")
        if email != "*" and not is_valid_email:
            raise PermissionParsingError(f"user {values.get('user')} is not a valid email or *")

        # listify permissions
        perms = values.get("permissions")
        if isinstance(perms, str):
            perms = [perms]
        if isinstance(perms, list):
            values["permissions"] = [PermissionType[p.upper()] if isinstance(p, str) else p for p in perms]
        else:
            raise ValueError(f"permissions should be a list of strings or a single string, received {type(perms)}")

        path = values.get("path")
        if path and "**" in path and "{useremail}" in path and path.index("**") < path.rindex("{useremail}"):
            # this would make creating the path2rule mapping more challenging to compute beforehand
            raise PermissionParsingError("** can never be after {useremail}")

        return values

    @classmethod
    def from_rule_dict(cls, dir_path: RelativePath, rule_dict: dict, priority: int) -> "PermissionRule":
        # initialize from dict
        return cls(dir_path=dir_path, **rule_dict, priority=priority)

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "PermissionRule":
        """Create a PermissionRule from a database row"""
        permissions = []
        if row["can_read"]:
            permissions.append(PermissionType.READ)
        if row["can_create"]:
            permissions.append(PermissionType.CREATE)
        if row["can_write"]:
            permissions.append(PermissionType.WRITE)
        if row["admin"]:
            permissions.append(PermissionType.ADMIN)

        return cls(
            dir_path=Path(row["permfile_path"]).parent,
            path=row["path"],
            user=row["user"],  # Default to all users since DB schema doesn't show user field
            allow=not row["disallow"],
            priority=row["priority"],
            permissions=permissions,
        )

    def to_db_row(self) -> dict:
        """Convert PermissionRule to a database row dictionary"""
        return {
            "permfile_path": str(self.permfile_path),  # Reconstruct full path
            "permfile_dir": str(self.dir_path),
            "permfile_depth": self.depth,
            "priority": self.priority,
            "path": self.path,
            "user": self.user,
            "can_read": PermissionType.READ in self.permissions,
            "can_create": PermissionType.CREATE in self.permissions,
            "can_write": PermissionType.WRITE in self.permissions,
            "admin": PermissionType.ADMIN in self.permissions,
            "disallow": not self.allow,
        }

    @property
    def permission_dict(self) -> dict:
        return {
            "read": PermissionType.READ in self.permissions,
            "create": PermissionType.CREATE in self.permissions,
            "write": PermissionType.WRITE in self.permissions,
            "admin": PermissionType.ADMIN in self.permissions,
        }

    def as_file_json(self) -> dict:
        res = {
            "path": self.path,
            "user": self.user,
            "permissions": [p.name.lower() for p in self.permissions],
        }
        if not self.allow:
            res["type"] = "disallow"
        return res

    def filepath_matches_rule_path(self, filepath: Path) -> Tuple[bool, Optional[str]]:
        if issubpath(self.dir_path, filepath):
            relative_file_path = filepath.relative_to(self.dir_path)
        else:
            return False, None

        match_for_email = None
        if self.has_email_template:
            match = False
            emails_in_file_path = [
                part for part in str(relative_file_path).split("/") if "@" in part
            ]  # todo: improve this
            for email in emails_in_file_path:
                if globmatch(
                    str(relative_file_path),
                    self.path.replace("{useremail}", email),
                    flags=wcmatch.glob.GLOBSTAR,
                ):
                    match = True
                    match_for_email = email
                    break
        else:
            match = globmatch(str(relative_file_path), self.path, flags=wcmatch.glob.GLOBSTAR)
        return match, match_for_email

    @property
    def has_email_template(self) -> bool:
        return "{useremail}" in self.path

    def resolve_path_pattern(self, email: str) -> str:
        return self.path.replace("{useremail}", email)


class SyftPermission(BaseModel):
    relative_filepath: RelativePath
    rules: List[PermissionRule]

    def save(self, path: Path) -> None:
        if path.is_dir():
            path = path / PERM_FILE
        with open(path, "w") as f:
            yaml.dump([x.as_file_json() for x in self.rules], f)

    def ensure(self, path: Path) -> bool:
        """For backwards compatibility, we ensure that the permission file exists with these permissions"""
        self.save(path)
        return True

    @property
    def depth(self) -> int:
        return len(self.relative_filepath.parts)

    def to_dict(self) -> list[dict]:
        return [x.as_file_json() for x in self.rules]

    @staticmethod
    def is_permission_file(path: Path) -> bool:
        return path.name == PERM_FILE

    @classmethod
    def is_valid(cls, path: Path, datasite_path: Path, _print: bool = True) -> bool:
        try:
            cls.from_file(path, datasite_path)
            return True
        except Exception as e:
            if _print:
                print(f"Invalid permission file {path}: {e}\n{traceback.format_exc()}")
            return False

    @classmethod
    def create(cls, context: SyftBoxContext, dir: Path) -> "SyftPermission":
        if not dir.is_absolute():
            raise ValueError("dir must be an absolute")

        if dir.exists() and dir.is_file():
            raise ValueError("dir must be a directory")

        dir.mkdir(parents=True, exist_ok=True)
        file_path = dir / PERM_FILE

        try:
            relative_path = file_path.relative_to(context.workspace.datasites)
        except ValueError:
            raise ValueError("dir must be inside the datasites folder")
        return cls(relative_filepath=relative_path, rules=[])

    @classmethod
    def datasite_default(cls, context: SyftBoxContext, dir: Path) -> "SyftPermission":
        perm = cls.create(context, dir)
        perm.add_rule(
            path="**",
            user=context.email,
            permission=["admin", "create", "write", "read"],
        )
        return perm

    @classmethod
    def mine_with_public_read(cls, context: SyftBoxContext, dir: Path) -> "SyftPermission":
        perm = cls.create(context, dir)
        perm.add_rule(path="**", user=context.email, permission=["admin"])
        perm.add_rule(path="**", user="*", permission=["read"])
        return perm

    @classmethod
    def mine_with_public_write(cls, context: SyftBoxContext, dir: Path) -> "SyftPermission":
        # for backwards compatibility
        return cls.mine_with_public_rw(context, dir)

    @classmethod
    def mine_with_public_rw(cls, context: SyftBoxContext, dir: Path) -> "SyftPermission":
        perm = cls.create(context, dir)
        perm.add_rule(path="**", user=context.email, permission=["admin"])
        perm.add_rule(path="**", user="*", permission=["write", "read"])
        return perm

    def add_rule(
        self, path: str, user: str, permission: Union[list[str], list[PermissionType]], allow: bool = True
    ) -> None:
        priority = len(self.rules)
        if isinstance(permission, list) and isinstance(permission[0], PermissionType):
            permission = [PermissionType[p.upper()] for p in permission if isinstance(p, str)]
        rule = PermissionRule(
            dir_path=self.dir_path, path=path, user=user, allow=allow, permissions=permission, priority=priority
        )
        self.rules.append(rule)

    @property
    def dir_path(self) -> Path:
        return self.relative_filepath.parent

    @classmethod
    def from_file(cls, path: Path, datasite_path: Path) -> "SyftPermission":
        with open(path, "r") as f:
            rule_dicts = yaml.safe_load(f)
            relative_path = path.relative_to(datasite_path)
            return cls.from_rule_dicts(relative_path, rule_dicts)

    @classmethod
    def from_rule_dicts(cls, permfile_file_path: PathLike, rule_dicts: list[dict]) -> "SyftPermission":
        if not isinstance(rule_dicts, list):
            raise ValueError(f"rules should be passed as a list of dicts, received {type(rule_dicts)}")
        rules = []
        dir_path = Path(permfile_file_path).parent
        for i, rule_dict in enumerate(rule_dicts):
            rule = PermissionRule.from_rule_dict(dir_path, rule_dict, priority=i)
            rules.append(rule)
        return cls(relative_filepath=permfile_file_path, rules=rules)

    @classmethod
    def from_string(cls, s: str, path: PathLike) -> "SyftPermission":
        dicts = yaml.safe_load(s)
        return cls.from_rule_dicts(Path(path), dicts)

    @classmethod
    def from_bytes(cls, b: bytes, path: PathLike) -> "SyftPermission":
        return cls.from_string(b.decode("utf-8"), path)


class ComputedPermission(BaseModel):
    user: str
    file_path: RelativePath

    perms: dict[PermissionType, bool] = {
        PermissionType.READ: False,
        PermissionType.CREATE: False,
        PermissionType.WRITE: False,
        PermissionType.ADMIN: False,
    }

    @classmethod
    def from_user_rules_and_path(cls, rules: List[PermissionRule], user: str, path: Path) -> "ComputedPermission":
        permission = cls(user=user, file_path=path)
        for rule in rules:
            permission.apply(rule)
        return permission

    @property
    def path_owner(self) -> str:
        """owner of the datasite for this path"""
        return str(self.file_path).split("/", 1)[0]

    def has_permission(self, permtype: PermissionType) -> bool:
        # exception for owners: they can always read and write to their own datasite
        if self.path_owner == self.user:
            return True
        # exception for admins: they can do anything for this path
        if self.perms[PermissionType.ADMIN]:
            return True
        # exception for permfiles: any modifications to permfiles are only allowed for admins
        if self.file_path.name == PERM_FILE and permtype in [PermissionType.CREATE, PermissionType.WRITE]:
            return self.perms[PermissionType.ADMIN]
        # exception for read/write, they are only allowed if read is also allowed
        if permtype in [PermissionType.CREATE, PermissionType.WRITE]:
            return self.perms[PermissionType.READ] and self.perms[permtype]
        # default case
        return self.perms[permtype]

    def user_matches(self, rule: PermissionRule) -> bool:
        """Computes if the user in the rule"""
        if rule.user == "*":
            return True
        elif rule.user == self.user:
            return True
        else:
            return False

    def rule_applies_to_path(self, rule: PermissionRule) -> bool:
        if rule.has_email_template:
            # we fill in a/b/{useremail}/*.txt -> a/b/user@email.org/*.txt
            resolved_path_pattern = rule.resolve_path_pattern(self.user)
        else:
            resolved_path_pattern = rule.path

        # target file path (the one that we want to check permissions for relative to the syftperm file
        # we need this because the syftperm file specifies path patterns relative to its own location

        if issubpath(rule.dir_path, self.file_path):
            relative_file_path = self.file_path.relative_to(rule.dir_path)
            return globmatch(relative_file_path, resolved_path_pattern, flags=wcmatch.glob.GLOBSTAR)
        else:
            return False

    def is_invalid_permission(self, permtype: PermissionType) -> bool:
        return self.file_path.name == PERM_FILE and permtype in [PermissionType.CREATE, PermissionType.WRITE]

    def apply(self, rule: PermissionRule) -> None:
        if self.user_matches(rule) and self.rule_applies_to_path(rule):
            for permtype in rule.permissions:
                if self.is_invalid_permission(permtype):
                    continue
                self.perms[permtype] = rule.allow


# migration code, can be deleted after prod migration is done


def map_email_to_permissions(json_data: dict) -> dict:
    email_permissions = defaultdict(list)
    for permission, emails in json_data.items():
        if permission not in ["read", "write", "create", "admin"]:
            continue
        for email in emails:
            if email is None:
                continue
            email_permissions[email].append(permission)
    return email_permissions


def convert_permission(old_perm_dict: dict) -> list:
    old_perm_dict.pop("filepath", None)  # not needed, we use the actual path of the perm file

    user_permissions = map_email_to_permissions(old_perm_dict)
    output = []

    for email in user_permissions:
        new_perm_dict = {
            "permissions": user_permissions[email],
            "path": "**",
            "user": (email if email != "GLOBAL" else "*"),  # "*" is a wildcard for all users
        }
        output.append(new_perm_dict)

    return output


def migrate_permissions(snapshot_folder: Path) -> None:
    """
    Migrate all `_.syftperm` files from old format to new format within a given snapshot folder.
    This function:
    - searches for files with the extension '_.syftperm' in the specified snapshot folder.
    -  converts their content from JSON to YAML format
    - writes the converted content to new files with the name 'syftperm.yaml' in the same path
    - deletes the original `_.syftperm` files

    Args:
        snapshot_folder (str): The path to the snapshot folder containing the permission files.
    Returns:
        None
    """
    old_syftperm_filename = "_.syftperm"
    files = list(snapshot_folder.rglob(old_syftperm_filename))
    for file in files:
        old_data = json.loads(file.read_text())
        try:
            new_data = convert_permission(old_data)
        except Exception as e:
            logger.error(f"Failed to migrate old permission: {old_data}, {e} skipping")
            continue
        new_file_path = file.with_name(file.name.replace(old_syftperm_filename, PERM_FILE))
        logger.info(
            f"migrating permission from {file.relative_to(snapshot_folder)} to {new_file_path.relative_to(snapshot_folder)}"
        )
        new_file_path.write_text(yaml.dump(new_data))
        file.unlink()
