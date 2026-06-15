from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from syft_permissions import PERMISSION_FILE_NAME, Access, Rule, RuleSet
from syft_permissions.engine.compiled_rule import (
    USER_PLACEHOLDER,
    _user_in_list,
    compile_rule,
)
from syft_permissions.engine.tree import _relative_path

if TYPE_CHECKING:
    from syft_perms.syftperm_context import SyftPermContext


class PermissionModifier:
    def __init__(self, perm_context: SyftPermContext):
        self._perm_context = perm_context

    def add_permission_for_user(
        self, path_in_datasite: str, level: str, user: str, is_folder: bool = False
    ) -> None:
        """Add a user to a specific access level, resolving the correct yaml."""
        target_yaml_path = self._find_governing_yaml(path_in_datasite, is_folder)
        target_pattern = self._compute_pattern(
            target_yaml_path, path_in_datasite, is_folder
        )
        self._write_user_to_yaml(target_yaml_path, target_pattern, level, user)

    def remove_permission_for_user(
        self, path_in_datasite: str, level: str, user: str, is_folder: bool = False
    ) -> None:
        """Remove a user from a specific access level, resolving the correct yaml."""
        target_yaml_path = self._find_governing_yaml(path_in_datasite, is_folder)
        target_pattern = self._compute_pattern(
            target_yaml_path, path_in_datasite, is_folder
        )
        self._remove_user_from_yaml(target_yaml_path, target_pattern, level, user)

    def remove_all_rules(self, path_in_datasite: str, is_folder: bool = False) -> None:
        """Remove all permission rules for a path, resolving the correct yaml."""
        target_yaml_path = self._find_governing_yaml(path_in_datasite, is_folder)
        target_pattern = self._compute_pattern(
            target_yaml_path, path_in_datasite, is_folder
        )
        self._remove_all_rules_from_yaml(target_yaml_path, target_pattern)

    def copy_permissions(self, path_in_datasite: str, new_abs: Path) -> None:
        """Copy permission entries from old path to new location."""
        node = self._perm_context.service.tree.get_nearest_node(path_in_datasite)
        if not node or not node.ruleset:
            return

        path_relative_to_syftpub = _relative_path(node.path, path_in_datasite)
        rule = _find_matching_rule(node.ruleset.rules, path_relative_to_syftpub)
        if not rule:
            return

        new_yaml = new_abs.parent / PERMISSION_FILE_NAME
        for level in ("read", "write", "admin"):
            for user in getattr(rule.access, level):
                self._write_user_to_yaml(new_yaml, new_abs.name, level, user)

    def _find_governing_yaml(self, path_in_datasite: str, is_folder: bool) -> Path:
        """Find the yaml file that governs this path (nearest node in the ACL tree)."""
        datasite = self._perm_context.datasite
        node = self._perm_context.service.tree.get_nearest_node(path_in_datasite)

        if node is not None and node.ruleset is not None:
            # Use the yaml from the governing node
            if node.path:
                return datasite / node.path / PERMISSION_FILE_NAME
            return datasite / PERMISSION_FILE_NAME

        # No governing yaml exists yet — create one at the target location
        if is_folder:
            return datasite / path_in_datasite / PERMISSION_FILE_NAME
        else:
            parent = str(Path(path_in_datasite).parent)
            # if its the root
            if parent == ".":
                return datasite / PERMISSION_FILE_NAME
            else:
                return datasite / parent / PERMISSION_FILE_NAME

    def _compute_pattern(
        self, yaml_path: Path, path_in_datasite: str, is_folder: bool
    ) -> str:
        """Compute the rule pattern relative to the governing yaml's directory."""
        datasite = self._perm_context.datasite
        # Path of the yaml's directory relative to the datasite
        yaml_dir = yaml_path.parent.relative_to(datasite)
        # Path of the target relative to the yaml's directory
        rel = _relative_path(str(yaml_dir), path_in_datasite)

        if is_folder:
            # Folders need a /** suffix to match all files inside
            return f"{rel}/**" if rel else "**"
        # Files use their path relative to the yaml, or just the filename
        return rel if rel else Path(path_in_datasite).name

    def _write_user_to_yaml(
        self, yaml_path: Path, pattern: str, level: str, user: str
    ) -> None:
        """Low-level: add a user to a specific yaml file and pattern."""
        if yaml_path.exists():
            ruleset = RuleSet.load(yaml_path)
        else:
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            ruleset = RuleSet(path=str(yaml_path.parent))

        rule = _find_or_create_rule(ruleset, pattern)
        access_list = _get_access_list(rule.access, level)

        if user not in access_list:
            access_list.append(user)

        ruleset.save(yaml_path)
        self._perm_context._reload()

    def _remove_user_from_yaml(
        self, yaml_path: Path, pattern: str, level: str, user: str
    ) -> None:
        """Low-level: remove a user from a specific yaml file and pattern."""
        if not yaml_path.exists():
            return

        ruleset = RuleSet.load(yaml_path)

        rule = _find_matching_rule(ruleset.rules, pattern, user)
        if rule is not None:
            access_list = _get_access_list(rule.access, level)
            _remove_user_entry(access_list, user, yaml_path, rule.pattern, level)

        ruleset.save(yaml_path)
        self._perm_context._reload()

    def _remove_all_rules_from_yaml(self, yaml_path: Path, pattern: str) -> None:
        """Low-level: remove all rules with a pattern from a yaml file."""
        if not yaml_path.exists():
            return
        ruleset = RuleSet.load(yaml_path)
        ruleset.rules = [r for r in ruleset.rules if r.pattern != pattern]
        ruleset.save(yaml_path)


def _find_or_create_rule(ruleset: RuleSet, pattern: str) -> Rule:
    """Find an existing rule with the given pattern, or create a new one."""
    for rule in ruleset.rules:
        if rule.pattern == pattern:
            return rule
    rule = Rule(pattern=pattern, access=Access())
    ruleset.rules.append(rule)
    return rule


def _find_matching_rule(
    rules: list[Rule], rel_path: str, user: str = "__probe__"
) -> Rule | None:
    """Return the first rule whose glob pattern matches the relative path."""
    for rule in rules:
        compiled = compile_rule(rule, user)
        if compiled.match(rel_path, user):
            return rule
    return None


def _remove_user_entry(
    access_list: list[str], user: str, yaml_path: Path, pattern: str, level: str
) -> None:
    """Remove all entries that grant access to a user, including wildcards."""
    # we make a copy here so we can remove items from the list while iterating
    for entry in list(access_list):
        if entry == user:
            access_list.remove(entry)
        elif entry == USER_PLACEHOLDER or _user_in_list(user, [entry]):
            access_list.remove(entry)
            warnings.warn(
                f"Removed '{entry}' from {level} in '{pattern}' ({yaml_path.name}) "
                f"while revoking access for {user} — this may affect other users"
            )


def _get_access_list(access: Access, level: str) -> list[str]:
    """Get the mutable list for a given access level."""
    if level == "read":
        return access.read
    elif level == "write":
        return access.write
    elif level == "admin":
        return access.admin
    else:
        raise ValueError(f"Unknown access level: {level}")
