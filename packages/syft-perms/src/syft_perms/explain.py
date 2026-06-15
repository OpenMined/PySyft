from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from syft_permissions import PERMISSION_FILE_NAME, ACLRequest, AccessLevel, User
from syft_permissions.engine.compiled_rule import ACLRule

if TYPE_CHECKING:
    from syft_perms.syftperm_context import SyftPermContext


@dataclass
class PermissionExplanation:
    path: str
    user: str
    is_owner: bool
    governing_yaml: str | None
    matched_rule: str | None
    read: bool
    write: bool
    admin: bool
    reasons: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        lines = [
            f"Permissions for '{self.path}' (user: {self.user})",
            f"  Owner: {self.is_owner}",
            f"  Read:  {self.read} — {self.reasons.get('read', '')}",
            f"  Write: {self.write} — {self.reasons.get('write', '')}",
            f"  Admin: {self.admin} — {self.reasons.get('admin', '')}",
        ]
        if self.governing_yaml:
            lines.append(f"  Governing file: {self.governing_yaml}")
        if self.matched_rule:
            lines.append(f"  Matched rule: {self.matched_rule}")
        return "\n".join(lines)


def explain(perm: SyftPermContext, rel_path: str, user: str) -> PermissionExplanation:
    """Build a PermissionExplanation for a given path and user."""
    if user == perm.service.owner:
        return _explain_owner(rel_path, user)

    node = perm.service.tree.get_nearest_node(rel_path)
    if node is None or node.ruleset is None:
        return _explain_denied(rel_path, user, "No permission file found")

    # NOTE: we are using AccessLevel.READ here because it cannot be empty but its not used
    compiled_rule = perm.service.tree.get_compiled_rule(
        ACLRequest(path=rel_path, user=User(id=user), level=AccessLevel.READ)
    )
    governing_yaml = str(Path(node.path) / PERMISSION_FILE_NAME)
    if compiled_rule is not None:
        return _explain_matched(
            rel_path, user, governing_yaml, compiled_rule.pattern, compiled_rule
        )

    return _explain_denied(
        rel_path, user, "No matching rule in permission file", governing_yaml
    )


def _explain_owner(rel_path: str, user: str) -> PermissionExplanation:
    reason = "Owner of datasite"
    return PermissionExplanation(
        path=rel_path,
        user=user,
        is_owner=True,
        governing_yaml=None,
        matched_rule=None,
        read=True,
        write=True,
        admin=True,
        reasons={"read": reason, "write": reason, "admin": reason},
    )


def _explain_denied(
    rel_path: str,
    user: str,
    reason: str,
    governing_yaml: str | None = None,
) -> PermissionExplanation:
    return PermissionExplanation(
        path=rel_path,
        user=user,
        is_owner=False,
        governing_yaml=governing_yaml,
        matched_rule=None,
        read=False,
        write=False,
        admin=False,
        reasons={"read": reason, "write": reason, "admin": reason},
    )


def _explain_matched(
    rel_path: str,
    user: str,
    governing_yaml: str,
    pattern: str,
    compiled: ACLRule,
) -> PermissionExplanation:
    read = compiled.has_read(user)
    write = compiled.has_write(user)
    admin = compiled.has_admin(user)
    reasons = _build_reasons(pattern, read=read, write=write, admin=admin)
    return PermissionExplanation(
        path=rel_path,
        user=user,
        is_owner=False,
        governing_yaml=governing_yaml,
        matched_rule=pattern,
        read=read,
        write=write,
        admin=admin,
        reasons=reasons,
    )


def _build_reasons(pattern: str, **levels: bool) -> dict[str, str]:
    reasons = {}
    for level_name, has_it in levels.items():
        if has_it:
            reasons[level_name] = f"Granted via '{pattern}' in {level_name} list"
        else:
            reasons[level_name] = f"Not in {level_name} list for '{pattern}'"
    return reasons
