from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from syft_permissions import ACLRequest, AccessLevel, User

from syft_perms.explain import PermissionExplanation, explain

if TYPE_CHECKING:
    from syft_perms.syftperm_context import SyftPermContext


class SyftFolder:
    def __init__(self, path: Path, perm_context: SyftPermContext):
        self.path = path
        self._perm_context = perm_context

    @property
    def _rel_path(self) -> str:
        return str(self.path.relative_to(self._perm_context.datasite))

    def grant_read_access(self, user: str) -> None:
        self._perm_context.modifier.add_permission_for_user(
            self._rel_path, "read", user, is_folder=True
        )

    def grant_write_access(self, user: str) -> None:
        self._perm_context.modifier.add_permission_for_user(
            self._rel_path, "write", user, is_folder=True
        )

    def grant_admin_access(self, user: str) -> None:
        self._perm_context.modifier.add_permission_for_user(
            self._rel_path, "admin", user, is_folder=True
        )

    def revoke_read_access(self, user: str) -> None:
        self._perm_context.modifier.remove_permission_for_user(
            self._rel_path, "read", user, is_folder=True
        )

    def revoke_write_access(self, user: str) -> None:
        self._perm_context.modifier.remove_permission_for_user(
            self._rel_path, "write", user, is_folder=True
        )

    def revoke_admin_access(self, user: str) -> None:
        self._perm_context.modifier.remove_permission_for_user(
            self._rel_path, "admin", user, is_folder=True
        )

    def has_read_access(self, user: str) -> bool:
        return self._check(AccessLevel.READ, user)

    def has_write_access(self, user: str) -> bool:
        return self._check(AccessLevel.WRITE, user)

    def has_admin_access(self, user: str) -> bool:
        return self._check(AccessLevel.ADMIN, user)

    def explain_permissions(self, user: str) -> PermissionExplanation:
        check_path = self._rel_path + "/__any__"
        expl = explain(self._perm_context, check_path, user)
        expl.path = self._rel_path
        return expl

    def _check(self, level: AccessLevel, user: str) -> bool:
        check_path = self._rel_path + "/__any__"
        request = ACLRequest(
            path=check_path,
            level=level,
            user=User(id=user),
        )
        return self._perm_context.service.can_access(request)

    def __repr__(self) -> str:
        return f"SyftFolder({self._rel_path})"
