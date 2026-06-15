from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from syft_permissions import PERMISSION_FILE_NAME

if TYPE_CHECKING:
    from syft_perms.syftperm_context import SyftPermContext
    from syft_perms.file import SyftFile
    from syft_perms.folder import SyftFolder

Kind = Literal["files", "folders", "all"]


class FilesBrowser:
    def __init__(self, perm_context: SyftPermContext, kind: Kind):
        self._perm_context = perm_context
        self._kind = kind

    def all(self) -> list[SyftFile | SyftFolder]:
        return self._files_and_folders()

    def get(self, limit: int, offset: int = 0) -> list[SyftFile | SyftFolder]:
        items = self._files_and_folders()
        return items[offset : offset + limit]

    def search(self, **kwargs: str) -> list[SyftFile | SyftFolder]:
        """Search for files/folders by permission.

        Keyword args: read, write, or admin with a user email as value.
        Example: search(admin="user@example.com")
        """
        query_clauses = kwargs
        items = self._files_and_folders()
        return [
            item
            for item in items
            if all(
                _has_access(item, level, user) for level, user in query_clauses.items()
            )
        ]

    def __getitem__(self, key: slice) -> list[SyftFile | SyftFolder]:
        items = self._files_and_folders()
        return items[key]

    def _files_and_folders(self) -> list[SyftFile | SyftFolder]:
        from syft_perms.file import SyftFile
        from syft_perms.folder import SyftFolder

        root = self._perm_context.datasite
        items: list[SyftFile | SyftFolder] = []

        for p in sorted(root.rglob("*")):
            if p.name == PERMISSION_FILE_NAME:
                continue
            if p.is_dir() and self._kind in ("folders", "all"):
                items.append(SyftFolder(p, self._perm_context))
            elif p.is_file() and self._kind in ("files", "all"):
                items.append(SyftFile(p, self._perm_context))

        return items


def _has_access(item: SyftFile | SyftFolder, level: str, user: str) -> bool:
    if level == "read":
        return item.has_read_access(user)
    elif level == "write":
        return item.has_write_access(user)
    elif level == "admin":
        return item.has_admin_access(user)
    else:
        raise ValueError(f"Unknown access level: {level}")
