from pathlib import Path

from syft_permissions import ACLService

from syft_perms.browser import FilesBrowser
from syft_perms.file import SyftFile
from syft_perms.folder import SyftFolder
from syft_perms.syftperm_modifier import PermissionModifier


class SyftPermContext:
    def __init__(self, datasite: str | Path):
        # datasite is Syftbox_folder/email
        self.datasite = Path(datasite)
        self.owner = self.datasite.name
        self.service = ACLService(owner=self.owner)
        self.modifier = PermissionModifier(self)
        self._reload()

    def open(self, path: str | Path) -> SyftFile | SyftFolder:
        """Open a file or folder for permission management.

        The file/folder doesn't need to exist yet.
        Returns SyftFolder if path ends with '/' or is an existing directory.
        """
        if isinstance(path, Path):
            path = str(path)

        abs_path = self.datasite / path
        if path.endswith("/") or abs_path.is_dir():
            return SyftFolder(abs_path, self)
        return SyftFile(abs_path, self)

    @property
    def files(self) -> FilesBrowser:
        return FilesBrowser(self, kind="files")

    @property
    def folders(self) -> FilesBrowser:
        return FilesBrowser(self, kind="folders")

    @property
    def files_and_folders(self) -> FilesBrowser:
        return FilesBrowser(self, kind="all")

    def _reload(self) -> None:
        """Rescan all permission files and rebuild the ACL tree."""
        self.service.load_permissions_from_filesystem(self.datasite)
