import sqlite3
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import HTTPException
from pydantic import BaseModel

from syftbox.lib.constants import PERM_FILE
from syftbox.lib.hash import hash_file
from syftbox.lib.permissions import (
    ComputedPermission,
    PermissionRule,
    PermissionType,
    SyftPermission,
)
from syftbox.server.db import db
from syftbox.server.db.db import (
    get_rules_for_path,
    link_existing_rules_to_file,
    set_rules_for_permfile,
)
from syftbox.server.db.schema import get_db
from syftbox.server.models.sync_models import AbsolutePath, FileMetadata, RelativePath
from syftbox.server.settings import ServerSettings


class SyftFile(BaseModel):
    metadata: FileMetadata
    data: bytes
    absolute_path: AbsolutePath


def computed_permission_for_user_and_path(connection: sqlite3.Connection, user: str, path: Path) -> ComputedPermission:
    rules: List[PermissionRule] = get_rules_for_path(connection, path)
    return ComputedPermission.from_user_rules_and_path(rules=rules, user=user, path=path)


class FileStore:
    def __init__(self, server_settings: ServerSettings) -> None:
        self.server_settings = server_settings

    @property
    def db_path(self) -> AbsolutePath:
        return self.server_settings.file_db_path

    def delete(self, path: RelativePath, user: str, skip_permission_check: bool = False) -> None:
        with get_db(self.db_path) as conn:
            if path.name.endswith(PERM_FILE) and not skip_permission_check:
                # check admin permission
                computed_perm = computed_permission_for_user_and_path(conn, user, path)
                if not computed_perm.has_permission(PermissionType.ADMIN):
                    raise HTTPException(
                        status_code=403,
                        detail=f"User {user} does not have permission to edit syftperm file for {path}",
                    )

            computed_perm = computed_permission_for_user_and_path(conn, user, path)
            if not computed_perm.has_permission(PermissionType.WRITE):
                raise HTTPException(
                    status_code=403,
                    detail=f"User {user} does not have write permission for {path}",
                )

            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE;")
            try:
                db.delete_file_metadata(conn, str(path))
            except ValueError:
                pass

            if path.name.endswith(PERM_FILE):
                # todo: implement delete for permfile
                permfile = SyftPermission(relative_filepath=path, rules=[])
                set_rules_for_permfile(conn, permfile)

            abs_path = self.server_settings.snapshot_folder / path
            abs_path.unlink(missing_ok=True)
            conn.commit()
            cursor.close()

    def get(self, path: RelativePath, user: str) -> SyftFile:
        with get_db(self.db_path) as conn:
            computed_perm = computed_permission_for_user_and_path(conn, user, path)
            if not computed_perm.has_permission(PermissionType.READ):
                raise HTTPException(
                    status_code=403,
                    detail=f"User {user} does not have read permission for {path}",
                )

            metadata = db.get_one_metadata(conn, path=str(path))
            abs_path = self.server_settings.snapshot_folder / metadata.path

            if not Path(abs_path).exists():
                self.delete(Path(metadata.path.as_posix()), user)
                raise ValueError("File not found")
            return SyftFile(
                metadata=metadata,
                data=self._read_bytes(abs_path),
                absolute_path=abs_path,
            )

    def exists(self, path: RelativePath) -> bool:
        with get_db(self.db_path) as conn:
            try:
                # we are skipping permission check here for now
                db.get_one_metadata(conn, path=str(path))
                return True
            except ValueError:
                return False

    def get_metadata(self, path: RelativePath, user: str, skip_permission_check: bool = False) -> FileMetadata:
        with get_db(self.db_path) as conn:
            if not skip_permission_check:
                computed_perm = computed_permission_for_user_and_path(conn, user, path)
                if not computed_perm.has_permission(PermissionType.READ):
                    raise HTTPException(
                        status_code=403,
                        detail=f"User {user} does not have read permission for {path}",
                    )
            metadata = db.get_one_metadata(conn, path=str(path))
            return metadata

    def _read_bytes(self, path: AbsolutePath) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def put(
        self,
        path: Path,
        contents: bytes,
        user: str,
        check_permission: Optional[PermissionType] = None,
        skip_permission_check: bool = False,
    ) -> None:
        with get_db(self.db_path) as conn:
            if path.name.endswith(PERM_FILE) and not skip_permission_check:
                # check admin permission
                computed_perm = computed_permission_for_user_and_path(conn, user, path)
                if not computed_perm.has_permission(PermissionType.ADMIN):
                    raise HTTPException(
                        status_code=403,
                        detail=f"User {user} does not have permission to edit syftperm file for {path}",
                    )

            if not skip_permission_check:
                computed_perm = computed_permission_for_user_and_path(conn, user, path)
                if check_permission not in [
                    PermissionType.WRITE,
                    PermissionType.CREATE,
                ]:
                    raise ValueError(f"check_permission must be either WRITE or CREATE, got {check_permission}")

                if not computed_perm.has_permission(check_permission):
                    raise HTTPException(
                        status_code=403,
                        detail=f"User {user} does not have write permission for {path}",
                    )

            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE;")
            abs_path = self.server_settings.snapshot_folder / path
            abs_path.parent.mkdir(exist_ok=True, parents=True)

            abs_path.write_bytes(contents)

            # TODO: this is currently not atomic (writing the file and adding rows to db)
            # but its also somehwat challenging to do so. Especially date modified is tricky.
            # Because: if we insert first and write the file later, the date modified it not known yet.
            # If we write the file first and then insert, we might have to revert the file, but we need to
            # set it to the old date modified.
            metadata = hash_file(abs_path, root_dir=self.server_settings.snapshot_folder)
            if metadata is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to hash file {abs_path}",
                )
            db.save_file_metadata(conn, metadata)
            if path.name.endswith(PERM_FILE):
                try:
                    permfile = SyftPermission.from_bytes(contents, path)
                except (yaml.YAMLError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail="invalid syftpermission contents, skipped writing",
                    )
                set_rules_for_permfile(conn, permfile)

            link_existing_rules_to_file(conn, path)

            conn.commit()
            cursor.close()

    def list_for_user(
        self,
        *,
        email: str,
        path: Optional[RelativePath] = None,
    ) -> list[FileMetadata]:
        with get_db(self.db_path) as conn:
            return db.get_filemetadata_with_read_access(conn, email, path)
