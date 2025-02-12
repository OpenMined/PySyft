import os
from pathlib import Path

import yaml
from loguru import logger
from packaging import version

from syftbox import __version__
from syftbox.lib.constants import PERM_FILE
from syftbox.lib.hash import collect_files, hash_files
from syftbox.lib.permissions import SyftPermission, migrate_permissions
from syftbox.server.db import db
from syftbox.server.db.schema import get_db
from syftbox.server.settings import ServerSettings


def create_folders(folders: list[Path]) -> None:
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)


def run_migrations(settings: ServerSettings) -> None:
    logger.info("Creating folders")
    create_folders(settings.folders)
    logger.info("Initializing DB")
    init_db(settings)


def init_db(settings: ServerSettings) -> None:
    # remove this after the upcoming release
    if version.parse(__version__) > version.parse("0.2.10"):
        # Delete existing DB to avoid conflicts
        db_path = settings.file_db_path.absolute()
        if db_path.exists():
            db_path.unlink()
    migrate_permissions(settings.snapshot_folder)

    # might take very long as snapshot folder grows
    logger.info(f"> Collecting Files from {settings.snapshot_folder.absolute()}")
    files = collect_files(settings.snapshot_folder.absolute())
    logger.info("> Hashing files")
    metadata = hash_files(files, settings.snapshot_folder)
    logger.info(f"> Updating file hashes at {settings.file_db_path.absolute()}")
    con = get_db(settings.file_db_path.absolute())
    cur = con.cursor()
    for m in metadata:
        db.save_file_metadata(con, m)

    # remove files that are not in the snapshot folder
    all_metadata = db.get_all_metadata(con)
    for m in all_metadata:
        abs_path = settings.snapshot_folder / m.path
        if not abs_path.exists():
            logger.info(f"{m.path} not found in {settings.snapshot_folder}, deleting from db")
            db.delete_file_metadata(con, m.path.as_posix())

    # fill the permission tables
    for file in settings.snapshot_folder.rglob(PERM_FILE):
        content = file.read_text()
        rule_dicts = yaml.safe_load(content)
        perm_file = SyftPermission.from_rule_dicts(
            permfile_file_path=file.relative_to(settings.snapshot_folder), rule_dicts=rule_dicts
        )
        db.set_rules_for_permfile(con, perm_file)
        db.link_existing_rules_to_file(con, file.relative_to(settings.snapshot_folder))

    cur.close()
    con.commit()
    con.close()
