import sqlite3
from pathlib import Path
from typing import Optional

import pytest

from syftbox.lib.constants import PERM_FILE
from syftbox.lib.permissions import PermissionType, SyftPermission
from syftbox.server.db.db import (
    get_read_permissions_for_user,
    get_rules_for_permfile,
    link_existing_rules_to_file,
    print_table,
    set_rules_for_permfile,
)
from syftbox.server.db.file_store import computed_permission_for_user_and_path
from syftbox.server.db.schema import get_db


@pytest.fixture
def connection_with_tables():
    return get_db(":memory:")


def insert_file_metadata(cursor: sqlite3.Cursor, fileid: int, path: str):
    cursor.execute(
        """
    INSERT INTO file_metadata (id, path, datasite, hash, signature, file_size, last_modified) VALUES
        (?, ?, ?, 'hash1', 'signature1', 100, '2024-01-01')
    """,
        (fileid, path, path.split("/")[0]),
    )


def insert_rule(
    cursor: sqlite3.Cursor,
    permfile_path: str,
    priority: int,
    path: str,
    user: str,
    can_read: bool,
    admin: bool,
    disallow: bool,
):
    permfile_dir = permfile_path.rsplit("/", 1)[0]
    permfile_depth = len(Path(permfile_path).parts)
    cursor.execute(
        """
    INSERT INTO rules (permfile_path, permfile_dir, permfile_depth, priority, path, user, can_read, can_create, can_write, admin, disallow) VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            permfile_path,
            permfile_dir,
            permfile_depth,
            priority,
            path,
            user,
            can_read,
            0,
            0,
            admin,
            disallow,
        ),
    )


def insert_rule_files(
    cursor: sqlite3.Cursor,
    permfile_path: str,
    priority: int,
    fileid: int,
    match_for_email: Optional[str] = None,
):
    cursor.execute(
        """
    INSERT INTO rule_files (permfile_path, priority, file_id, match_for_email) VALUES
        (?, ?, ?, ?)
    """,
        (permfile_path, priority, fileid, match_for_email),
    )


def insert_file_mock(connection: sqlite3.Connection, path: str):
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO file_metadata (path, datasite, hash, signature, file_size, last_modified)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (path, path.split("/")[0], "hash", "sig", 0, "2021-01-01"),
    )
    connection.commit()


def get_all_file_mappings(connection: sqlite3.Connection):
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT permfile_path, priority, file_id, match_for_email
        FROM rule_files
    """
    )
    return [dict(row) for row in cursor.fetchall()]


def test_insert_permissions_from_file(connection_with_tables: sqlite3.Connection):
    for f in ["a.txt", "b.txt", "c.txt"]:
        insert_file_mock(connection_with_tables, f"user@example.org/test2/{f}")

    yaml_string = """
    - permissions: read
      path: a.txt
      user: user@example.org

    - permissions: read
      path: "*"
      user: user@example.org

    - permissions: write
      path: b.txt
      user: user@example.org

    - permissions: write
      path: z.txt
      user: "*"
      type: disallow
    """
    file_path = f"user@example.org/test2/{PERM_FILE}"
    file = SyftPermission.from_string(yaml_string, file_path)

    set_rules_for_permfile(connection_with_tables, file)
    connection_with_tables.commit()

    assert len(get_all_file_mappings(connection_with_tables)) == 5

    rules_before = len(get_all_file_mappings(connection_with_tables))

    path = "user@example.org/test2/d.txt"
    insert_file_mock(connection_with_tables, path)
    assert len(get_all_file_mappings(connection_with_tables)) == rules_before

    link_existing_rules_to_file(connection_with_tables, Path(path))
    assert len(get_all_file_mappings(connection_with_tables)) == rules_before + 1


def test_overwrite_permissions_from_file(connection_with_tables: sqlite3.Connection):
    for f in ["a.txt", "b.txt", "c.txt"]:
        insert_file_mock(connection_with_tables, f"user@example.org/test2/{f}")

    yaml_string = """
    - permissions: read
      path: a.txt
      user: user@example.org

    - permissions: write
      path: b.txt
      user: user@example.org

    - permissions: write
      path: z.txt
      user: "*"
      type: disallow
    """
    file_path = f"user@example.org/test2/{PERM_FILE}"
    file = SyftPermission.from_string(yaml_string, file_path)
    set_rules_for_permfile(connection_with_tables, file)
    connection_with_tables.commit()
    written_rules = get_rules_for_permfile(connection_with_tables, file)
    # print all the tables
    print_table(connection_with_tables, "rules")
    print_table(connection_with_tables, "rule_files")
    print_table(connection_with_tables, "file_metadata")

    permissions = [x.permissions for x in written_rules]
    users = [x.user for x in written_rules]
    allows = [x.allow for x in written_rules]
    assert len(written_rules) == 3
    assert permissions == [
        [PermissionType.READ],
        [PermissionType.WRITE],
        [PermissionType.WRITE],
    ]
    assert users == ["user@example.org", "user@example.org", "*"]
    assert allows == [True, True, False]

    assert (
        len(
            [
                x
                for x in get_all_file_mappings(connection_with_tables)
                if x["permfile_path"] == str(file.relative_filepath)
            ]
        )
        == 2
    )

    # overwrite
    yaml_string = """
    - permissions: read
      path: a.txt
      user: user@example.org

    - permissions: create
      path: x.txt
      user: user@example.org

    - permissions: create
      path: z.txt
      user: "*"
      type: disallow

    - permissions: create
      path: d.txt
      user: "*"
    """

    file_path = f"user@example.org/test2/{PERM_FILE}"
    file = SyftPermission.from_string(yaml_string, file_path)
    set_rules_for_permfile(connection_with_tables, file)
    connection_with_tables.commit()
    new_existing_rules = get_rules_for_permfile(connection_with_tables, file)
    paths = [x.path for x in new_existing_rules]
    permissions = [x.permissions for x in new_existing_rules]
    users = [x.user for x in new_existing_rules]
    allows = [x.allow for x in new_existing_rules]
    assert len(new_existing_rules) == 4
    assert paths == ["a.txt", "x.txt", "z.txt", "d.txt"]
    assert permissions == [
        [PermissionType.READ],
        [PermissionType.CREATE],
        [PermissionType.CREATE],
        [PermissionType.CREATE],
    ]
    assert users == ["user@example.org", "user@example.org", "*", "*"]
    assert allows == [True, True, False, True]
    assert len(get_all_file_mappings(connection_with_tables)) == 1


def test_computed_permissions(connection_with_tables: sqlite3.Connection):
    for f in ["a.txt", "b.txt", "c.txt"]:
        insert_file_mock(connection_with_tables, f"user@example.org/test2/{f}")

    # overwrite
    yaml_string = """
    - permissions: read
      path: a.txt
      user: user@example.org

    - permissions: create
      path: x.txt
      user: user@example.org

    - permissions: create
      path: z.txt
      user: "*"
      type: disallow

    - permissions: create
      path: d.txt
      user: "*"
    """

    file_path = f"user@example.org/test2/{PERM_FILE}"
    file = SyftPermission.from_string(yaml_string, file_path)
    set_rules_for_permfile(connection_with_tables, file)
    connection_with_tables.commit()

    computed_permission = computed_permission_for_user_and_path(
        connection_with_tables, "user@example.org", Path("user@example.org/test2/a.txt")
    )
    assert computed_permission.has_permission(PermissionType.READ)


def test_get_all_read_permissions_for_user_default(
    connection_with_tables: sqlite3.Connection,
):
    # Clear existing data
    cursor = connection_with_tables.cursor()

    # Insert some example file metadata
    insert_file_metadata(cursor=cursor, fileid=1, path="user@example.org/test2/a.txt")

    connection_with_tables.commit()
    res = [dict(x) for x in get_read_permissions_for_user(connection_with_tables, "user2@example.org")]

    assert len(res) == 1
    assert res[0]["path"] == "user@example.org/test2/a.txt"
    assert not res[0]["read_permission"]


def test_get_all_read_permissions_for_owner(
    connection_with_tables: sqlite3.Connection,
):
    # Clear existing data
    cursor = connection_with_tables.cursor()

    # Insert some example file metadata
    insert_file_metadata(cursor=cursor, fileid=1, path="user@example.org/test2/a.txt")

    connection_with_tables.commit()
    res = [dict(x) for x in get_read_permissions_for_user(connection_with_tables, "user@example.org")]

    assert len(res) == 1
    assert res[0]["path"] == "user@example.org/test2/a.txt"
    assert res[0]["read_permission"]


def test_single_read_permission(connection_with_tables: sqlite3.Connection):
    cursor = connection_with_tables.cursor()
    path = "user@example.org/test2/a.txt"
    insert_file_metadata(cursor=cursor, fileid=1, path=path)

    insert_rule(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=1,
        path="*",
        user="*",
        can_read=True,
        admin=False,
        disallow=False,
    )

    insert_rule_files(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=1,
        fileid=1,
    )

    connection_with_tables.commit()
    res = [dict(x) for x in get_read_permissions_for_user(connection_with_tables, "user@example.org")]

    assert len(res) == 1
    assert res[0]["path"] == path
    assert res[0]["read_permission"]


def test_single_admin_permission(connection_with_tables: sqlite3.Connection):
    cursor = connection_with_tables.cursor()
    path = "user@example.org/test2/a.txt"
    insert_file_metadata(cursor=cursor, fileid=1, path=path)

    insert_rule(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=1,
        path="*",
        user="*",
        can_read=False,
        admin=True,
        disallow=False,
    )

    insert_rule_files(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=1,
        fileid=1,
    )

    connection_with_tables.commit()
    res = [dict(x) for x in get_read_permissions_for_user(connection_with_tables, "user@example.org")]

    assert len(res) == 1
    assert res[0]["path"] == path
    assert res[0]["read_permission"]


def test_disallow_permission(connection_with_tables: sqlite3.Connection):
    cursor = connection_with_tables.cursor()

    insert_file_metadata(cursor=cursor, fileid=1, path="user@example.org/test2/a.txt")

    insert_rule(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=1,
        path="*",
        user="*",
        can_read=True,
        admin=False,
        disallow=False,
    )

    insert_rule(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=2,
        path="*",
        user="*",
        can_read=True,
        admin=False,
        disallow=True,
    )

    insert_rule_files(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=1,
        fileid=1,
    )

    insert_rule_files(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=2,
        fileid=1,
    )

    connection_with_tables.commit()
    res = [dict(x) for x in get_read_permissions_for_user(connection_with_tables, "user2@example.org")]

    # Print all rule mappings
    cursor.execute("SELECT * FROM rule_files")
    print("\nRule Mappings:")
    for row in cursor.fetchall():
        print(dict(row))

    # Print all rules
    cursor.execute("SELECT * FROM rules")
    print("\nRules:")
    for row in cursor.fetchall():
        print(dict(row))

    assert len(res) == 1
    assert res[0]["path"] == "user@example.org/test2/a.txt"
    assert not res[0]["read_permission"]


def test_inheritance(connection_with_tables: sqlite3.Connection):
    cursor = connection_with_tables.cursor()
    insert_file_metadata(cursor=cursor, fileid=1, path="user@example.org/test2/subdir/a.txt")
    insert_rule(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=1,
        path="*",
        user="*",
        can_read=True,
        admin=False,
        disallow=False,
    )

    insert_rule(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=2,
        path="subdir/*",
        user="*",
        can_read=True,
        admin=False,
        disallow=False,
    )

    insert_rule(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/subdir/{PERM_FILE}",
        priority=1,
        path="*",
        user="*",
        can_read=True,
        admin=False,
        disallow=True,
    )

    insert_rule_files(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=1,
        fileid=1,
    )

    insert_rule_files(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/{PERM_FILE}",
        priority=2,
        fileid=1,
    )

    insert_rule_files(
        cursor=cursor,
        permfile_path=f"user@example.org/test2/subdir/{PERM_FILE}",
        priority=1,
        fileid=1,
    )

    connection_with_tables.commit()
    res = [dict(x) for x in get_read_permissions_for_user(connection_with_tables, "otheruser@example.org")]

    assert len(res) == 1
    assert res[0]["path"] == "user@example.org/test2/subdir/a.txt"
    assert not res[0]["read_permission"]


def test_for_email(connection_with_tables: sqlite3.Connection):
    cursor = connection_with_tables.cursor()
    # Insert file metadata for specific user email
    insert_file_metadata(cursor=cursor, fileid=1, path="alice@example.org/test/bob@example.org/data.txt")

    # Insert rule with {useremail} placeholder
    insert_rule(
        cursor=cursor,
        permfile_path=f"alice@example.org/test/{PERM_FILE}",
        priority=1,
        path="{useremail}/data.txt",
        user="*",
        can_read=True,
        admin=False,
        disallow=False,
    )

    # Insert rule_file mapping that only applies for specific email
    insert_rule_files(
        cursor=cursor,
        permfile_path=f"alice@example.org/test/{PERM_FILE}",
        priority=1,
        fileid=1,
        match_for_email="bob@example.org",
    )

    connection_with_tables.commit()

    # Check that bob@example.org has read permission
    res = [dict(x) for x in get_read_permissions_for_user(connection_with_tables, "bob@example.org")]
    assert len(res) == 1
    assert res[0]["path"] == "alice@example.org/test/bob@example.org/data.txt"
    assert res[0]["read_permission"]


def test_like_clause(connection_with_tables: sqlite3.Connection):
    cursor = connection_with_tables.cursor()
    # Insert file metadata for specific user email
    insert_file_metadata(cursor=cursor, fileid=1, path="alice@example.org/data.txt")
    insert_file_metadata(cursor=cursor, fileid=2, path="bob@example.org/data.txt")

    insert_rule(
        cursor=cursor,
        permfile_path=f"alice@example.org/{PERM_FILE}",
        priority=1,
        path="*",
        user="*",
        can_read=True,
        admin=False,
        disallow=False,
    )

    insert_rule(
        cursor=cursor,
        permfile_path=f"bob@example.org/{PERM_FILE}",
        priority=1,
        path="*",
        user="*",
        can_read=True,
        admin=False,
        disallow=False,
    )

    # Insert rule_file mapping that only applies for specific email
    insert_rule_files(
        cursor=cursor,
        permfile_path=f"alice@example.org/{PERM_FILE}",
        priority=1,
        fileid=1,
    )
    # Insert rule_file mapping that only applies for specific email
    insert_rule_files(
        cursor=cursor,
        permfile_path=f"bob@example.org/{PERM_FILE}",
        priority=1,
        fileid=2,
    )

    connection_with_tables.commit()

    # Check like clause
    res = [
        dict(x)
        for x in get_read_permissions_for_user(
            connection_with_tables, "bob@example.org", path_like="alice@example.org/"
        )
    ]

    assert len(res) == 1
    assert res[0]["path"] == "alice@example.org/data.txt"
    assert res[0]["read_permission"]
