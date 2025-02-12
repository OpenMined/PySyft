from pathlib import Path

import pytest

from syftbox.lib.constants import PERM_FILE
from syftbox.lib.permissions import (
    ComputedPermission,
    PermissionParsingError,
    PermissionRule,
    PermissionType,
    SyftPermission,
)


def test_parsing_dicts():
    d = {"permissions": "read", "path": "x.txt", "user": "user@example.org"}
    rule = PermissionRule.from_rule_dict(dir_path=Path("."), rule_dict=d, priority=0)
    assert rule.permissions == [PermissionType.READ]
    assert rule.path == "x.txt"
    assert rule.user == "user@example.org"
    assert rule.allow is True


def test_parsing():
    yaml_string = """
    - permissions: read
      path: x.txt
      user: user@example.org

    - permissions: [read, write]
      path: x.txt
      user: "*"
      type: disallow
    """
    file = SyftPermission.from_string(yaml_string, ".")
    assert len(file.rules) == 2

    assert file.rules[0].permissions == [PermissionType.READ]
    assert file.rules[0].path == "x.txt"
    assert file.rules[0].user == "user@example.org"
    assert file.rules[0].allow is True

    # check the same for the second rule
    assert file.rules[1].permissions == [PermissionType.READ, PermissionType.WRITE]
    assert file.rules[1].path == "x.txt"
    assert file.rules[1].user == "*"
    assert file.rules[1].allow is False


def test_parsing_fails():
    yaml_string = """
    - permissions: read
      path: "../*/x.txt"
      user: user@example.org
    """
    with pytest.raises(PermissionParsingError):
        SyftPermission.from_string(yaml_string, ".")


def test_parsing_useremail():
    yaml_string = """
        - permissions: read
          path: "{useremail}/*"
          user: user@example.org
    """

    file = SyftPermission.from_string(yaml_string, ".")
    rule = file.rules[0]
    assert rule.has_email_template
    assert rule.resolve_path_pattern("user@example.org") == "user@example.org/*"


def test_globstar():
    rule = PermissionRule.from_rule_dict(
        dir_path=Path("."),
        rule_dict={"path": "**", "permissions": ["admin", "read", "write"], "user": "user@example.org"},
        priority=0,
    )
    computed_permission = ComputedPermission(user="user@example.org", file_path=Path("a.txt"))
    computed_permission.apply(rule)
    assert computed_permission.has_permission(PermissionType.READ)

    computed_permission = ComputedPermission(user="user@example.org", file_path=Path("b/a.txt"))
    computed_permission.apply(rule)
    assert computed_permission.has_permission(PermissionType.READ)


def test_computed_permission_root_user():
    computed_permission = ComputedPermission.from_user_rules_and_path(
        rules=[], user="user@example.org", path=Path("user@example.org/test/a.txt")
    )
    assert computed_permission.has_permission(PermissionType.READ)
    assert computed_permission.has_permission(PermissionType.WRITE)
    assert computed_permission.has_permission(PermissionType.CREATE)
    assert computed_permission.has_permission(PermissionType.ADMIN)


def test_computed_permission_admin():
    base_rule = PermissionRule.from_rule_dict(
        dir_path=Path("."),
        rule_dict={"path": "user@example.org/test/a.txt", "permissions": ["admin"], "user": "user@example.org"},
        priority=0,
    )

    computed_permission = ComputedPermission.from_user_rules_and_path(
        rules=[base_rule], user="user@example.org", path=Path("user@example.org/test/a.txt")
    )
    assert computed_permission.has_permission(PermissionType.READ)
    assert computed_permission.has_permission(PermissionType.WRITE)
    assert computed_permission.has_permission(PermissionType.CREATE)
    assert computed_permission.has_permission(PermissionType.ADMIN)


def test_computed_permissions_read():
    read_rule = PermissionRule.from_rule_dict(
        dir_path=Path("."),
        rule_dict={"path": "user@example.org/test/a.txt", "permissions": ["read"], "user": "user_2@example.org"},
        priority=0,
    )
    computed_permission = ComputedPermission.from_user_rules_and_path(
        rules=[read_rule], user="user_2@example.org", path=Path("user@example.org/test/a.txt")
    )
    assert computed_permission.has_permission(PermissionType.READ)


def test_computed_permission_disallow():
    allow_rule = PermissionRule.from_rule_dict(
        dir_path=Path("."),
        rule_dict={"path": "user@example.org/test/a.txt", "permissions": ["read"], "user": "user_2@example.org"},
        priority=0,
    )
    disallow_rule = PermissionRule.from_rule_dict(
        dir_path=Path("."),
        rule_dict={
            "path": "user@example.org/test/a.txt",
            "permissions": ["read"],
            "user": "user_2@example.org",
            "type": "disallow",
        },
        priority=1,
    )
    computed_permission = ComputedPermission.from_user_rules_and_path(
        rules=[allow_rule, disallow_rule], user="user_2@example.org", path=Path("user@example.org/test/a.txt")
    )
    assert not computed_permission.has_permission(PermissionType.READ)


def test_computed_permissions_write_create_with_read():
    write_create_rule = PermissionRule.from_rule_dict(
        dir_path=Path("."),
        rule_dict={
            "path": "user@example.org/test/a.txt",
            "permissions": ["read", "write", "create"],
            "user": "user_2@example.org",
        },
        priority=0,
    )
    computed_permission = ComputedPermission.from_user_rules_and_path(
        rules=[write_create_rule], user="user_2@example.org", path=Path("user@example.org/test/a.txt")
    )
    assert computed_permission.has_permission(PermissionType.READ)
    assert computed_permission.has_permission(PermissionType.WRITE)
    assert computed_permission.has_permission(PermissionType.CREATE)


def test_computed_permissions_write_create_without_read():
    write_create_rule = PermissionRule.from_rule_dict(
        dir_path=Path("."),
        rule_dict={
            "path": "user@example.org/test/a.txt",
            "permissions": ["write", "create"],
            "user": "user_2@example.org",
        },
        priority=0,
    )
    computed_permission = ComputedPermission.from_user_rules_and_path(
        rules=[write_create_rule], user="user_2@example.org", path=Path("user@example.org/test/a.txt")
    )
    assert not computed_permission.has_permission(PermissionType.READ)
    assert not computed_permission.has_permission(PermissionType.WRITE)
    assert not computed_permission.has_permission(PermissionType.CREATE)


def test_computed_permission_permfile_access():
    rwc_rule = PermissionRule.from_rule_dict(
        dir_path=Path("."),
        rule_dict={
            "path": f"user@example.org/test/{PERM_FILE}",
            "permissions": ["read", "write", "create"],
            "user": "user_2@example.org",
        },
        priority=0,
    )

    computed_permission = ComputedPermission.from_user_rules_and_path(
        rules=[rwc_rule], user="user_2@example.org", path=Path("user@example.org/test/permfile.yaml")
    )

    assert not computed_permission.has_permission(PermissionType.READ)
    assert not computed_permission.has_permission(PermissionType.WRITE)
    assert not computed_permission.has_permission(PermissionType.CREATE)


def test_permission_file_add_rule():
    perm_file = SyftPermission.from_rule_dicts(
        Path("test") / PERM_FILE,
        [
            {
                "path": "**",
                "user": "user@example.org",
                "permissions": ["read"],
            }
        ],
    )

    # Initially only has read permission
    computed_permission = ComputedPermission.from_user_rules_and_path(
        rules=perm_file.rules, user="user@example.org", path=Path("test/text.txt")
    )
    assert computed_permission.has_permission(PermissionType.READ)
    assert not computed_permission.has_permission(PermissionType.WRITE)
    assert not computed_permission.has_permission(PermissionType.CREATE)

    # Add write and create permissions
    perm_file.add_rule(path="**", user="user@example.org", allow=True, permission=["write", "create"])

    # Check permissions are updated
    computed_permission = ComputedPermission.from_user_rules_and_path(
        rules=perm_file.rules, user="user@example.org", path=Path("test/test.txt")
    )
    assert computed_permission.has_permission(PermissionType.READ)
    assert computed_permission.has_permission(PermissionType.WRITE)
    assert computed_permission.has_permission(PermissionType.CREATE)
