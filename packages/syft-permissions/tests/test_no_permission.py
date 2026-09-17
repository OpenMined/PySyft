"""Categories 6+7: No permission file, empty rules, no matching rule — deny."""

from syft_permissions import (
    ACLRequest,
    ACLService,
    AccessLevel,
    Access,
    Rule,
    RuleSet,
    User,
)

from .conftest import TEST_OWNER_EMAIL


def _service() -> ACLService:
    return ACLService(owner=TEST_OWNER_EMAIL)


def _req(path: str, level: AccessLevel, user: str) -> ACLRequest:
    return ACLRequest(path=path, level=level, user=User(id=user))


def test_no_permission_file_denies():
    service = _service()
    # No rulesets added at all
    assert not service.can_access(_req("file.txt", AccessLevel.READ, "user@test.com"))


def test_no_permission_file_owner_still_can_access():
    service = _service()
    assert service.can_access(_req("file.txt", AccessLevel.ADMIN, TEST_OWNER_EMAIL))


def test_empty_rules_deny():
    service = _service()
    service.add_ruleset(RuleSet(rules=[], path=""))
    assert not service.can_access(_req("file.txt", AccessLevel.READ, "user@test.com"))


def test_no_matching_rule_denies():
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="reports/**", access=Access(read=["*"]))],
            path="",
        )
    )
    # Path doesn't match any rule
    assert not service.can_access(_req("data.csv", AccessLevel.READ, "user@test.com"))


def test_permission_file_access_requires_admin():
    """Accessing syft.pub.yaml itself requires ADMIN-level access."""
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(write=["user@test.com"]))],
            path="",
        )
    )
    # User has write access generally, but syft.pub.yaml requires admin
    assert not service.can_access(
        _req("syft.pub.yaml", AccessLevel.READ, "user@test.com")
    )


def test_permission_file_admin_can_access():
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(admin=["user@test.com"]))],
            path="",
        )
    )
    assert service.can_access(_req("syft.pub.yaml", AccessLevel.READ, "user@test.com"))


def test_remove_ruleset_then_deny():
    service = _service()
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="**", access=Access(read=["*"]))], path="")
    )
    assert service.can_access(_req("file.txt", AccessLevel.READ, "user@test.com"))
    service.remove_ruleset("")
    assert not service.can_access(_req("file.txt", AccessLevel.READ, "user@test.com"))


def test_permission_file_gate_ignores_case():
    """A mis-cased name still needs ADMIN.

    macOS and Windows keep the case of a name but match it without case. A
    write to SYFT.PUB.YAML there replaces the content of syft.pub.yaml, so the
    gate must not read the case.
    """
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[
                Rule(
                    pattern="**",
                    access=Access(admin=["admin@test.com"], write=["user@test.com"]),
                )
            ],
            path="",
        )
    )
    for path in ("SYFT.PUB.YAML", "sub/Syft.Pub.Yaml"):
        # The level is raised to ADMIN, not refused: an admin keeps the write.
        assert not service.can_access(_req(path, AccessLevel.WRITE, "user@test.com"))
        assert service.can_access(_req(path, AccessLevel.WRITE, "admin@test.com"))


def test_permission_file_gate_matches_the_last_path_part_only():
    """A name that ends with the constant is not a permission file."""
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(write=["user@test.com"]))],
            path="",
        )
    )
    assert service.can_access(
        _req("notsyft.pub.yaml", AccessLevel.WRITE, "user@test.com")
    )
