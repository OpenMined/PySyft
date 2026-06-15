"""Category 4: Tree traversal — closest permission file wins, no merging."""

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


def test_closest_permission_file_wins():
    """Child permission file overrides parent — no merging."""
    service = _service()
    # Parent: grants read to everyone
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="**", access=Access(read=["*"]))], path="")
    )
    # Child: only alice can read
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(read=["alice@test.com"]))],
            path="projects/reports",
        )
    )

    # bob can read at root level
    assert service.can_access(_req("readme.txt", AccessLevel.READ, "bob@test.com"))
    # bob cannot read under projects/reports (child overrides parent)
    assert not service.can_access(
        _req("projects/reports/q1.csv", AccessLevel.READ, "bob@test.com")
    )
    # alice can read under projects/reports
    assert service.can_access(
        _req("projects/reports/q1.csv", AccessLevel.READ, "alice@test.com")
    )


def test_parent_grants_child_denies():
    service = _service()
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="**", access=Access(write=["*"]))], path="")
    )
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(read=["*"]))],
            path="secret",
        )
    )
    # At root: everyone can write
    assert service.can_access(_req("file.txt", AccessLevel.WRITE, "user@test.com"))
    # Under secret/: only read allowed
    assert service.can_access(
        _req("secret/file.txt", AccessLevel.READ, "user@test.com")
    )
    assert not service.can_access(
        _req("secret/file.txt", AccessLevel.WRITE, "user@test.com")
    )


def test_parent_denies_child_grants():
    service = _service()
    service.add_ruleset(RuleSet(rules=[Rule(pattern="**", access=Access())], path=""))
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(read=["*"]))],
            path="public",
        )
    )
    # At root: nothing is allowed
    assert not service.can_access(_req("file.txt", AccessLevel.READ, "user@test.com"))
    # Under public/: read is allowed
    assert service.can_access(
        _req("public/file.txt", AccessLevel.READ, "user@test.com")
    )


def test_deeply_nested_permission_file():
    service = _service()
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="**", access=Access(read=["*"]))], path="")
    )
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(write=["*"]))],
            path="a/b/c",
        )
    )
    # At root: read only
    assert not service.can_access(_req("file.txt", AccessLevel.WRITE, "user@test.com"))
    # At a/b/c: write allowed
    assert service.can_access(
        _req("a/b/c/file.txt", AccessLevel.WRITE, "user@test.com")
    )
