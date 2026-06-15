"""Category 5: Terminal rulesets block child permission files."""

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


def test_terminal_blocks_child_permission_file():
    service = _service()
    # Parent: terminal, read only
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(read=["*"]))],
            terminal=True,
            path="",
        )
    )
    # Child: would grant write, but parent is terminal
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(write=["*"]))],
            path="subdir",
        )
    )
    # Root rules apply even to subdir content
    assert service.can_access(
        _req("subdir/file.txt", AccessLevel.READ, "user@test.com")
    )
    assert not service.can_access(
        _req("subdir/file.txt", AccessLevel.WRITE, "user@test.com")
    )


def test_terminal_blocks_grandchild():
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(read=["*"]))],
            terminal=True,
            path="",
        )
    )
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(write=["*"]))],
            path="a/b/c",
        )
    )
    assert service.can_access(_req("a/b/c/file.txt", AccessLevel.READ, "user@test.com"))
    assert not service.can_access(
        _req("a/b/c/file.txt", AccessLevel.WRITE, "user@test.com")
    )


def test_mid_level_terminal():
    service = _service()
    # Root: read for all
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="**", access=Access(read=["*"]))], path="")
    )
    # Mid-level: terminal, grants write
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(write=["*"]))],
            terminal=True,
            path="projects",
        )
    )
    # Deep child: would grant admin, blocked by terminal at projects/
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(admin=["*"]))],
            path="projects/secret",
        )
    )
    # At root: read only
    assert service.can_access(_req("readme.txt", AccessLevel.READ, "user@test.com"))
    assert not service.can_access(
        _req("readme.txt", AccessLevel.WRITE, "user@test.com")
    )
    # At projects/: write allowed
    assert service.can_access(
        _req("projects/file.txt", AccessLevel.WRITE, "user@test.com")
    )
    # At projects/secret/: terminal at projects blocks it, write applies
    assert service.can_access(
        _req("projects/secret/file.txt", AccessLevel.WRITE, "user@test.com")
    )
    assert not service.can_access(
        _req("projects/secret/file.txt", AccessLevel.ADMIN, "user@test.com")
    )
