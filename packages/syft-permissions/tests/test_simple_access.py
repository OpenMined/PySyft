"""Category 1: Simple access control — owner bypass, wildcards, email patterns, hierarchy."""

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


def _make_service(owner: str = TEST_OWNER_EMAIL) -> ACLService:
    return ACLService(owner=owner)


def _req(path: str, level: AccessLevel, user: str) -> ACLRequest:
    return ACLRequest(path=path, level=level, user=User(id=user))


def test_owner_always_has_access():
    service = _make_service()
    service.add_ruleset(RuleSet(rules=[], path=""))
    # Owner can access even with no rules
    assert service.can_access(_req("any/file.txt", AccessLevel.ADMIN, TEST_OWNER_EMAIL))


def test_star_wildcard_grants_read_to_everyone():
    service = _make_service()
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="**", access=Access(read=["*"]))], path="")
    )
    assert service.can_access(_req("file.txt", AccessLevel.READ, "anyone@example.com"))
    assert not service.can_access(
        _req("file.txt", AccessLevel.WRITE, "anyone@example.com")
    )


def test_empty_access_lists_deny_all():
    service = _make_service()
    service.add_ruleset(RuleSet(rules=[Rule(pattern="**", access=Access())], path=""))
    assert not service.can_access(_req("file.txt", AccessLevel.READ, "user@test.com"))


def test_specific_email_grants_access():
    service = _make_service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(write=["alice@test.com"]))],
            path="",
        )
    )
    assert service.can_access(_req("file.txt", AccessLevel.WRITE, "alice@test.com"))
    assert not service.can_access(_req("file.txt", AccessLevel.WRITE, "bob@test.com"))


def test_domain_wildcard():
    service = _make_service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(read=["*@company.com"]))],
            path="",
        )
    )
    assert service.can_access(_req("file.txt", AccessLevel.READ, "alice@company.com"))
    assert not service.can_access(_req("file.txt", AccessLevel.READ, "alice@other.com"))


def test_admin_implies_write_and_read():
    service = _make_service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(admin=["alice@test.com"]))],
            path="",
        )
    )
    assert service.can_access(_req("file.txt", AccessLevel.ADMIN, "alice@test.com"))
    assert service.can_access(_req("file.txt", AccessLevel.WRITE, "alice@test.com"))
    assert service.can_access(_req("file.txt", AccessLevel.READ, "alice@test.com"))


def test_write_implies_read_but_not_admin():
    service = _make_service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(write=["alice@test.com"]))],
            path="",
        )
    )
    assert service.can_access(_req("file.txt", AccessLevel.READ, "alice@test.com"))
    assert service.can_access(_req("file.txt", AccessLevel.WRITE, "alice@test.com"))
    assert not service.can_access(_req("file.txt", AccessLevel.ADMIN, "alice@test.com"))


def test_read_only_no_write_or_admin():
    service = _make_service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="**", access=Access(read=["alice@test.com"]))],
            path="",
        )
    )
    assert service.can_access(_req("file.txt", AccessLevel.READ, "alice@test.com"))
    assert not service.can_access(_req("file.txt", AccessLevel.WRITE, "alice@test.com"))
    assert not service.can_access(_req("file.txt", AccessLevel.ADMIN, "alice@test.com"))
