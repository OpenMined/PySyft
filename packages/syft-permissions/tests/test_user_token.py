"""Category 3: UserEmail template and USER placeholder resolution."""

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


def test_useremail_template_with_user_placeholder():
    """{{.UserEmail}}/** pattern with USER in access list resolves per-user."""
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[
                Rule(
                    pattern="{{.UserEmail}}/**",
                    access=Access(write=["USER"]),
                )
            ],
            path="",
        )
    )
    # alice can write in her own directory
    assert service.can_access(
        _req("alice@test.com/file.txt", AccessLevel.WRITE, "alice@test.com")
    )
    # alice cannot write in bob's directory
    assert not service.can_access(
        _req("bob@test.com/file.txt", AccessLevel.WRITE, "alice@test.com")
    )
    # bob can write in his own directory
    assert service.can_access(
        _req("bob@test.com/file.txt", AccessLevel.WRITE, "bob@test.com")
    )


def test_user_without_template_resolves_to_star():
    """USER without {{.UserEmail}} template resolves to * (any authenticated user)."""
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="public/**", access=Access(read=["USER"]))],
            path="",
        )
    )
    # Anyone can read public files
    assert service.can_access(
        _req("public/data.csv", AccessLevel.READ, "anyone@test.com")
    )


def test_useremail_template_exact_match():
    """{{.UserEmail}} without glob matches exactly."""
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[
                Rule(
                    pattern="{{.UserEmail}}/profile.txt",
                    access=Access(read=["USER"]),
                )
            ],
            path="",
        )
    )
    assert service.can_access(
        _req("alice@test.com/profile.txt", AccessLevel.READ, "alice@test.com")
    )
    assert not service.can_access(
        _req("alice@test.com/other.txt", AccessLevel.READ, "alice@test.com")
    )


def test_useremail_template_ranked_above_literal():
    """Template patterns are more specific than literal patterns."""
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[
                Rule(pattern="**", access=Access(read=["*"])),
                Rule(
                    pattern="{{.UserEmail}}/**",
                    access=Access(write=["USER"]),
                ),
            ],
            path="",
        )
    )
    # alice gets write in her directory (template rule matches first)
    assert service.can_access(
        _req("alice@test.com/file.txt", AccessLevel.WRITE, "alice@test.com")
    )
