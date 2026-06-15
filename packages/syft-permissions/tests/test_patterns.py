"""Category 2: Pattern matching and specificity ordering."""

from syft_permissions import (
    ACLRequest,
    ACLService,
    AccessLevel,
    Access,
    Rule,
    RuleSet,
    User,
)
from syft_permissions.engine.utils import specificity_key

from .conftest import TEST_OWNER_EMAIL


def _service() -> ACLService:
    return ACLService(owner=TEST_OWNER_EMAIL)


def _req(path: str, level: AccessLevel, user: str) -> ACLRequest:
    return ACLRequest(path=path, level=level, user=User(id=user))


def test_single_star_matches_current_directory_only():
    """Single * matches files in the current directory, not subdirectories."""
    service = _service()
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="*", access=Access(read=["*"]))], path="")
    )
    assert service.can_access(_req("file.txt", AccessLevel.READ, "user@test.com"))
    assert service.can_access(_req("data.csv", AccessLevel.READ, "user@test.com"))
    assert not service.can_access(
        _req("sub/file.txt", AccessLevel.READ, "user@test.com")
    )
    assert not service.can_access(_req("a/b/c.txt", AccessLevel.READ, "user@test.com"))


def test_doublestar_matches_all():
    service = _service()
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="**", access=Access(read=["*"]))], path="")
    )
    assert service.can_access(_req("a/b/c.txt", AccessLevel.READ, "user@test.com"))


def test_star_csv_matches_csv_in_root():
    service = _service()
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="*.csv", access=Access(read=["*"]))], path="")
    )
    assert service.can_access(_req("data.csv", AccessLevel.READ, "user@test.com"))
    assert not service.can_access(
        _req("sub/data.csv", AccessLevel.READ, "user@test.com")
    )


def test_doublestar_csv_matches_csv_anywhere():
    service = _service()
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="**/*.csv", access=Access(read=["*"]))], path="")
    )
    assert service.can_access(_req("sub/data.csv", AccessLevel.READ, "user@test.com"))
    assert service.can_access(_req("a/b/data.csv", AccessLevel.READ, "user@test.com"))


def test_reports_doublestar_matches_under_reports():
    service = _service()
    service.add_ruleset(
        RuleSet(rules=[Rule(pattern="reports/**", access=Access(read=["*"]))], path="")
    )
    assert service.can_access(_req("reports/q1.csv", AccessLevel.READ, "user@test.com"))
    assert service.can_access(
        _req("reports/2024/q1.csv", AccessLevel.READ, "user@test.com")
    )
    assert not service.can_access(
        _req("other/file.txt", AccessLevel.READ, "user@test.com")
    )


def test_exact_path_match():
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="reports/q1.csv", access=Access(read=["*"]))],
            path="",
        )
    )
    assert service.can_access(_req("reports/q1.csv", AccessLevel.READ, "user@test.com"))
    assert not service.can_access(
        _req("reports/q2.csv", AccessLevel.READ, "user@test.com")
    )


def test_specificity_more_specific_rule_wins():
    """When multiple rules match, most specific wins (first after sorting)."""
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[
                Rule(pattern="**", access=Access(read=["*"])),
                Rule(
                    pattern="reports/q1.csv",
                    access=Access(write=["alice@test.com"]),
                ),
            ],
            path="",
        )
    )
    # The exact match is more specific — alice gets write
    assert service.can_access(
        _req("reports/q1.csv", AccessLevel.WRITE, "alice@test.com")
    )
    # For a non-matching path, falls back to ** (read-only)
    assert not service.can_access(
        _req("other.txt", AccessLevel.WRITE, "alice@test.com")
    )


def test_no_catch_all_denies():
    service = _service()
    service.add_ruleset(
        RuleSet(
            rules=[Rule(pattern="reports/**", access=Access(read=["*"]))],
            path="",
        )
    )
    assert not service.can_access(_req("data.csv", AccessLevel.READ, "user@test.com"))


def test_specificity_ordering():
    """Verify the specificity spec ordering."""
    patterns = [
        "{{.UserEmail}}/**",
        "reports/q1.csv",
        "reports/**",
        "*.csv",
        "**/*.csv",
        "**",
    ]
    sorted_patterns = sorted(patterns, key=specificity_key, reverse=True)
    assert sorted_patterns == patterns
