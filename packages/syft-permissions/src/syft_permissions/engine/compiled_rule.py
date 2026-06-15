import fnmatch
from dataclasses import dataclass

from syft_permissions.engine.matchers import (
    SUPPORTED_TEMPLATE,
    Matcher,
    create_matcher,
)
from syft_permissions.engine.request import ACLRequest, AccessLevel
from syft_permissions.spec.access import Access
from syft_permissions.spec.rule import Rule


USER_PLACEHOLDER = "USER"


@dataclass
class ResolvedAccess:
    """Access lists after USER placeholders have been resolved.

    Values can be:
    - exact email: "alice@example.com"
    - domain wildcard: "*@example.com"
    - everyone: "*"

    Unlike Access, "USER" never appears here — it has been replaced
    with the requesting user's email (in template rules) or "*" (otherwise).
    """

    admin: list[str]
    write: list[str]
    read: list[str]


class ACLRule:
    def __init__(self, pattern: str, access: ResolvedAccess, matcher: Matcher):
        self.pattern = pattern
        self.access = access
        self.matcher = matcher

    def match(self, path: str, user: str) -> bool:
        return self.matcher.match(path, user)

    def has_admin(self, user_email: str) -> bool:
        return _user_in_list(user_email, self.access.admin)

    def has_write(self, user_email: str) -> bool:
        return self.has_admin(user_email) or _user_in_list(
            user_email, self.access.write
        )

    def has_read(self, user_email: str) -> bool:
        return self.has_write(user_email) or _user_in_list(user_email, self.access.read)

    def check_access(self, request: ACLRequest) -> bool:
        user_email = request.user.id
        if request.level == AccessLevel.ADMIN:
            return self.has_admin(user_email)
        if request.level == AccessLevel.WRITE:
            return self.has_write(user_email)
        return self.has_read(user_email)


def compile_rule(rule: Rule, user: str) -> ACLRule:
    """Compile a spec Rule into an ACLRule with resolved pattern and access lists."""
    pattern = rule.pattern
    access = rule.access

    if SUPPORTED_TEMPLATE in pattern:
        pattern = pattern.replace(SUPPORTED_TEMPLATE, user)
        resolved = _resolve_access(access, user)
    else:
        resolved = _resolve_access(access, "*")

    matcher = create_matcher(pattern)
    return ACLRule(pattern=pattern, access=resolved, matcher=matcher)


def _resolve_access(access: Access, user_replacement: str) -> ResolvedAccess:
    """Resolve USER placeholders in access lists."""
    return ResolvedAccess(
        admin=_replace_in_list(access.admin, user_replacement),
        write=_replace_in_list(access.write, user_replacement),
        read=_replace_in_list(access.read, user_replacement),
    )


def _replace_in_list(lst: list[str], replacement: str) -> list[str]:
    return [replacement if item == USER_PLACEHOLDER else item for item in lst]


def _user_in_list(user_email: str, allowed: list[str]) -> bool:
    for pattern in allowed:
        if pattern == "*":
            return True
        if "*" in pattern or "?" in pattern:
            # this does *@company.com
            if fnmatch.fnmatch(user_email, pattern):
                return True
        elif user_email == pattern:
            return True
    return False
