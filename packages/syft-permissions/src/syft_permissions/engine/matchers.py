from typing import Protocol

from wcmatch.glob import globmatch, GLOBSTAR

SUPPORTED_TEMPLATE = "{{.UserEmail}}"
UNSUPPORTED_TEMPLATES = ["{{.UserHash}}", "{{.Year}}", "{{.Month}}", "{{.Date}}"]


class Matcher(Protocol):
    def match(self, path: str, user: str | None = None) -> bool: ...


class ExactMatcher:
    def __init__(self, pattern: str):
        self.pattern = pattern

    def match(self, path: str, user: str | None = None) -> bool:
        return path == self.pattern


class GlobMatcher:
    def __init__(self, pattern: str):
        self.pattern = pattern

    def match(self, path: str, user: str | None = None) -> bool:
        return globmatch(path, self.pattern, flags=GLOBSTAR)


class TemplateMatcher:
    def __init__(self, pattern: str):
        for tpl in UNSUPPORTED_TEMPLATES:
            if tpl in pattern:
                raise ValueError(f"Unsupported template: {tpl}")
        self.pattern = pattern

    def match(self, path: str, user: str | None = None) -> bool:
        if user is None:
            return False
        resolved = self.pattern.replace(SUPPORTED_TEMPLATE, user)
        if any(c in resolved for c in ("*", "?", "[")):
            return globmatch(path, resolved, flags=GLOBSTAR)
        return path == resolved


def create_matcher(pattern: str) -> Matcher:
    if SUPPORTED_TEMPLATE in pattern:
        return TemplateMatcher(pattern)
    if any(c in pattern for c in ("*", "?", "[")):
        return GlobMatcher(pattern)
    return ExactMatcher(pattern)
