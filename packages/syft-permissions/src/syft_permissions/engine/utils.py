from syft_permissions.engine.matchers import SUPPORTED_TEMPLATE


def specificity_key(pattern: str) -> tuple[int, int, int, int, int]:
    """Higher tuple = more specific. Used to sort rules so most specific wins."""
    segments = pattern.split("/")
    has_template = 1 if SUPPORTED_TEMPLATE in pattern else 0
    literal_count = sum(
        1
        for s in segments
        if not any(c in s for c in ("*", "?", "[", "{")) and SUPPORTED_TEMPLATE not in s
    )
    wildcard_count = sum(
        1
        for s in segments
        if s != "**"
        and any(c in s for c in ("*", "?", "["))
        and SUPPORTED_TEMPLATE not in s
    )
    doublestar_count = sum(1 for s in segments if s == "**")
    total_count = len(segments)

    return (has_template, literal_count, wildcard_count, -doublestar_count, total_count)
