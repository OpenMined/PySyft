"""Module to handle features that are not implemented yet."""

_IGNORED_FEATURES = {
    "array_filters": False,
    "collation": False,
    "let": False,
    "session": False,
}


def _ensure_ignorable_feature(feature) -> None:
    if feature not in _IGNORED_FEATURES:
        msg = (
            f"{feature} is not an error that can be ignored: maybe it has been implemented in Mongomock. "
            f"Here is the list of features that can be ignored: {_IGNORED_FEATURES.keys()}"
        )
        raise KeyError(
            msg,
        )


def ignore_feature(feature) -> None:
    """Ignore a feature instead of raising a NotImplementedError."""
    _ensure_ignorable_feature(feature)
    _IGNORED_FEATURES[feature] = True


def warn_on_feature(feature) -> None:
    """Rasie a NotImplementedError the next times a feature is used."""
    _ensure_ignorable_feature(feature)
    _IGNORED_FEATURES[feature] = False


def raise_for_feature(feature, reason) -> bool:
    _ensure_ignorable_feature(feature)
    if _IGNORED_FEATURES[feature]:
        return False
    raise NotImplementedError(reason)
