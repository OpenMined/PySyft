# stdlib
import math
from typing import Any


def _safe_isnan(v: Any) -> bool:
    try:
        return math.isnan(v)
    except TypeError:
        return False


def _syft_equals(v1: Any, v2: Any) -> bool:
    if v1 is None:
        return v2 is None

    # handle nan, since nan==nan is False
    if _safe_isnan(v1):
        return _safe_isnan(v2)

    if isinstance(v1, dict) and isinstance(v2, dict):
        return _syft_recursive_dict_equals(v1, v2)

    return v1 == v2


def _syft_recursive_dict_equals(d1: dict, d2: dict) -> bool:
    if len(d1) != len(d2):
        return False

    for k in d1.keys():
        if k not in d2:
            return False
        if not _syft_equals(d1[k], d2[k]):
            return False

    return True
