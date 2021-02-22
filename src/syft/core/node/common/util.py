# stdlib
from typing import Any
from typing import List


def listify(x: Any) -> List[Any]:
    """turns x into a list.
    If x is a list or tuple, return as list.
    if x is not a list: return [x]
    if x is None: return []

    Args:
        x (Any): some object

    Returns:
        List[Any]: x, as a list
    """
    return list(x) if isinstance(x, (list, tuple)) else ([] if x is None else [x])
