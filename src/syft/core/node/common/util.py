# stdlib
from typing import Any
from typing import List


def listify(x: Any) -> List[Any]:
    return list(x) if isinstance(x, (list, tuple)) else ([] if x is None else [x])
