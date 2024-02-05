# stdlib
import json
from typing import Iterable


def iterator_to_string(iterator: Iterable) -> str:
    log = ""
    for line in iterator:
        for item in line.values():
            if isinstance(item, str):
                log += item
            elif isinstance(item, dict):
                log += json.dumps(item) + "\n"
            else:
                log += str(item)
    return log
