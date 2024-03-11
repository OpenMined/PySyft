# stdlib
from collections import defaultdict
import secrets
from typing import Any

# third party
from faker import Faker


class CachedFaker:
    """Normal faker object can get slow when sampling large datasets, e.g. for Mocks. This one is cached
    And therefore faster
    """

    def __init__(self) -> None:
        self.fake = Faker()
        self.cache: dict[str, list[Any]] = defaultdict(list)

    def __getattr__(self, name: str) -> Any:
        if len(self.cache.get(name, [])) > 100:
            return lambda: secrets.choice(self.cache[name])
        else:

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                method = getattr(self.fake, name)
                res = method(*args, **kwargs)
                self.cache[name].append(res)
                return res

            return wrapper
