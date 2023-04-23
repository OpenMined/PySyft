# stdlib
from collections import defaultdict
import random

# third party
from faker import Faker


class CachedFaker:
    """Normal faker object can get slow when sampling large datasets, e.g. for Mocks. This one is cached
    And therefore faster
    """

    def __init__(self):
        self.fake = Faker()
        self.cache = defaultdict(list)

    def __getattr__(self, name):
        if len(self.cache.get(name, [])) > 100:
            return lambda: random.choice(self.cache[name])
        else:

            def wrapper(*args, **kwargs):
                method = getattr(self.fake, name)
                res = method(*args, **kwargs)
                self.cache[name].append(res)
                return res

            return wrapper
