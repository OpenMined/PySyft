import random
from ..typecheck import type_hints
from typing import final


@final
class UID(object):
    """A unique id"""

    def __init__(self):
        self.value = random.getrandbits(32)
