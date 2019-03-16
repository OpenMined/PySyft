import random
from abc import ABC


class AbstractWorker(ABC):
    pass


class IdGenerator:
    def __init__(self):
        self.generated = []

    def pop(self, *args):
        """Provide random ids and store them"""
        random_id = int(10e10 * random.random())
        self.generated.append(random_id)
        return random_id
