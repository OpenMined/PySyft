import random
from abc import ABC


class AbstractWorker(ABC):
    pass


class IdGenerator:
    """
    Generate id and store the list of ids generated
    Can take a pre set list in input and will complete
    when it's empty.
    """
    def __init__(self, given_ids=list()):
        self.given_ids = given_ids
        self.generated = []

    def pop(self, *args):
        """Provide random ids and store them"""
        if len(self.given_ids) > 0:
            random_id = self.given_ids.pop(-1)
        else:
            random_id = int(10e10 * random.random())
        self.generated.append(random_id)
        return random_id
