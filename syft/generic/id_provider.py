import random
from typing import List
from syft import exceptions


def create_random_id():
    return int(10e10 * random.random())


class IdProvider:
    """Provides Id to all syft objects.

    Generate id and store the list of ids generated
    Can take a pre set list in input and will complete
    when it's empty.

    An instance of IdProvider is accessible via sy.ID_PROVIDER.
    """

    def __init__(self, given_ids=None):
        self.given_ids = given_ids if given_ids is not None else []
        self.generated = set()
        self.record_ids = False
        self.recorded_ids = []

    def pop(self, *args) -> int:
        """Provides random ids and store them.

        The syntax .pop() mimics the list syntax for convenience
        and not the generator syntax.

        Returns:
            Random Id.
        """
        if len(self.given_ids):
            random_id = self.given_ids.pop(-1)
        else:
            random_id = create_random_id()
            while random_id in self.generated:
                random_id = create_random_id()
        self.generated.add(random_id)
        if self.record_ids:
            self.recorded_ids.append(random_id)

        return random_id

    def set_next_ids(self, given_ids: List, check_ids: bool = True):
        """Sets the next ids returned by the id provider

        Note that the ids are returned in reverse order of the list, as a pop()
        operation is applied.

        Args:
            given_ids: List, next ids returned by the id provider
            check_ids: bool, check whether these ids conflict with already generated ids

        """
        if check_ids:
            intersect = self.generated.intersection(set(given_ids))
            if len(intersect) > 0:
                message = f"Provided IDs {intersect} are contained in already generated IDs"
                raise exceptions.IdNotUniqueError(message)

        self.given_ids += given_ids

    def start_recording_ids(self):
        """Starts the recording in form of a list of the generated ids."""
        self.record_ids = True
        self.recorded_ids = []

    def get_recorded_ids(self, continue_recording=False):
        """Returns the generated ids since the last call to start_recording_ids.

        Args:
            continue_recording: if False, the recording is stopped and the
                list of recorded ids is reset

        Returns:
            list of recorded ids
        """
        ret_val = self.recorded_ids
        if not continue_recording:
            self.record_ids = False
            self.recorded_ids = []
        return ret_val

    @staticmethod
    def seed(seed=0):
        random.seed(seed)
