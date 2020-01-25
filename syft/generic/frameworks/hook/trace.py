from contextlib import contextmanager


class Trace:
    """
    Manage logging operations on the fly as they run

    Attributes:
        active: if True, recording of operation is active
        out_of_operation: it True, is ready to record the next operation,
            means that it's not inside the execution of an already recorded
            operation
        logs: the list of operations recorded
    """

    def __init__(self):
        self.active = False
        self.out_of_operation = True
        self.logs = []

    def clear(self):
        self.active = False
        self.out_of_operation = True
        self.logs = []

    @contextmanager
    def enabled(self):
        self.logs = []
        self.active = True
        self.out_of_operation = True
        try:
            yield self
        finally:
            self.active = False
            self.out_of_operation = False
