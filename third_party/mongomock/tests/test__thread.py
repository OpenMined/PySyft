# stdlib
import unittest

# third party
from mongomock.store import RWLock


class LockTestCase(unittest.TestCase):
    def test_rwlock_exception(self):
        """Asserts exceptions occur between a lock's acquire/release"""
        lock = RWLock()

        for method in [lock.reader, lock.writer]:
            try:
                with method():
                    raise ValueError
            except ValueError:
                pass

            # Accessing private attributes but oh well
            self.assertFalse(lock._no_writers.locked())
            self.assertFalse(lock._no_readers.locked())
