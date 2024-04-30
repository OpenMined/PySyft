# stdlib
from contextlib import contextmanager
import threading


class RWLock:
    """Lock enabling multiple readers but only 1 exclusive writer

    Source: https://cutt.ly/Ij70qaq
    """

    def __init__(self):
        self._read_switch = _LightSwitch()
        self._write_switch = _LightSwitch()
        self._no_readers = threading.Lock()
        self._no_writers = threading.Lock()
        self._readers_queue = threading.RLock()

    @contextmanager
    def reader(self):
        self._reader_acquire()
        try:
            yield
        except Exception:  # pylint: disable=W0706
            raise
        finally:
            self._reader_release()

    @contextmanager
    def writer(self):
        self._writer_acquire()
        try:
            yield
        except Exception:  # pylint: disable=W0706
            raise
        finally:
            self._writer_release()

    def _reader_acquire(self):
        """Readers should block whenever a writer has acquired"""
        self._readers_queue.acquire()
        self._no_readers.acquire()
        self._read_switch.acquire(self._no_writers)
        self._no_readers.release()
        self._readers_queue.release()

    def _reader_release(self):
        self._read_switch.release(self._no_writers)

    def _writer_acquire(self):
        """Acquire the writer lock.

        Only the first writer will lock the readtry and then
        all subsequent writers can simply use the resource as
        it gets freed by the previous writer. The very last writer must
        release the readtry semaphore, thus opening the gate for readers
        to try reading.

        No reader can engage in the entry section if the readtry semaphore
        has been set by a writer previously
        """
        self._write_switch.acquire(self._no_readers)
        self._no_writers.acquire()

    def _writer_release(self):
        self._no_writers.release()
        self._write_switch.release(self._no_readers)


class _LightSwitch:
    """An auxiliary "light switch"-like object

    The first thread turns on the "switch", the last one turns it off.

    Source: https://cutt.ly/Ij70qaq
    """

    def __init__(self):
        self._counter = 0
        self._mutex = threading.RLock()

    def acquire(self, lock):
        self._mutex.acquire()
        self._counter += 1
        if self._counter == 1:
            lock.acquire()
        self._mutex.release()

    def release(self, lock):
        self._mutex.acquire()
        self._counter -= 1
        if self._counter == 0:
            lock.release()
        self._mutex.release()
