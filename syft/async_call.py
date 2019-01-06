"""Asynchronous operations related classes."""

import threading

from .exceptions import AsyncOperationNotFinalizedError
from .exceptions import ThreadTimeOutError


class AsyncCallWrapper(object):
    """Asynchronous call definition.

    Runs `self.fnc` asynchronously using `threading`.

    Attributes:
        fnc: A function to be executed asynchronously.
        callaback: A function that should be executed after the asynchronous
            operations.
        result: The return of `fnc` if it has finished the execution
            otherwhise raises `AsyncOperationNotFinalizedError()`.
    """

    def __init__(self, fnc, callback=None):
        self.fnc = fnc
        self.callback = callback
        self._thread, self._result = None, None

    def __call__(self, *args, **kwargs):
        """Calls `self.run` using `threading`."""
        self._thread = threading.Thread(
            target=self.run, name=self.fnc.__name__, args=args, kwargs=kwargs
        )
        self._thread.start()
        return self

    @property
    def result(self):
        if not self._thread or self._thread.isAlive():
            raise AsyncOperationNotFinalizedError()
        else:
            return self._result

    def wait(self, timeout=None):
        self._thread.join(timeout)
        if self._thread.isAlive():
            raise ThreadTimeOutError()
        else:
            return self._result

    def run(self, *args, **kwargs):
        """Calls `self.fnc` and a callback function if available."""
        self._result = self.fnc(*args, **kwargs)
        if self.callback:
            self.callback(self._result)


def Async(fnc=None, callback=None):
    def async_wrapper(fnc):
        return AsyncCallWrapper(fnc, callback)

    if fnc:
        return AsyncCallWrapper(fnc, callback)
    else:
        return async_wrapper
