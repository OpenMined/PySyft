# type: ignore

# stdlib
from typing import Optional
from typing import Union

# relative
from .backends import ShylockAsyncBackend
from .backends import ShylockSyncBackend

BACKEND: Optional[Union[ShylockAsyncBackend, ShylockSyncBackend]] = None


def configure(backend: Union[ShylockAsyncBackend, ShylockSyncBackend]):
    """
    Configure default backend to use
    :param backend: The ready to use backend
    """
    global BACKEND
    BACKEND = backend
