"""This file tests async.py."""

import time
from syft.async_call import Async


def test_async():
    """Tests if operations are executed async when using `Async``decorator."""
    array = []

    @Async
    def custom_append(obj):
        """Sleeps for 1 second and then adds object to array."""
        time.sleep(1)
        array.append(obj)

    custom_append(1)
    array.append(2)
    assert array == [2]
    time.sleep(1)
    assert array == [2, 1]
