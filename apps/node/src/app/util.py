import numpy
import math


def mask_payload_fast(self, payload):
    """Monkey patch geventwebsocket.websocket.Header.mask_payload(). Version
    currently in geventwebsocket does a very slow python for loop to mask the
    payload.

    We take advantage of numpy to do this faster.
    """
    key = (self.mask * int(math.ceil(float(len(payload)) / float(len(self.mask)))))[
        : len(payload)
    ]

    # Select the type size in bytes
    if len(payload) % 8 == 0:
        dt = numpy.dtype("<Q")
    else:
        dt = numpy.dtype("B")

    return numpy.bitwise_xor(
        numpy.fromstring(key, dtype=dt), numpy.fromstring(payload, dtype=dt)
    ).tostring()
