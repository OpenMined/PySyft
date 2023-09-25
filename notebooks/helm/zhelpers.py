"""
Helper module for example applications. Mimics ZeroMQ Guide's zhelpers.h.
"""
# future

# stdlib
import binascii
import os
from random import randint

# third party
import zmq


def socket_set_hwm(socket, hwm=-1):
    """libzmq 2/3/4 compatible sethwm"""
    try:
        socket.sndhwm = socket.rcvhwm = hwm
    except AttributeError:
        socket.hwm = hwm


def dump(msg_or_socket):
    """Receives all message parts from socket, printing each frame neatly"""
    if isinstance(msg_or_socket, zmq.Socket):
        # it's a socket, call on current message
        msg = msg_or_socket.recv_multipart()
    else:
        msg = msg_or_socket
    print("----------------------------------------")
    for part in msg:
        print("[%03d]" % len(part), end=" ")
        try:
            print(part.decode("ascii"))
        except UnicodeDecodeError:
            print(r"0x%s" % (binascii.hexlify(part).decode("ascii")))


def set_id(zsocket):
    """Set simple random printable identity on socket"""
    identity = f"{randint(0, 0x10000):04x}-{randint(0, 0x10000):04x}"
    zsocket.setsockopt_string(zmq.IDENTITY, identity)


def zpipe(ctx):
    """build inproc pipe for talking to threads

    mimic pipe used in czmq zthread_fork.

    Returns a pair of PAIRs connected via inproc
    """
    a = ctx.socket(zmq.PAIR)
    b = ctx.socket(zmq.PAIR)
    a.linger = b.linger = 0
    a.hwm = b.hwm = 1
    iface = "inproc://%s" % binascii.hexlify(os.urandom(8))
    a.bind(iface)
    b.connect(iface)
    return a, b
