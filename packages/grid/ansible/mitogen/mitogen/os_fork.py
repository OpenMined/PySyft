# Copyright 2019, David Wilson
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# !mitogen: minify_safe

"""
Support for operating in a mixed threading/forking environment.
"""

# stdlib
import os
import socket
import sys
import threading
import weakref

# third party
import mitogen.core

# List of weakrefs. On Python 2.4, mitogen.core registers its Broker on this
# list and mitogen.service registers its Pool too.
_brokers = weakref.WeakKeyDictionary()
_pools = weakref.WeakKeyDictionary()


def _notice_broker_or_pool(obj):
    """
    Used by :mod:`mitogen.core` and :mod:`mitogen.service` to automatically
    register every broker and pool on Python 2.4/2.5.
    """
    if isinstance(obj, mitogen.core.Broker):
        _brokers[obj] = True
    else:
        _pools[obj] = True


def wrap_os__fork():
    corker = Corker(
        brokers=list(_brokers),
        pools=list(_pools),
    )
    try:
        corker.cork()
        return os__fork()
    finally:
        corker.uncork()


# If Python 2.4/2.5 where threading state is not fixed up, subprocess.Popen()
# may still deadlock due to the broker thread. In this case, pause os.fork() so
# that all active threads are paused during fork.
if sys.version_info < (2, 6):
    os__fork = os.fork
    os.fork = wrap_os__fork


class Corker(object):
    """
    Arrange for :class:`mitogen.core.Broker` and optionally
    :class:`mitogen.service.Pool` to be temporarily "corked" while fork
    operations may occur.

    In a mixed threading/forking environment, it is critical no threads are
    active at the moment of fork, as they could hold mutexes whose state is
    unrecoverably snapshotted in the locked state in the fork child, causing
    deadlocks at random future moments.

    To ensure a target thread has all locks dropped, it is made to write a
    large string to a socket with a small buffer that has :data:`os.O_NONBLOCK`
    disabled. CPython will drop the GIL and enter the ``write()`` system call,
    where it will block until the socket buffer is drained, or the write side
    is closed.

    :class:`mitogen.core.Poller` is used to ensure the thread really has
    blocked outside any Python locks, by checking if the socket buffer has
    started to fill.

    Since this necessarily involves posting a message to every existent thread
    and verifying acknowledgement, it will never be a fast operation.

    This does not yet handle the case of corking being initiated from within a
    thread that is also a cork target.

    :param brokers:
        Sequence of :class:`mitogen.core.Broker` instances to cork.
    :param pools:
        Sequence of :class:`mitogen.core.Pool` instances to cork.
    """

    def __init__(self, brokers=(), pools=()):
        self.brokers = brokers
        self.pools = pools

    def _do_cork(self, s, wsock):
        try:
            try:
                while True:
                    # at least EINTR is possible. Do our best to keep handling
                    # outside the GIL in this case using sendall().
                    wsock.sendall(s)
            except socket.error:
                pass
        finally:
            wsock.close()

    def _cork_one(self, s, obj):
        """
        Construct a socketpair, saving one side of it, and passing the other to
        `obj` to be written to by one of its threads.
        """
        rsock, wsock = mitogen.parent.create_socketpair(size=4096)
        mitogen.core.set_cloexec(rsock.fileno())
        mitogen.core.set_cloexec(wsock.fileno())
        mitogen.core.set_block(wsock)  # gevent
        self._rsocks.append(rsock)
        obj.defer(self._do_cork, s, wsock)

    def _verify_one(self, rsock):
        """
        Pause until the socket `rsock` indicates readability, due to
        :meth:`_do_cork` triggering a blocking write on another thread.
        """
        poller = mitogen.core.Poller()
        poller.start_receive(rsock.fileno())
        try:
            while True:
                for fd in poller.poll():
                    return
        finally:
            poller.close()

    def cork(self):
        """
        Arrange for any associated brokers and pools to be paused with no locks
        held. This will not return until each thread acknowledges it has ceased
        execution.
        """
        current = threading.currentThread()
        s = mitogen.core.b("CORK") * ((128 // 4) * 1024)
        self._rsocks = []

        # Pools must be paused first, as existing work may require the
        # participation of a broker in order to complete.
        for pool in self.pools:
            if not pool.closed:
                for th in pool._threads:
                    if th != current:
                        self._cork_one(s, pool)

        for broker in self.brokers:
            if broker._alive:
                if broker._thread != current:
                    self._cork_one(s, broker)

        # Pause until we can detect every thread has entered write().
        for rsock in self._rsocks:
            self._verify_one(rsock)

    def uncork(self):
        """
        Arrange for paused threads to resume operation.
        """
        for rsock in self._rsocks:
            rsock.close()
