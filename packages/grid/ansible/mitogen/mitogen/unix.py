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
Permit connection of additional contexts that may act with the authority of
this context. For now, the UNIX socket is always mode 0600, i.e. can only be
accessed by root or the same UID. Therefore we can always trust connections to
have the same privilege (auth_id) as the current process.
"""

# stdlib
import errno
import logging
import os
import socket
import struct
import sys
import tempfile

# third party
import mitogen.core
import mitogen.master

LOG = logging.getLogger(__name__)


class Error(mitogen.core.Error):
    """
    Base for errors raised by :mod:`mitogen.unix`.
    """

    pass


class ConnectError(Error):
    """
    Raised when :func:`mitogen.unix.connect` fails to connect to the listening
    socket.
    """

    #: UNIX error number reported by underlying exception.
    errno = None


def is_path_dead(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            s.connect(path)
        except socket.error:
            e = sys.exc_info()[1]
            return e.args[0] in (errno.ECONNREFUSED, errno.ENOENT)
    finally:
        s.close()
    return False


def make_socket_path():
    return tempfile.mktemp(prefix="mitogen_unix_", suffix=".sock")


class ListenerStream(mitogen.core.Stream):
    def on_receive(self, broker):
        sock, _ = self.receive_side.fp.accept()
        try:
            self.protocol.on_accept_client(sock)
        except:
            sock.close()
            raise


class Listener(mitogen.core.Protocol):
    stream_class = ListenerStream
    keep_alive = True

    @classmethod
    def build_stream(cls, router, path=None, backlog=100):
        if not path:
            path = make_socket_path()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.exists(path) and is_path_dead(path):
            LOG.debug("%r: deleting stale %r", cls.__name__, path)
            os.unlink(path)

        sock.bind(path)
        os.chmod(path, int("0600", 8))
        sock.listen(backlog)

        stream = super(Listener, cls).build_stream(router, path)
        stream.accept(sock, sock)
        router.broker.start_receive(stream)
        return stream

    def __repr__(self):
        return "%s.%s(%r)" % (
            __name__,
            self.__class__.__name__,
            self.path,
        )

    def __init__(self, router, path):
        self._router = router
        self.path = path

    def _unlink_socket(self):
        try:
            os.unlink(self.path)
        except OSError:
            e = sys.exc_info()[1]
            # Prevent a shutdown race with the parent process.
            if e.args[0] != errno.ENOENT:
                raise

    def on_shutdown(self, broker):
        broker.stop_receive(self.stream)
        self._unlink_socket()
        self.stream.receive_side.close()

    def on_accept_client(self, sock):
        sock.setblocking(True)
        try:
            (pid,) = struct.unpack(">L", sock.recv(4))
        except (struct.error, socket.error):
            LOG.error("listener: failed to read remote identity: %s", sys.exc_info()[1])
            return

        context_id = self._router.id_allocator.allocate()
        try:
            sock.send(struct.pack(">LLL", context_id, mitogen.context_id, os.getpid()))
        except socket.error:
            LOG.error(
                "listener: failed to assign identity to PID %d: %s",
                pid,
                sys.exc_info()[1],
            )
            return

        context = mitogen.parent.Context(self._router, context_id)
        stream = mitogen.core.MitogenProtocol.build_stream(
            router=self._router,
            remote_id=context_id,
            auth_id=mitogen.context_id,
        )
        stream.name = u"unix_client.%d" % (pid,)
        stream.accept(sock, sock)
        LOG.debug("listener: accepted connection from PID %d: %s", pid, stream.name)
        self._router.register(context, stream)


def _connect(path, broker, sock):
    try:
        # ENOENT, ECONNREFUSED
        sock.connect(path)

        # ECONNRESET
        sock.send(struct.pack(">L", os.getpid()))
        mitogen.context_id, remote_id, pid = struct.unpack(">LLL", sock.recv(12))
    except socket.error:
        e = sys.exc_info()[1]
        ce = ConnectError("could not connect to %s: %s", path, e.args[1])
        ce.errno = e.args[0]
        raise ce

    mitogen.parent_id = remote_id
    mitogen.parent_ids = [remote_id]

    LOG.debug("client: local ID is %r, remote is %r", mitogen.context_id, remote_id)

    router = mitogen.master.Router(broker=broker)
    stream = mitogen.core.MitogenProtocol.build_stream(router, remote_id)
    stream.accept(sock, sock)
    stream.name = u"unix_listener.%d" % (pid,)

    mitogen.core.listen(stream, "disconnect", _cleanup)
    mitogen.core.listen(
        router.broker, "shutdown", lambda: router.disconnect_stream(stream)
    )

    context = mitogen.parent.Context(router, remote_id)
    router.register(context, stream)
    return router, context


def connect(path, broker=None):
    LOG.debug("client: connecting to %s", path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        return _connect(path, broker, sock)
    except:
        sock.close()
        raise


def _cleanup():
    """
    Reset mitogen.context_id and friends when our connection to the parent is
    lost. Per comments on #91, these globals need to move to the Router so
    fix-ups like this become unnecessary.
    """
    mitogen.context_id = 0
    mitogen.parent_id = None
    mitogen.parent_ids = []
