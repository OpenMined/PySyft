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
This module defines functionality common to master and parent processes. It is
sent to any child context that is due to become a parent, due to recursive
connection.
"""

# stdlib
import codecs
import errno
import fcntl
import getpass
import heapq
import inspect
import logging
import os
import platform
import re
import signal
import socket
import struct
import subprocess
import sys
import termios
import textwrap
import threading
import zlib

# Absolute imports for <2.5.
select = __import__("select")

try:
    # third party
    import thread
except ImportError:
    import threading as thread

# third party
import mitogen.core
from mitogen.core import IOLOG
from mitogen.core import b
from mitogen.core import bytes_partition

LOG = logging.getLogger(__name__)

# #410: we must avoid the use of socketpairs if SELinux is enabled.
try:
    fp = open("/sys/fs/selinux/enforce", "rb")
    try:
        SELINUX_ENABLED = bool(int(fp.read()))
    finally:
        fp.close()
except IOError:
    SELINUX_ENABLED = False


try:
    next
except NameError:
    # Python 2.4/2.5
    # third party
    from mitogen.core import next


itervalues = getattr(dict, "itervalues", dict.values)

if mitogen.core.PY3:
    xrange = range
    closure_attr = "__closure__"
    IM_SELF_ATTR = "__self__"
else:
    closure_attr = "func_closure"
    IM_SELF_ATTR = "im_self"


try:
    SC_OPEN_MAX = os.sysconf("SC_OPEN_MAX")
except ValueError:
    SC_OPEN_MAX = 1024

BROKER_SHUTDOWN_MSG = (
    "Connection cancelled because the associated Broker began to shut down."
)

OPENPTY_MSG = (
    "Failed to create a PTY: %s. It is likely the maximum number of PTYs has "
    "been reached. Consider increasing the 'kern.tty.ptmx_max' sysctl on OS "
    "X, the 'kernel.pty.max' sysctl on Linux, or modifying your configuration "
    "to avoid PTY use."
)

SYS_EXECUTABLE_MSG = (
    "The Python sys.executable variable is unset, indicating Python was "
    "unable to determine its original program name. Unless explicitly "
    "configured otherwise, child contexts will be started using "
    "'/usr/bin/python'"
)
_sys_executable_warning_logged = False


def _ioctl_cast(n):
    """
    Linux ioctl() request parameter is unsigned, whereas on BSD/Darwin it is
    signed. Until 2.5 Python exclusively implemented the BSD behaviour,
    preventing use of large unsigned int requests like the TTY layer uses
    below. So on 2.4, we cast our unsigned to look like signed for Python.
    """
    if sys.version_info < (2, 5):
        (n,) = struct.unpack("i", struct.pack("I", n))
    return n


# If not :data:`None`, called prior to exec() of any new child process. Used by
# :func:`mitogen.utils.reset_affinity` to allow the child to be freely
# scheduled.
_preexec_hook = None

# Get PTY number; asm-generic/ioctls.h
LINUX_TIOCGPTN = _ioctl_cast(2147767344)

# Lock/unlock PTY; asm-generic/ioctls.h
LINUX_TIOCSPTLCK = _ioctl_cast(1074025521)

IS_LINUX = os.uname()[0] == "Linux"

SIGNAL_BY_NUM = dict(
    (getattr(signal, name), name)
    for name in sorted(vars(signal), reverse=True)
    if name.startswith("SIG") and not name.startswith("SIG_")
)

_core_source_lock = threading.Lock()
_core_source_partial = None


def get_log_level():
    return LOG.getEffectiveLevel() or logging.INFO


def get_sys_executable():
    """
    Return :data:`sys.executable` if it is set, otherwise return
    ``"/usr/bin/python"`` and log a warning.
    """
    if sys.executable:
        return sys.executable

    global _sys_executable_warning_logged
    if not _sys_executable_warning_logged:
        LOG.warn(SYS_EXECUTABLE_MSG)
        _sys_executable_warning_logged = True

    return "/usr/bin/python"


def _get_core_source():
    """
    In non-masters, simply fetch the cached mitogen.core source code via the
    import mechanism. In masters, this function is replaced with a version that
    performs minification directly.
    """
    return inspect.getsource(mitogen.core)


def get_core_source_partial():
    """
    _get_core_source() is expensive, even with @lru_cache in minify.py, threads
    can enter it simultaneously causing severe slowdowns.
    """
    global _core_source_partial

    if _core_source_partial is None:
        _core_source_lock.acquire()
        try:
            if _core_source_partial is None:
                _core_source_partial = PartialZlib(_get_core_source().encode("utf-8"))
        finally:
            _core_source_lock.release()

    return _core_source_partial


def get_default_remote_name():
    """
    Return the default name appearing in argv[0] of remote machines.
    """
    s = u"%s@%s:%d"
    s %= (getpass.getuser(), socket.gethostname(), os.getpid())
    # In mixed UNIX/Windows environments, the username may contain slashes.
    return s.translate({ord(u"\\"): ord(u"_"), ord(u"/"): ord(u"_")})


def is_immediate_child(msg, stream):
    """
    Handler policy that requires messages to arrive only from immediately
    connected children.
    """
    return msg.src_id == stream.protocol.remote_id


def flags(names):
    """
    Return the result of ORing a set of (space separated) :py:mod:`termios`
    module constants together.
    """
    return sum(getattr(termios, name, 0) for name in names.split())


def cfmakeraw(tflags):
    """
    Given a list returned by :py:func:`termios.tcgetattr`, return a list
    modified in a manner similar to the `cfmakeraw()` C library function, but
    additionally disabling local echo.
    """
    # BSD: github.com/freebsd/freebsd/blob/master/lib/libc/gen/termios.c#L162
    # Linux: github.com/lattera/glibc/blob/master/termios/cfmakeraw.c#L20
    iflag, oflag, cflag, lflag, ispeed, ospeed, cc = tflags
    iflag &= ~flags(
        "IMAXBEL IXOFF INPCK BRKINT PARMRK " "ISTRIP INLCR ICRNL IXON IGNPAR"
    )
    iflag &= ~flags("IGNBRK BRKINT PARMRK")
    oflag &= ~flags("OPOST")
    lflag &= ~flags(
        "ECHO ECHOE ECHOK ECHONL ICANON ISIG " "IEXTEN NOFLSH TOSTOP PENDIN"
    )
    cflag &= ~flags("CSIZE PARENB")
    cflag |= flags("CS8 CREAD")
    return [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]


def disable_echo(fd):
    old = termios.tcgetattr(fd)
    new = cfmakeraw(old)
    flags = getattr(termios, "TCSASOFT", 0)
    if not mitogen.core.IS_WSL:
        # issue #319: Windows Subsystem for Linux as of July 2018 throws EINVAL
        # if TCSAFLUSH is specified.
        flags |= termios.TCSAFLUSH
    termios.tcsetattr(fd, flags, new)


def create_socketpair(size=None):
    """
    Create a :func:`socket.socketpair` for use as a child's UNIX stdio
    channels. As socketpairs are bidirectional, they are economical on file
    descriptor usage as one descriptor can be used for ``stdin`` and
    ``stdout``. As they are sockets their buffers are tunable, allowing large
    buffers to improve file transfer throughput and reduce IO loop iterations.
    """
    if size is None:
        size = mitogen.core.CHUNK_SIZE

    parentfp, childfp = socket.socketpair()
    for fp in parentfp, childfp:
        fp.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, size)

    return parentfp, childfp


def create_best_pipe(escalates_privilege=False):
    """
    By default we prefer to communicate with children over a UNIX socket, as a
    single file descriptor can represent bidirectional communication, and a
    cross-platform API exists to align buffer sizes with the needs of the
    library.

    SELinux prevents us setting up a privileged process to inherit an AF_UNIX
    socket, a facility explicitly designed as a better replacement for pipes,
    because at some point in the mid 90s it might have been commonly possible
    for AF_INET sockets to end up undesirably connected to a privileged
    process, so let's make up arbitrary rules breaking all sockets instead.

    If SELinux is detected, fall back to using pipes.

    :param bool escalates_privilege:
        If :data:`True`, the target program may escalate privileges, causing
        SELinux to disconnect AF_UNIX sockets, so avoid those.
    :returns:
        `(parent_rfp, child_wfp, child_rfp, parent_wfp)`
    """
    if (not escalates_privilege) or (not SELINUX_ENABLED):
        parentfp, childfp = create_socketpair()
        return parentfp, childfp, childfp, parentfp

    parent_rfp, child_wfp = mitogen.core.pipe()
    try:
        child_rfp, parent_wfp = mitogen.core.pipe()
        return parent_rfp, child_wfp, child_rfp, parent_wfp
    except:
        parent_rfp.close()
        child_wfp.close()
        raise


def popen(**kwargs):
    """
    Wrap :class:`subprocess.Popen` to ensure any global :data:`_preexec_hook`
    is invoked in the child.
    """
    real_preexec_fn = kwargs.pop("preexec_fn", None)

    def preexec_fn():
        if _preexec_hook:
            _preexec_hook()
        if real_preexec_fn:
            real_preexec_fn()

    return subprocess.Popen(preexec_fn=preexec_fn, **kwargs)


def create_child(
    args,
    merge_stdio=False,
    stderr_pipe=False,
    escalates_privilege=False,
    preexec_fn=None,
):
    """
    Create a child process whose stdin/stdout is connected to a socket.

    :param list args:
        Program argument vector.
    :param bool merge_stdio:
        If :data:`True`, arrange for `stderr` to be connected to the `stdout`
        socketpair, rather than inherited from the parent process. This may be
        necessary to ensure that no TTY is connected to any stdio handle, for
        instance when using LXC.
    :param bool stderr_pipe:
        If :data:`True` and `merge_stdio` is :data:`False`, arrange for
        `stderr` to be connected to a separate pipe, to allow any ongoing debug
        logs generated by e.g. SSH to be output as the session progresses,
        without interfering with `stdout`.
    :param bool escalates_privilege:
        If :data:`True`, the target program may escalate privileges, causing
        SELinux to disconnect AF_UNIX sockets, so avoid those.
    :param function preexec_fn:
        If not :data:`None`, a function to run within the post-fork child
        before executing the target program.
    :returns:
        :class:`Process` instance.
    """
    parent_rfp, child_wfp, child_rfp, parent_wfp = create_best_pipe(
        escalates_privilege=escalates_privilege
    )

    stderr = None
    stderr_r = None
    if merge_stdio:
        stderr = child_wfp
    elif stderr_pipe:
        stderr_r, stderr = mitogen.core.pipe()
        mitogen.core.set_cloexec(stderr_r.fileno())

    try:
        proc = popen(
            args=args,
            stdin=child_rfp,
            stdout=child_wfp,
            stderr=stderr,
            close_fds=True,
            preexec_fn=preexec_fn,
        )
    except:
        child_rfp.close()
        child_wfp.close()
        parent_rfp.close()
        parent_wfp.close()
        if stderr_pipe:
            stderr.close()
            stderr_r.close()
        raise

    child_rfp.close()
    child_wfp.close()
    if stderr_pipe:
        stderr.close()

    return PopenProcess(
        proc=proc,
        stdin=parent_wfp,
        stdout=parent_rfp,
        stderr=stderr_r,
    )


def _acquire_controlling_tty():
    os.setsid()
    if sys.platform in ("linux", "linux2"):
        # On Linux, the controlling tty becomes the first tty opened by a
        # process lacking any prior tty.
        os.close(os.open(os.ttyname(2), os.O_RDWR))
    if hasattr(termios, "TIOCSCTTY") and not mitogen.core.IS_WSL:
        # #550: prehistoric WSL does not like TIOCSCTTY.
        # On BSD an explicit ioctl is required. For some inexplicable reason,
        # Python 2.6 on Travis also requires it.
        fcntl.ioctl(2, termios.TIOCSCTTY)


def _linux_broken_devpts_openpty():
    """
    #462: On broken Linux hosts with mismatched configuration (e.g. old
    /etc/fstab template installed), /dev/pts may be mounted without the gid=
    mount option, causing new slave devices to be created with the group ID of
    the calling process. This upsets glibc, whose openpty() is required by
    specification to produce a slave owned by a special group ID (which is
    always the 'tty' group).

    Glibc attempts to use "pt_chown" to fix ownership. If that fails, it
    chown()s the PTY directly, which fails due to non-root, causing openpty()
    to fail with EPERM ("Operation not permitted"). Since we don't need the
    magical TTY group to run sudo and su, open the PTY ourselves in this case.
    """
    master_fd = None
    try:
        # Opening /dev/ptmx causes a PTY pair to be allocated, and the
        # corresponding slave /dev/pts/* device to be created, owned by UID/GID
        # matching this process.
        master_fd = os.open("/dev/ptmx", os.O_RDWR)
        # Clear the lock bit from the PTY. This a prehistoric feature from a
        # time when slave device files were persistent.
        fcntl.ioctl(master_fd, LINUX_TIOCSPTLCK, struct.pack("i", 0))
        # Since v4.13 TIOCGPTPEER exists to open the slave in one step, but we
        # must support older kernels. Ask for the PTY number.
        pty_num_s = fcntl.ioctl(master_fd, LINUX_TIOCGPTN, struct.pack("i", 0))
        (pty_num,) = struct.unpack("i", pty_num_s)
        pty_name = "/dev/pts/%d" % (pty_num,)
        # Now open it with O_NOCTTY to ensure it doesn't change our controlling
        # TTY. Otherwise when we close the FD we get killed by the kernel, and
        # the child we spawn that should really attach to it will get EPERM
        # during _acquire_controlling_tty().
        slave_fd = os.open(pty_name, os.O_RDWR | os.O_NOCTTY)
        return master_fd, slave_fd
    except OSError:
        if master_fd is not None:
            os.close(master_fd)
        e = sys.exc_info()[1]
        raise mitogen.core.StreamError(OPENPTY_MSG, e)


def openpty():
    """
    Call :func:`os.openpty`, raising a descriptive error if the call fails.

    :raises mitogen.core.StreamError:
        Creating a PTY failed.
    :returns:
        `(master_fp, slave_fp)` file-like objects.
    """
    try:
        master_fd, slave_fd = os.openpty()
    except OSError:
        e = sys.exc_info()[1]
        if not (IS_LINUX and e.args[0] == errno.EPERM):
            raise mitogen.core.StreamError(OPENPTY_MSG, e)
        master_fd, slave_fd = _linux_broken_devpts_openpty()

    master_fp = os.fdopen(master_fd, "r+b", 0)
    slave_fp = os.fdopen(slave_fd, "r+b", 0)
    disable_echo(master_fd)
    disable_echo(slave_fd)
    mitogen.core.set_block(slave_fd)
    return master_fp, slave_fp


def tty_create_child(args):
    """
    Return a file descriptor connected to the master end of a pseudo-terminal,
    whose slave end is connected to stdin/stdout/stderr of a new child process.
    The child is created such that the pseudo-terminal becomes its controlling
    TTY, ensuring access to /dev/tty returns a new file descriptor open on the
    slave end.

    :param list args:
        Program argument vector.
    :returns:
        :class:`Process` instance.
    """
    master_fp, slave_fp = openpty()
    try:
        proc = popen(
            args=args,
            stdin=slave_fp,
            stdout=slave_fp,
            stderr=slave_fp,
            preexec_fn=_acquire_controlling_tty,
            close_fds=True,
        )
    except:
        master_fp.close()
        slave_fp.close()
        raise

    slave_fp.close()
    return PopenProcess(
        proc=proc,
        stdin=master_fp,
        stdout=master_fp,
    )


def hybrid_tty_create_child(args, escalates_privilege=False):
    """
    Like :func:`tty_create_child`, except attach stdin/stdout to a socketpair
    like :func:`create_child`, but leave stderr and the controlling TTY
    attached to a TTY.

    This permits high throughput communication with programs that are reached
    via some program that requires a TTY for password input, like many
    configurations of sudo. The UNIX TTY layer tends to have tiny (no more than
    14KiB) buffers, forcing many IO loop iterations when transferring bulk
    data, causing significant performance loss.

    :param bool escalates_privilege:
        If :data:`True`, the target program may escalate privileges, causing
        SELinux to disconnect AF_UNIX sockets, so avoid those.
    :param list args:
        Program argument vector.
    :returns:
        :class:`Process` instance.
    """
    master_fp, slave_fp = openpty()
    try:
        parent_rfp, child_wfp, child_rfp, parent_wfp = create_best_pipe(
            escalates_privilege=escalates_privilege,
        )
        try:
            mitogen.core.set_block(child_rfp)
            mitogen.core.set_block(child_wfp)
            proc = popen(
                args=args,
                stdin=child_rfp,
                stdout=child_wfp,
                stderr=slave_fp,
                preexec_fn=_acquire_controlling_tty,
                close_fds=True,
            )
        except:
            parent_rfp.close()
            child_wfp.close()
            parent_wfp.close()
            child_rfp.close()
            raise
    except:
        master_fp.close()
        slave_fp.close()
        raise

    slave_fp.close()
    child_rfp.close()
    child_wfp.close()
    return PopenProcess(
        proc=proc,
        stdin=parent_wfp,
        stdout=parent_rfp,
        stderr=master_fp,
    )


class Timer(object):
    """
    Represents a future event.
    """

    #: Set to :data:`False` if :meth:`cancel` has been called, or immediately
    #: prior to being executed by :meth:`TimerList.expire`.
    active = True

    def __init__(self, when, func):
        self.when = when
        self.func = func

    def __repr__(self):
        return f"Timer({self.when!r}, {self.func!r})"

    def __eq__(self, other):
        return self.when == other.when

    def __lt__(self, other):
        return self.when < other.when

    def __le__(self, other):
        return self.when <= other.when

    def cancel(self):
        """
        Cancel this event. If it has not yet executed, it will not execute
        during any subsequent :meth:`TimerList.expire` call.
        """
        self.active = False


class TimerList(object):
    """
    Efficiently manage a list of cancellable future events relative to wall
    clock time. An instance of this class is installed as
    :attr:`mitogen.master.Broker.timers` by default, and as
    :attr:`mitogen.core.Broker.timers` in children after a call to
    :func:`mitogen.parent.upgrade_router`.

    You can use :class:`TimerList` to cause the broker to wake at arbitrary
    future moments, useful for implementing timeouts and polling in an
    asynchronous context.

    :class:`TimerList` methods can only be called from asynchronous context,
    for example via :meth:`mitogen.core.Broker.defer`.

    The broker automatically adjusts its sleep delay according to the installed
    timer list, and arranges for timers to expire via automatic calls to
    :meth:`expire`. The main user interface to :class:`TimerList` is
    :meth:`schedule`.
    """

    _now = mitogen.core.now

    def __init__(self):
        self._lst = []

    def get_timeout(self):
        """
        Return the floating point seconds until the next event is due.

        :returns:
            Floating point delay, or 0.0, or :data:`None` if no events are
            scheduled.
        """
        while self._lst and not self._lst[0].active:
            heapq.heappop(self._lst)
        if self._lst:
            return max(0, self._lst[0].when - self._now())

    def schedule(self, when, func):
        """
        Schedule a future event.

        :param float when:
            UNIX time in seconds when event should occur.
        :param callable func:
            Callable to invoke on expiry.
        :returns:
            A :class:`Timer` instance, exposing :meth:`Timer.cancel`, which may
            be used to cancel the future invocation.
        """
        timer = Timer(when, func)
        heapq.heappush(self._lst, timer)
        return timer

    def expire(self):
        """
        Invoke callbacks for any events in the past.
        """
        now = self._now()
        while self._lst and self._lst[0].when <= now:
            timer = heapq.heappop(self._lst)
            if timer.active:
                timer.active = False
                timer.func()


class PartialZlib(object):
    """
    Because the mitogen.core source has a line appended to it during bootstrap,
    it must be recompressed for each connection. This is not a problem for a
    small number of connections, but it amounts to 30 seconds CPU time by the
    time 500 targets are in use.

    For that reason, build a compressor containing mitogen.core and flush as
    much of it as possible into an initial buffer. Then to append the custom
    line, clone the compressor and compress just that line.

    A full compression costs ~6ms on a modern machine, this method costs ~35
    usec.
    """

    def __init__(self, s):
        self.s = s
        if sys.version_info > (2, 5):
            self._compressor = zlib.compressobj(9)
            self._out = self._compressor.compress(s)
            self._out += self._compressor.flush(zlib.Z_SYNC_FLUSH)
        else:
            self._compressor = None

    def append(self, s):
        """
        Append the bytestring `s` to the compressor state and return the
        final compressed output.
        """
        if self._compressor is None:
            return zlib.compress(self.s + s, 9)
        else:
            compressor = self._compressor.copy()
            out = self._out
            out += compressor.compress(s)
            return out + compressor.flush()


def _upgrade_broker(broker):
    """
    Extract the poller state from Broker and replace it with the industrial
    strength poller for this OS. Must run on the Broker thread.
    """
    # This function is deadly! The act of calling start_receive() generates log
    # messages which must be silenced as the upgrade progresses, otherwise the
    # poller state will change as it is copied, resulting in write fds that are
    # lost. (Due to LogHandler->Router->Stream->Protocol->Broker->Poller, where
    # Stream only calls start_transmit() when transitioning from empty to
    # non-empty buffer. If the start_transmit() is lost, writes from the child
    # hang permanently).
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.CRITICAL)
    try:
        old = broker.poller
        new = PREFERRED_POLLER()
        for fd, data in old.readers:
            new.start_receive(fd, data)
        for fd, data in old.writers:
            new.start_transmit(fd, data)

        old.close()
        broker.poller = new
    finally:
        root.setLevel(old_level)

    broker.timers = TimerList()
    LOG.debug(
        "upgraded %r with %r (new: %d readers, %d writers; "
        "old: %d readers, %d writers)",
        old,
        new,
        len(new.readers),
        len(new.writers),
        len(old.readers),
        len(old.writers),
    )


@mitogen.core.takes_econtext
def upgrade_router(econtext):
    if not isinstance(econtext.router, Router):  # TODO
        econtext.broker.defer(_upgrade_broker, econtext.broker)
        econtext.router.__class__ = Router  # TODO
        econtext.router.upgrade(
            importer=econtext.importer,
            parent=econtext.parent,
        )


def get_connection_class(name):
    """
    Given the name of a Mitogen connection method, import its implementation
    module and return its Stream subclass.
    """
    if name == u"local":
        name = u"parent"
    module = mitogen.core.import_module(u"mitogen." + name)
    return module.Connection


@mitogen.core.takes_econtext
def _proxy_connect(name, method_name, kwargs, econtext):
    """
    Implements the target portion of Router._proxy_connect() by upgrading the
    local process to a parent if it was not already, then calling back into
    Router._connect() using the arguments passed to the parent's
    Router.connect().

    :returns:
        Dict containing:
        * ``id``: :data:`None`, or integer new context ID.
        * ``name``: :data:`None`, or string name attribute of new Context.
        * ``msg``: :data:`None`, or StreamError exception text.
    """
    upgrade_router(econtext)

    try:
        context = econtext.router._connect(
            klass=get_connection_class(method_name), name=name, **kwargs
        )
    except mitogen.core.StreamError:
        return {
            u"id": None,
            u"name": None,
            u"msg": "error occurred on host %s: %s"
            % (
                socket.gethostname(),
                sys.exc_info()[1],
            ),
        }

    return {
        u"id": context.context_id,
        u"name": context.name,
        u"msg": None,
    }


def returncode_to_str(n):
    """
    Parse and format a :func:`os.waitpid` exit status.
    """
    if n < 0:
        return "exited due to signal %d (%s)" % (-n, SIGNAL_BY_NUM.get(-n))
    return "exited with return code %d" % (n,)


class EofError(mitogen.core.StreamError):
    """
    Raised by :class:`Connection` when an empty read is detected from the
    remote process before bootstrap completes.
    """

    # inherits from StreamError to maintain compatibility.
    pass


class CancelledError(mitogen.core.StreamError):
    """
    Raised by :class:`Connection` when :meth:`mitogen.core.Broker.shutdown` is
    called before bootstrap completes.
    """

    pass


class Argv(object):
    """
    Wrapper to defer argv formatting when debug logging is disabled.
    """

    def __init__(self, argv):
        self.argv = argv

    must_escape = frozenset('\\$"`!')
    must_escape_or_space = must_escape | frozenset(" ")

    def escape(self, x):
        if not self.must_escape_or_space.intersection(x):
            return x

        s = '"'
        for c in x:
            if c in self.must_escape:
                s += "\\"
            s += c
        s += '"'
        return s

    def __str__(self):
        return " ".join(map(self.escape, self.argv))


class CallSpec(object):
    """
    Wrapper to defer call argument formatting when debug logging is disabled.
    """

    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _get_name(self):
        bits = [self.func.__module__]
        if inspect.ismethod(self.func):
            im_self = getattr(self.func, IM_SELF_ATTR)
            bits.append(
                getattr(im_self, "__name__", None)
                or getattr(type(im_self), "__name__", None)
            )
        bits.append(self.func.__name__)
        return u".".join(bits)

    def _get_args(self):
        return u", ".join(repr(a) for a in self.args)

    def _get_kwargs(self):
        s = u""
        if self.kwargs:
            s = u", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
            if self.args:
                s = u", " + s
        return s

    def __repr__(self):
        return f"{self._get_name()}({self._get_args()}{self._get_kwargs()})"


class PollPoller(mitogen.core.Poller):
    """
    Poller based on the POSIX :linux:man2:`poll` interface. Not available on
    some versions of OS X, otherwise it is the preferred poller for small FD
    counts, as there is no setup/teardown/configuration system call overhead.
    """

    SUPPORTED = hasattr(select, "poll")
    _repr = "PollPoller()"

    def __init__(self):
        super(PollPoller, self).__init__()
        self._pollobj = select.poll()

    # TODO: no proof we dont need writemask too
    _readmask = getattr(select, "POLLIN", 0) | getattr(select, "POLLHUP", 0)

    def _update(self, fd):
        mask = ((fd in self._rfds) and self._readmask) | (
            (fd in self._wfds) and select.POLLOUT
        )
        if mask:
            self._pollobj.register(fd, mask)
        else:
            try:
                self._pollobj.unregister(fd)
            except KeyError:
                pass

    def _poll(self, timeout):
        if timeout:
            timeout *= 1000

        events, _ = mitogen.core.io_op(self._pollobj.poll, timeout)
        for fd, event in events:
            if event & self._readmask:
                IOLOG.debug("%r: POLLIN|POLLHUP for %r", self, fd)
                data, gen = self._rfds.get(fd, (None, None))
                if gen and gen < self._generation:
                    yield data
            if event & select.POLLOUT:
                IOLOG.debug("%r: POLLOUT for %r", self, fd)
                data, gen = self._wfds.get(fd, (None, None))
                if gen and gen < self._generation:
                    yield data


class KqueuePoller(mitogen.core.Poller):
    """
    Poller based on the FreeBSD/Darwin :freebsd:man2:`kqueue` interface.
    """

    SUPPORTED = hasattr(select, "kqueue")
    _repr = "KqueuePoller()"

    def __init__(self):
        super(KqueuePoller, self).__init__()
        self._kqueue = select.kqueue()
        self._changelist = []

    def close(self):
        super(KqueuePoller, self).close()
        self._kqueue.close()

    def _control(self, fd, filters, flags):
        mitogen.core._vv and IOLOG.debug(
            "%r._control(%r, %r, %r)", self, fd, filters, flags
        )
        # TODO: at shutdown it is currently possible for KQ_EV_ADD/KQ_EV_DEL
        # pairs to be pending after the associated file descriptor has already
        # been closed. Fixing this requires maintaining extra state, or perhaps
        # making fd closure the poller's responsibility. In the meantime,
        # simply apply changes immediately.
        # self._changelist.append(select.kevent(fd, filters, flags))
        changelist = [select.kevent(fd, filters, flags)]
        events, _ = mitogen.core.io_op(self._kqueue.control, changelist, 0, 0)
        assert not events

    def start_receive(self, fd, data=None):
        mitogen.core._vv and IOLOG.debug("%r.start_receive(%r, %r)", self, fd, data)
        if fd not in self._rfds:
            self._control(fd, select.KQ_FILTER_READ, select.KQ_EV_ADD)
        self._rfds[fd] = (data or fd, self._generation)

    def stop_receive(self, fd):
        mitogen.core._vv and IOLOG.debug("%r.stop_receive(%r)", self, fd)
        if fd in self._rfds:
            self._control(fd, select.KQ_FILTER_READ, select.KQ_EV_DELETE)
            del self._rfds[fd]

    def start_transmit(self, fd, data=None):
        mitogen.core._vv and IOLOG.debug("%r.start_transmit(%r, %r)", self, fd, data)
        if fd not in self._wfds:
            self._control(fd, select.KQ_FILTER_WRITE, select.KQ_EV_ADD)
        self._wfds[fd] = (data or fd, self._generation)

    def stop_transmit(self, fd):
        mitogen.core._vv and IOLOG.debug("%r.stop_transmit(%r)", self, fd)
        if fd in self._wfds:
            self._control(fd, select.KQ_FILTER_WRITE, select.KQ_EV_DELETE)
            del self._wfds[fd]

    def _poll(self, timeout):
        changelist = self._changelist
        self._changelist = []
        events, _ = mitogen.core.io_op(self._kqueue.control, changelist, 32, timeout)
        for event in events:
            fd = event.ident
            if event.flags & select.KQ_EV_ERROR:
                LOG.debug(
                    "ignoring stale event for fd %r: errno=%d: %s",
                    fd,
                    event.data,
                    errno.errorcode.get(event.data),
                )
            elif event.filter == select.KQ_FILTER_READ:
                data, gen = self._rfds.get(fd, (None, None))
                # Events can still be read for an already-discarded fd.
                if gen and gen < self._generation:
                    mitogen.core._vv and IOLOG.debug("%r: POLLIN: %r", self, fd)
                    yield data
            elif event.filter == select.KQ_FILTER_WRITE and fd in self._wfds:
                data, gen = self._wfds.get(fd, (None, None))
                if gen and gen < self._generation:
                    mitogen.core._vv and IOLOG.debug("%r: POLLOUT: %r", self, fd)
                    yield data


class EpollPoller(mitogen.core.Poller):
    """
    Poller based on the Linux :linux:man2:`epoll` interface.
    """

    SUPPORTED = hasattr(select, "epoll")
    _repr = "EpollPoller()"

    def __init__(self):
        super(EpollPoller, self).__init__()
        self._epoll = select.epoll(32)
        self._registered_fds = set()

    def close(self):
        super(EpollPoller, self).close()
        self._epoll.close()

    def _control(self, fd):
        mitogen.core._vv and IOLOG.debug("%r._control(%r)", self, fd)
        mask = ((fd in self._rfds) and select.EPOLLIN) | (
            (fd in self._wfds) and select.EPOLLOUT
        )
        if mask:
            if fd in self._registered_fds:
                self._epoll.modify(fd, mask)
            else:
                self._epoll.register(fd, mask)
                self._registered_fds.add(fd)
        elif fd in self._registered_fds:
            self._epoll.unregister(fd)
            self._registered_fds.remove(fd)

    def start_receive(self, fd, data=None):
        mitogen.core._vv and IOLOG.debug("%r.start_receive(%r, %r)", self, fd, data)
        self._rfds[fd] = (data or fd, self._generation)
        self._control(fd)

    def stop_receive(self, fd):
        mitogen.core._vv and IOLOG.debug("%r.stop_receive(%r)", self, fd)
        self._rfds.pop(fd, None)
        self._control(fd)

    def start_transmit(self, fd, data=None):
        mitogen.core._vv and IOLOG.debug("%r.start_transmit(%r, %r)", self, fd, data)
        self._wfds[fd] = (data or fd, self._generation)
        self._control(fd)

    def stop_transmit(self, fd):
        mitogen.core._vv and IOLOG.debug("%r.stop_transmit(%r)", self, fd)
        self._wfds.pop(fd, None)
        self._control(fd)

    _inmask = getattr(select, "EPOLLIN", 0) | getattr(select, "EPOLLHUP", 0)

    def _poll(self, timeout):
        the_timeout = -1
        if timeout is not None:
            the_timeout = timeout

        events, _ = mitogen.core.io_op(self._epoll.poll, the_timeout, 32)
        for fd, event in events:
            if event & self._inmask:
                data, gen = self._rfds.get(fd, (None, None))
                if gen and gen < self._generation:
                    # Events can still be read for an already-discarded fd.
                    mitogen.core._vv and IOLOG.debug("%r: POLLIN: %r", self, fd)
                    yield data
            if event & select.EPOLLOUT:
                data, gen = self._wfds.get(fd, (None, None))
                if gen and gen < self._generation:
                    mitogen.core._vv and IOLOG.debug("%r: POLLOUT: %r", self, fd)
                    yield data


# 2.4 and 2.5 only had select.select() and select.poll().
for _klass in mitogen.core.Poller, PollPoller, KqueuePoller, EpollPoller:
    if _klass.SUPPORTED:
        PREFERRED_POLLER = _klass

# For processes that start many threads or connections, it's possible Latch
# will also get high-numbered FDs, and so select() becomes useless there too.
# So swap in our favourite poller.
if PollPoller.SUPPORTED:
    mitogen.core.Latch.poller_class = PollPoller
else:
    mitogen.core.Latch.poller_class = PREFERRED_POLLER


class LineLoggingProtocolMixin(object):
    def __init__(self, **kwargs):
        super(LineLoggingProtocolMixin, self).__init__(**kwargs)
        self.logged_lines = []
        self.logged_partial = None

    def on_line_received(self, line):
        self.logged_partial = None
        self.logged_lines.append((mitogen.core.now(), line))
        self.logged_lines[:] = self.logged_lines[-100:]
        return super(LineLoggingProtocolMixin, self).on_line_received(line)

    def on_partial_line_received(self, line):
        self.logged_partial = line
        return super(LineLoggingProtocolMixin, self).on_partial_line_received(line)

    def on_disconnect(self, broker):
        if self.logged_partial:
            self.logged_lines.append((mitogen.core.now(), self.logged_partial))
            self.logged_partial = None
        super(LineLoggingProtocolMixin, self).on_disconnect(broker)


def get_history(streams):
    history = []
    for stream in streams:
        if stream:
            history.extend(getattr(stream.protocol, "logged_lines", []))
    history.sort()

    s = b("\n").join(h[1] for h in history)
    return mitogen.core.to_text(s)


class RegexProtocol(LineLoggingProtocolMixin, mitogen.core.DelimitedProtocol):
    """
    Implement a delimited protocol where messages matching a set of regular
    expressions are dispatched to individual handler methods. Input is
    dispatches using :attr:`PATTERNS` and :attr:`PARTIAL_PATTERNS`, before
    falling back to :meth:`on_unrecognized_line_received` and
    :meth:`on_unrecognized_partial_line_received`.
    """

    #: A sequence of 2-tuples of the form `(compiled pattern, method)` for
    #: patterns that should be matched against complete (delimited) messages,
    #: i.e. full lines.
    PATTERNS = []

    #: Like :attr:`PATTERNS`, but patterns that are matched against incomplete
    #: lines.
    PARTIAL_PATTERNS = []

    def on_line_received(self, line):
        super(RegexProtocol, self).on_line_received(line)
        for pattern, func in self.PATTERNS:
            match = pattern.search(line)
            if match is not None:
                return func(self, line, match)

        return self.on_unrecognized_line_received(line)

    def on_unrecognized_line_received(self, line):
        LOG.debug(
            "%s: (unrecognized): %s", self.stream.name, line.decode("utf-8", "replace")
        )

    def on_partial_line_received(self, line):
        super(RegexProtocol, self).on_partial_line_received(line)
        LOG.debug(
            "%s: (partial): %s", self.stream.name, line.decode("utf-8", "replace")
        )
        for pattern, func in self.PARTIAL_PATTERNS:
            match = pattern.search(line)
            if match is not None:
                return func(self, line, match)

        return self.on_unrecognized_partial_line_received(line)

    def on_unrecognized_partial_line_received(self, line):
        LOG.debug(
            "%s: (unrecognized partial): %s",
            self.stream.name,
            line.decode("utf-8", "replace"),
        )


class BootstrapProtocol(RegexProtocol):
    """
    Respond to stdout of a child during bootstrap. Wait for :attr:`EC0_MARKER`
    to be written by the first stage to indicate it can receive the bootstrap,
    then await :attr:`EC1_MARKER` to indicate success, and
    :class:`MitogenProtocol` can be enabled.
    """

    #: Sentinel value emitted by the first stage to indicate it is ready to
    #: receive the compressed bootstrap. For :mod:`mitogen.ssh` this must have
    #: length of at least `max(len('password'), len('debug1:'))`
    EC0_MARKER = b("MITO000")
    EC1_MARKER = b("MITO001")
    EC2_MARKER = b("MITO002")

    def __init__(self, broker):
        super(BootstrapProtocol, self).__init__()
        self._writer = mitogen.core.BufferedWriter(broker, self)

    def on_transmit(self, broker):
        self._writer.on_transmit(broker)

    def _on_ec0_received(self, line, match):
        LOG.debug("%r: first stage started succcessfully", self)
        self._writer.write(self.stream.conn.get_preamble())

    def _on_ec1_received(self, line, match):
        LOG.debug("%r: first stage received mitogen.core source", self)

    def _on_ec2_received(self, line, match):
        LOG.debug("%r: new child booted successfully", self)
        self.stream.conn._complete_connection()
        return False

    def on_unrecognized_line_received(self, line):
        LOG.debug("%s: stdout: %s", self.stream.name, line.decode("utf-8", "replace"))

    PATTERNS = [
        (re.compile(EC0_MARKER), _on_ec0_received),
        (re.compile(EC1_MARKER), _on_ec1_received),
        (re.compile(EC2_MARKER), _on_ec2_received),
    ]


class LogProtocol(LineLoggingProtocolMixin, mitogen.core.DelimitedProtocol):
    """
    For "hybrid TTY/socketpair" mode, after connection setup a spare TTY master
    FD exists that cannot be closed, and to which SSH or sudo may continue
    writing log messages.

    The descriptor cannot be closed since the UNIX TTY layer sends SIGHUP to
    processes whose controlling TTY is the slave whose master side was closed.
    LogProtocol takes over this FD and creates log messages for anything
    written to it.
    """

    def on_line_received(self, line):
        """
        Read a line, decode it as UTF-8, and log it.
        """
        super(LogProtocol, self).on_line_received(line)
        LOG.info(u"%s: %s", self.stream.name, line.decode("utf-8", "replace"))


class MitogenProtocol(mitogen.core.MitogenProtocol):
    """
    Extend core.MitogenProtocol to cause SHUTDOWN to be sent to the child
    during graceful shutdown.
    """

    def on_shutdown(self, broker):
        """
        Respond to the broker's request for the stream to shut down by sending
        SHUTDOWN to the child.
        """
        LOG.debug("%r: requesting child shutdown", self)
        self._send(
            mitogen.core.Message(
                src_id=mitogen.context_id,
                dst_id=self.remote_id,
                handle=mitogen.core.SHUTDOWN,
            )
        )


class Options(object):
    name = None

    #: The path to the remote Python interpreter.
    python_path = get_sys_executable()

    #: Maximum time to wait for a connection attempt.
    connect_timeout = 30.0

    #: True to cause context to write verbose /tmp/mitogen.<pid>.log.
    debug = False

    #: True to cause context to write /tmp/mitogen.stats.<pid>.<thread>.log.
    profiling = False

    #: True if unidirectional routing is enabled in the new child.
    unidirectional = False

    #: Passed via Router wrapper methods, must eventually be passed to
    #: ExternalContext.main().
    max_message_size = None

    #: Remote name.
    remote_name = None

    #: Derived from :py:attr:`connect_timeout`; absolute floating point
    #: UNIX timestamp after which the connection attempt should be abandoned.
    connect_deadline = None

    def __init__(
        self,
        max_message_size,
        name=None,
        remote_name=None,
        python_path=None,
        debug=False,
        connect_timeout=None,
        profiling=False,
        unidirectional=False,
        old_router=None,
    ):
        self.name = name
        self.max_message_size = max_message_size
        if python_path:
            self.python_path = python_path
        if connect_timeout:
            self.connect_timeout = connect_timeout
        if remote_name is None:
            remote_name = get_default_remote_name()
        if "/" in remote_name or "\\" in remote_name:
            raise ValueError("remote_name= cannot contain slashes")
        if remote_name:
            self.remote_name = mitogen.core.to_text(remote_name)
        self.debug = debug
        self.profiling = profiling
        self.unidirectional = unidirectional
        self.max_message_size = max_message_size
        self.connect_deadline = mitogen.core.now() + self.connect_timeout


class Connection(object):
    """
    Manage the lifetime of a set of :class:`Streams <Stream>` connecting to a
    remote Python interpreter, including bootstrap, disconnection, and external
    tool integration.

    Base for streams capable of starting children.
    """

    options_class = Options

    #: The protocol attached to stdio of the child.
    stream_protocol_class = BootstrapProtocol

    #: The protocol attached to stderr of the child.
    diag_protocol_class = LogProtocol

    #: :class:`Process`
    proc = None

    #: :class:`mitogen.core.Stream` with sides connected to stdin/stdout.
    stdio_stream = None

    #: If `proc.stderr` is set, referencing either a plain pipe or the
    #: controlling TTY, this references the corresponding
    #: :class:`LogProtocol`'s stream, allowing it to be disconnected when this
    #: stream is disconnected.
    stderr_stream = None

    #: Function with the semantics of :func:`create_child` used to create the
    #: child process.
    create_child = staticmethod(create_child)

    #: Dictionary of extra kwargs passed to :attr:`create_child`.
    create_child_args = {}

    #: :data:`True` if the remote has indicated that it intends to detach, and
    #: should not be killed on disconnect.
    detached = False

    #: If :data:`True`, indicates the child should not be killed during
    #: graceful detachment, as it the actual process implementing the child
    #: context. In all other cases, the subprocess is SSH, sudo, or a similar
    #: tool that should be reminded to quit during disconnection.
    child_is_immediate_subprocess = True

    #: Prefix given to default names generated by :meth:`connect`.
    name_prefix = u"local"

    #: :class:`Timer` that runs :meth:`_on_timer_expired` when connection
    #: timeout occurs.
    _timer = None

    #: When disconnection completes, instance of :class:`Reaper` used to wait
    #: on the exit status of the subprocess.
    _reaper = None

    #: On failure, the exception object that should be propagated back to the
    #: user.
    exception = None

    #: Extra text appended to :class:`EofError` if that exception is raised on
    #: a failed connection attempt. May be used in subclasses to hint at common
    #: problems with a particular connection method.
    eof_error_hint = None

    def __init__(self, options, router):
        #: :class:`Options`
        self.options = options
        self._router = router

    def __repr__(self):
        return f"Connection({self.stdio_stream!r})"

    # Minimised, gzipped, base64'd and passed to 'python -c'. It forks, dups
    # file descriptor 0 as 100, creates a pipe, then execs a new interpreter
    # with a custom argv.
    #   * Optimized for minimum byte count after minification & compression.
    #   * 'CONTEXT_NAME' and 'PREAMBLE_COMPRESSED_LEN' are substituted with
    #     their respective values.
    #   * CONTEXT_NAME must be prefixed with the name of the Python binary in
    #     order to allow virtualenvs to detect their install prefix.
    #   * For Darwin, OS X installs a craptacular argv0-introspecting Python
    #     version switcher as /usr/bin/python. Override attempts to call it
    #     with an explicit call to python2.7
    #
    # Locals:
    #   R: read side of interpreter stdin.
    #   W: write side of interpreter stdin.
    #   r: read side of core_src FD.
    #   w: write side of core_src FD.
    #   C: the decompressed core source.

    # Final os.close(2) to avoid --py-debug build from corrupting stream with
    # "[1234 refs]" during exit.
    @staticmethod
    def _first_stage():
        R, W = os.pipe()
        r, w = os.pipe()
        if os.fork():
            os.dup2(0, 100)
            os.dup2(R, 0)
            os.dup2(r, 101)
            os.close(R)
            os.close(r)
            os.close(W)
            os.close(w)
            # this doesn't apply anymore to Mac OSX 10.15+ (Darwin 19+), new interpreter looks like this:
            # /System/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python
            if (
                sys.platform == "darwin"
                and sys.executable == "/usr/bin/python"
                and int(platform.release()[:2]) < 19
            ):
                sys.executable += sys.version[:3]
            os.environ["ARGV0"] = sys.executable
            os.execl(sys.executable, sys.executable + "(mitogen:CONTEXT_NAME)")
        os.write(1, "MITO000\n".encode())
        C = _(os.fdopen(0, "rb").read(PREAMBLE_COMPRESSED_LEN), "zip")
        fp = os.fdopen(W, "wb", 0)
        fp.write(C)
        fp.close()
        fp = os.fdopen(w, "wb", 0)
        fp.write(C)
        fp.close()
        os.write(1, "MITO001\n".encode())
        os.close(2)

    def get_python_argv(self):
        """
        Return the initial argument vector elements necessary to invoke Python,
        by returning a 1-element list containing :attr:`python_path` if it is a
        string, or simply returning it if it is already a list.

        This allows emulation of existing tools where the Python invocation may
        be set to e.g. `['/usr/bin/env', 'python']`.
        """
        if isinstance(self.options.python_path, list):
            return self.options.python_path
        return [self.options.python_path]

    def get_boot_command(self):
        source = inspect.getsource(self._first_stage)
        source = textwrap.dedent("\n".join(source.strip().split("\n")[2:]))
        source = source.replace("    ", "\t")
        source = source.replace("CONTEXT_NAME", self.options.remote_name)
        preamble_compressed = self.get_preamble()
        source = source.replace(
            "PREAMBLE_COMPRESSED_LEN", str(len(preamble_compressed))
        )
        compressed = zlib.compress(source.encode(), 9)
        encoded = codecs.encode(compressed, "base64").replace(b("\n"), b(""))
        # We can't use bytes.decode() in 3.x since it was restricted to always
        # return unicode, so codecs.decode() is used instead. In 3.x
        # codecs.decode() requires a bytes object. Since we must be compatible
        # with 2.4 (no bytes literal), an extra .encode() either returns the
        # same str (2.x) or an equivalent bytes (3.x).
        return self.get_python_argv() + [
            "-c",
            "import codecs,os,sys;_=codecs.decode;"
            'exec(_(_("%s".encode(),"base64"),"zip"))' % (encoded.decode(),),
        ]

    def get_econtext_config(self):
        assert self.options.max_message_size is not None
        parent_ids = mitogen.parent_ids[:]
        parent_ids.insert(0, mitogen.context_id)
        return {
            "parent_ids": parent_ids,
            "context_id": self.context.context_id,
            "debug": self.options.debug,
            "profiling": self.options.profiling,
            "unidirectional": self.options.unidirectional,
            "log_level": get_log_level(),
            "whitelist": self._router.get_module_whitelist(),
            "blacklist": self._router.get_module_blacklist(),
            "max_message_size": self.options.max_message_size,
            "version": mitogen.__version__,
        }

    def get_preamble(self):
        suffix = f"\nExternalContext({self.get_econtext_config()!r}).main()\n"
        partial = get_core_source_partial()
        return partial.append(suffix.encode("utf-8"))

    def _get_name(self):
        """
        Called by :meth:`connect` after :attr:`pid` is known. Subclasses can
        override it to specify a default stream name, or set
        :attr:`name_prefix` to generate a default format.
        """
        return f"{self.name_prefix}.{self.proc.pid}"

    def start_child(self):
        args = self.get_boot_command()
        LOG.debug("command line for %r: %s", self, Argv(args))
        try:
            return self.create_child(args=args, **self.create_child_args)
        except OSError:
            e = sys.exc_info()[1]
            msg = f"Child start failed: {e}. Command was: {Argv(args)}"
            raise mitogen.core.StreamError(msg)

    def _adorn_eof_error(self, e):
        """
        Subclasses may provide additional information in the case of a failed
        connection.
        """
        if self.eof_error_hint:
            e.args = (f"{e.args[0]}\n\n{self.eof_error_hint}",)

    def _complete_connection(self):
        self._timer.cancel()
        if not self.exception:
            mitogen.core.unlisten(
                self._router.broker, "shutdown", self._on_broker_shutdown
            )
            self._router.register(self.context, self.stdio_stream)
            self.stdio_stream.set_protocol(
                MitogenProtocol(
                    router=self._router,
                    remote_id=self.context.context_id,
                )
            )
            self._router.route_monitor.notice_stream(self.stdio_stream)
        self.latch.put()

    def _fail_connection(self, exc):
        """
        Fail the connection attempt.
        """
        LOG.debug(
            "failing connection %s due to %r",
            self.stdio_stream and self.stdio_stream.name,
            exc,
        )
        if self.exception is None:
            self._adorn_eof_error(exc)
            self.exception = exc
            mitogen.core.unlisten(
                self._router.broker, "shutdown", self._on_broker_shutdown
            )
        for stream in self.stdio_stream, self.stderr_stream:
            if stream and not stream.receive_side.closed:
                stream.on_disconnect(self._router.broker)
        self._complete_connection()

    eof_error_msg = "EOF on stream; last 100 lines received:\n"

    def on_stdio_disconnect(self):
        """
        Handle stdio stream disconnection by failing the Connection if the
        stderr stream has already been closed. Otherwise, wait for it to close
        (or timeout), to allow buffered diagnostic logs to be consumed.

        It is normal that when a subprocess aborts, stdio has nothing buffered
        when it is closed, thus signalling readability, causing an empty read
        (interpreted as indicating disconnection) on the next loop iteration,
        even if its stderr pipe has lots of diagnostic logs still buffered in
        the kernel. Therefore we must wait for both pipes to indicate they are
        empty before triggering connection failure.
        """
        stderr = self.stderr_stream
        if stderr is None or stderr.receive_side.closed:
            self._on_streams_disconnected()

    def on_stderr_disconnect(self):
        """
        Inverse of :func:`on_stdio_disconnect`.
        """
        if self.stdio_stream.receive_side.closed:
            self._on_streams_disconnected()

    def _on_streams_disconnected(self):
        """
        When disconnection has been detected for both streams, cancel the
        connection timer, mark the connection failed, and reap the child
        process. Do nothing if the timer has already been cancelled, indicating
        some existing failure has already been noticed.
        """
        if self._timer.active:
            self._timer.cancel()
            self._fail_connection(
                EofError(
                    self.eof_error_msg
                    + get_history([self.stdio_stream, self.stderr_stream])
                )
            )

        if self._reaper:
            return

        self._reaper = Reaper(
            broker=self._router.broker,
            proc=self.proc,
            kill=not (
                (self.detached and self.child_is_immediate_subprocess)
                or
                # Avoid killing so child has chance to write cProfile data
                self._router.profiling
            ),
            # Don't delay shutdown waiting for a detached child, since the
            # detached child may expect to live indefinitely after its parent
            # exited.
            wait_on_shutdown=(not self.detached),
        )
        self._reaper.reap()

    def _on_broker_shutdown(self):
        """
        Respond to broker.shutdown() being called by failing the connection
        attempt.
        """
        self._fail_connection(CancelledError(BROKER_SHUTDOWN_MSG))

    def stream_factory(self):
        return self.stream_protocol_class.build_stream(
            broker=self._router.broker,
        )

    def stderr_stream_factory(self):
        return self.diag_protocol_class.build_stream()

    def _setup_stdio_stream(self):
        stream = self.stream_factory()
        stream.conn = self
        stream.name = self.options.name or self._get_name()
        stream.accept(self.proc.stdout, self.proc.stdin)

        mitogen.core.listen(stream, "disconnect", self.on_stdio_disconnect)
        self._router.broker.start_receive(stream)
        return stream

    def _setup_stderr_stream(self):
        stream = self.stderr_stream_factory()
        stream.conn = self
        stream.name = self.options.name or self._get_name()
        stream.accept(self.proc.stderr, self.proc.stderr)

        mitogen.core.listen(stream, "disconnect", self.on_stderr_disconnect)
        self._router.broker.start_receive(stream)
        return stream

    def _on_timer_expired(self):
        self._fail_connection(
            mitogen.core.TimeoutError(
                "Failed to setup connection after %.2f seconds",
                self.options.connect_timeout,
            )
        )

    def _async_connect(self):
        LOG.debug(
            "creating connection to context %d using %s",
            self.context.context_id,
            self.__class__.__module__,
        )
        mitogen.core.listen(self._router.broker, "shutdown", self._on_broker_shutdown)
        self._timer = self._router.broker.timers.schedule(
            when=self.options.connect_deadline,
            func=self._on_timer_expired,
        )

        try:
            self.proc = self.start_child()
        except Exception:
            LOG.debug("failed to start child", exc_info=True)
            self._fail_connection(sys.exc_info()[1])
            return

        LOG.debug(
            "child for %r started: pid:%r stdin:%r stdout:%r stderr:%r",
            self,
            self.proc.pid,
            self.proc.stdin.fileno(),
            self.proc.stdout.fileno(),
            self.proc.stderr and self.proc.stderr.fileno(),
        )

        self.stdio_stream = self._setup_stdio_stream()
        if self.context.name is None:
            self.context.name = self.stdio_stream.name
        self.proc.name = self.stdio_stream.name
        if self.proc.stderr:
            self.stderr_stream = self._setup_stderr_stream()

    def connect(self, context):
        self.context = context
        self.latch = mitogen.core.Latch()
        self._router.broker.defer(self._async_connect)
        self.latch.get()
        if self.exception:
            raise self.exception


class ChildIdAllocator(object):
    """
    Allocate new context IDs from a block of unique context IDs allocated by
    the master process.
    """

    def __init__(self, router):
        self.router = router
        self.lock = threading.Lock()
        self.it = iter(xrange(0))

    def allocate(self):
        """
        Allocate an ID, requesting a fresh block from the master if the
        existing block is exhausted.

        :returns:
            The new context ID.

        .. warning::

            This method is not safe to call from the :class:`Broker` thread, as
            it may block on IO of its own.
        """
        self.lock.acquire()
        try:
            for id_ in self.it:
                return id_

            master = self.router.context_by_id(0)
            start, end = master.send_await(
                mitogen.core.Message(dst_id=0, handle=mitogen.core.ALLOCATE_ID)
            )
            self.it = iter(xrange(start, end))
        finally:
            self.lock.release()

        return self.allocate()


class CallChain(object):
    """
    Deliver :data:`mitogen.core.CALL_FUNCTION` messages to a target context,
    optionally threading related calls so an exception in an earlier call
    cancels subsequent calls.

    :param mitogen.core.Context context:
        Target context.
    :param bool pipelined:
        Enable pipelining.

    :meth:`call`, :meth:`call_no_reply` and :meth:`call_async`
    normally issue calls and produce responses with no memory of prior
    exceptions. If a call made with :meth:`call_no_reply` fails, the exception
    is logged to the target context's logging framework.

    **Pipelining**

    When pipelining is enabled, if an exception occurs during a call,
    subsequent calls made by the same :class:`CallChain` fail with the same
    exception, including those already in-flight on the network, and no further
    calls execute until :meth:`reset` is invoked.

    No exception is logged for calls made with :meth:`call_no_reply`, instead
    the exception is saved and reported as the result of subsequent
    :meth:`call` or :meth:`call_async` calls.

    Sequences of asynchronous calls can be made without wasting network
    round-trips to discover if prior calls succeed, and chains originating from
    multiple unrelated source contexts may overlap concurrently at a target
    context without interference.

    In this example, 4 calls complete in one round-trip::

        chain = mitogen.parent.CallChain(context, pipelined=True)
        chain.call_no_reply(os.mkdir, '/tmp/foo')

        # If previous mkdir() failed, this never runs:
        chain.call_no_reply(os.mkdir, '/tmp/foo/bar')

        # If either mkdir() failed, this never runs, and the exception is
        # asynchronously delivered to the receiver.
        recv = chain.call_async(subprocess.check_output, '/tmp/foo')

        # If anything so far failed, this never runs, and raises the exception.
        chain.call(do_something)

        # If this code was executed, the exception would also be raised.
        if recv.get().unpickle() == 'baz':
            pass

    When pipelining is enabled, :meth:`reset` must be invoked to ensure any
    exception is discarded, otherwise unbounded memory usage is possible in
    long-running programs. The context manager protocol is supported to ensure
    :meth:`reset` is always invoked::

        with mitogen.parent.CallChain(context, pipelined=True) as chain:
            chain.call_no_reply(...)
            chain.call_no_reply(...)
            chain.call_no_reply(...)
            chain.call(...)

        # chain.reset() automatically invoked.
    """

    def __init__(self, context, pipelined=False):
        self.context = context
        if pipelined:
            self.chain_id = self.make_chain_id()
        else:
            self.chain_id = None

    @classmethod
    def make_chain_id(cls):
        return "%s-%s-%x-%x" % (
            socket.gethostname(),
            os.getpid(),
            thread.get_ident(),
            int(1e6 * mitogen.core.now()),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.context})"

    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        self.reset()

    def reset(self):
        """
        Instruct the target to forget any related exception.
        """
        if not self.chain_id:
            return

        saved, self.chain_id = self.chain_id, None
        try:
            self.call_no_reply(mitogen.core.Dispatcher.forget_chain, saved)
        finally:
            self.chain_id = saved

    closures_msg = (
        "Mitogen cannot invoke closures, as doing so would require "
        "serializing arbitrary program state, and no universal "
        "method exists to recover a reference to them."
    )

    lambda_msg = (
        "Mitogen cannot invoke anonymous functions, as no universal method "
        "exists to recover a reference to an anonymous function."
    )

    method_msg = (
        "Mitogen cannot invoke instance methods, as doing so would require "
        "serializing arbitrary program state."
    )

    def make_msg(self, fn, *args, **kwargs):
        if getattr(fn, closure_attr, None) is not None:
            raise TypeError(self.closures_msg)
        if fn.__name__ == "<lambda>":
            raise TypeError(self.lambda_msg)

        if inspect.ismethod(fn):
            im_self = getattr(fn, IM_SELF_ATTR)
            if not inspect.isclass(im_self):
                raise TypeError(self.method_msg)
            klass = mitogen.core.to_text(im_self.__name__)
        else:
            klass = None

        tup = (
            self.chain_id,
            mitogen.core.to_text(fn.__module__),
            klass,
            mitogen.core.to_text(fn.__name__),
            args,
            mitogen.core.Kwargs(kwargs),
        )
        return mitogen.core.Message.pickled(tup, handle=mitogen.core.CALL_FUNCTION)

    def call_no_reply(self, fn, *args, **kwargs):
        """
        Like :meth:`call_async`, but do not wait for a return value, and inform
        the target context no reply is expected. If the call fails and
        pipelining is disabled, the exception will be logged to the target
        context's logging framework.
        """
        LOG.debug(
            "starting no-reply function call to %r: %r",
            self.context.name or self.context.context_id,
            CallSpec(fn, args, kwargs),
        )
        self.context.send(self.make_msg(fn, *args, **kwargs))

    def call_async(self, fn, *args, **kwargs):
        """
        Arrange for `fn(*args, **kwargs)` to be invoked on the context's main
        thread.

        :param fn:
            A free function in module scope or a class method of a class
            directly reachable from module scope:

            .. code-block:: python

                # mymodule.py

                def my_func():
                    '''A free function reachable as mymodule.my_func'''

                class MyClass:
                    @classmethod
                    def my_classmethod(cls):
                        '''Reachable as mymodule.MyClass.my_classmethod'''

                    def my_instancemethod(self):
                        '''Unreachable: requires a class instance!'''

                    class MyEmbeddedClass:
                        @classmethod
                        def my_classmethod(cls):
                            '''Not directly reachable from module scope!'''

        :param tuple args:
            Function arguments, if any. See :ref:`serialization-rules` for
            permitted types.
        :param dict kwargs:
            Function keyword arguments, if any. See :ref:`serialization-rules`
            for permitted types.
        :returns:
            :class:`mitogen.core.Receiver` configured to receive the result of
            the invocation:

            .. code-block:: python

                recv = context.call_async(os.check_output, 'ls /tmp/')
                try:
                    # Prints output once it is received.
                    msg = recv.get()
                    print(msg.unpickle())
                except mitogen.core.CallError, e:
                    print('Call failed:', str(e))

            Asynchronous calls may be dispatched in parallel to multiple
            contexts and consumed as they complete using
            :class:`mitogen.select.Select`.
        """
        LOG.debug(
            "starting function call to %s: %r",
            self.context.name or self.context.context_id,
            CallSpec(fn, args, kwargs),
        )
        return self.context.send_async(self.make_msg(fn, *args, **kwargs))

    def call(self, fn, *args, **kwargs):
        """
        Like :meth:`call_async`, but block until the return value is available.
        Equivalent to::

            call_async(fn, *args, **kwargs).get().unpickle()

        :returns:
            The function's return value.
        :raises mitogen.core.CallError:
            An exception was raised in the remote context during execution.
        """
        receiver = self.call_async(fn, *args, **kwargs)
        return receiver.get().unpickle(throw_dead=False)


class Context(mitogen.core.Context):
    """
    Extend :class:`mitogen.core.Context` with functionality useful to masters,
    and child contexts who later become parents. Currently when this class is
    required, the target context's router is upgraded at runtime.
    """

    #: A :class:`CallChain` instance constructed by default, with pipelining
    #: disabled. :meth:`call`, :meth:`call_async` and :meth:`call_no_reply` use
    #: this instance.
    call_chain_class = CallChain

    via = None

    def __init__(self, *args, **kwargs):
        super(Context, self).__init__(*args, **kwargs)
        self.default_call_chain = self.call_chain_class(self)

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        return (
            isinstance(other, mitogen.core.Context)
            and (other.context_id == self.context_id)
            and (other.router == self.router)
        )

    def __hash__(self):
        return hash((self.router, self.context_id))

    def call_async(self, fn, *args, **kwargs):
        """
        See :meth:`CallChain.call_async`.
        """
        return self.default_call_chain.call_async(fn, *args, **kwargs)

    def call(self, fn, *args, **kwargs):
        """
        See :meth:`CallChain.call`.
        """
        return self.default_call_chain.call(fn, *args, **kwargs)

    def call_no_reply(self, fn, *args, **kwargs):
        """
        See :meth:`CallChain.call_no_reply`.
        """
        self.default_call_chain.call_no_reply(fn, *args, **kwargs)

    def shutdown(self, wait=False):
        """
        Arrange for the context to receive a ``SHUTDOWN`` message, triggering
        graceful shutdown.

        Due to a lack of support for timers, no attempt is made yet to force
        terminate a hung context using this method. This will be fixed shortly.

        :param bool wait:
            If :data:`True`, block the calling thread until the context has
            completely terminated.

        :returns:
            If `wait` is :data:`False`, returns a :class:`mitogen.core.Latch`
            whose :meth:`get() <mitogen.core.Latch.get>` method returns
            :data:`None` when shutdown completes. The `timeout` parameter may
            be used to implement graceful timeouts.
        """
        LOG.debug("%r.shutdown() sending SHUTDOWN", self)
        latch = mitogen.core.Latch()
        mitogen.core.listen(self, "disconnect", lambda: latch.put(None))
        self.send(
            mitogen.core.Message(
                handle=mitogen.core.SHUTDOWN,
            )
        )

        if wait:
            latch.get()
        else:
            return latch


class RouteMonitor(object):
    """
    Generate and respond to :data:`mitogen.core.ADD_ROUTE` and
    :data:`mitogen.core.DEL_ROUTE` messages sent to the local context by
    maintaining a table of available routes, and propagating messages towards
    parents and siblings as appropriate.

    :class:`RouteMonitor` is responsible for generating routing messages for
    directly attached children. It learns of new children via
    :meth:`notice_stream` called by :class:`Router`, and subscribes to their
    ``disconnect`` event to learn when they disappear.

    In children, constructing this class overwrites the stub
    :data:`mitogen.core.DEL_ROUTE` handler installed by
    :class:`mitogen.core.ExternalContext`, which is expected behaviour when a
    child is beging upgraded in preparation to become a parent of children of
    its own.

    By virtue of only being active while responding to messages from a handler,
    RouteMonitor lives entirely on the broker thread, so its data requires no
    locking.

    :param mitogen.master.Router router:
        Router to install handlers on.
    :param mitogen.core.Context parent:
        :data:`None` in the master process, or reference to the parent context
        we should propagate route updates towards.
    """

    def __init__(self, router, parent=None):
        self.router = router
        self.parent = parent
        self._log = logging.getLogger("mitogen.route_monitor")
        #: Mapping of Stream instance to integer context IDs reachable via the
        #: stream; used to cleanup routes during disconnection.
        self._routes_by_stream = {}
        self.router.add_handler(
            fn=self._on_add_route,
            handle=mitogen.core.ADD_ROUTE,
            persist=True,
            policy=is_immediate_child,
            overwrite=True,
        )
        self.router.add_handler(
            fn=self._on_del_route,
            handle=mitogen.core.DEL_ROUTE,
            persist=True,
            policy=is_immediate_child,
            overwrite=True,
        )

    def __repr__(self):
        return "RouteMonitor()"

    def _send_one(self, stream, handle, target_id, name):
        """
        Compose and send an update message on a stream.

        :param mitogen.core.Stream stream:
            Stream to send it on.
        :param int handle:
            :data:`mitogen.core.ADD_ROUTE` or :data:`mitogen.core.DEL_ROUTE`
        :param int target_id:
            ID of the connecting or disconnecting context.
        :param str name:
            Context name or :data:`None`.
        """
        if not stream:
            # We may not have a stream during shutdown.
            return

        data = str(target_id)
        if name:
            data = f"{target_id}:{name}"
        stream.protocol.send(
            mitogen.core.Message(
                handle=handle,
                data=data.encode("utf-8"),
                dst_id=stream.protocol.remote_id,
            )
        )

    def _propagate_up(self, handle, target_id, name=None):
        """
        In a non-master context, propagate an update towards the master.

        :param int handle:
            :data:`mitogen.core.ADD_ROUTE` or :data:`mitogen.core.DEL_ROUTE`
        :param int target_id:
            ID of the connecting or disconnecting context.
        :param str name:
            For :data:`mitogen.core.ADD_ROUTE`, the name of the new context
            assigned by its parent. This is used by parents to assign the
            :attr:`mitogen.core.Context.name` attribute.
        """
        if self.parent:
            stream = self.router.stream_by_id(self.parent.context_id)
            self._send_one(stream, handle, target_id, name)

    def _propagate_down(self, handle, target_id):
        """
        For DEL_ROUTE, we additionally want to broadcast the message to any
        stream that has ever communicated with the disconnecting ID, so
        core.py's :meth:`mitogen.core.Router._on_del_route` can turn the
        message into a disconnect event.

        :param int handle:
            :data:`mitogen.core.ADD_ROUTE` or :data:`mitogen.core.DEL_ROUTE`
        :param int target_id:
            ID of the connecting or disconnecting context.
        """
        for stream in self.router.get_streams():
            if target_id in stream.protocol.egress_ids and (
                (self.parent is None)
                or (self.parent.context_id != stream.protocol.remote_id)
            ):
                self._send_one(stream, mitogen.core.DEL_ROUTE, target_id, None)

    def notice_stream(self, stream):
        """
        When this parent is responsible for a new directly connected child
        stream, we're also responsible for broadcasting
        :data:`mitogen.core.DEL_ROUTE` upstream when that child disconnects.
        """
        self._routes_by_stream[stream] = set([stream.protocol.remote_id])
        self._propagate_up(
            mitogen.core.ADD_ROUTE, stream.protocol.remote_id, stream.name
        )
        mitogen.core.listen(
            obj=stream,
            name="disconnect",
            func=lambda: self._on_stream_disconnect(stream),
        )

    def get_routes(self, stream):
        """
        Return the set of context IDs reachable on a stream.

        :param mitogen.core.Stream stream:
        :returns: set([int])
        """
        return self._routes_by_stream.get(stream) or set()

    def _on_stream_disconnect(self, stream):
        """
        Respond to disconnection of a local stream by propagating DEL_ROUTE for
        any contexts we know were attached to it.
        """
        # During a stream crash it is possible for disconnect signal to fire
        # twice, in which case ignore the second instance.
        routes = self._routes_by_stream.pop(stream, None)
        if routes is None:
            return

        self._log.debug(
            "stream %s is gone; propagating DEL_ROUTE for %r", stream.name, routes
        )
        for target_id in routes:
            self.router.del_route(target_id)
            self._propagate_up(mitogen.core.DEL_ROUTE, target_id)
            self._propagate_down(mitogen.core.DEL_ROUTE, target_id)

            context = self.router.context_by_id(target_id, create=False)
            if context:
                mitogen.core.fire(context, "disconnect")

    def _on_add_route(self, msg):
        """
        Respond to :data:`mitogen.core.ADD_ROUTE` by validating the source of
        the message, updating the local table, and propagating the message
        upwards.
        """
        if msg.is_dead:
            return

        target_id_s, _, target_name = bytes_partition(msg.data, b(":"))
        target_name = target_name.decode()
        target_id = int(target_id_s)
        self.router.context_by_id(target_id).name = target_name
        stream = self.router.stream_by_id(msg.src_id)
        current = self.router.stream_by_id(target_id)
        if current and current.protocol.remote_id != mitogen.parent_id:
            self._log.error(
                "Cannot add duplicate route to %r via %r, "
                "already have existing route via %r",
                target_id,
                stream,
                current,
            )
            return

        self._log.debug("Adding route to %d via %r", target_id, stream)
        self._routes_by_stream[stream].add(target_id)
        self.router.add_route(target_id, stream)
        self._propagate_up(mitogen.core.ADD_ROUTE, target_id, target_name)

    def _on_del_route(self, msg):
        """
        Respond to :data:`mitogen.core.DEL_ROUTE` by validating the source of
        the message, updating the local table, propagating the message
        upwards, and downwards towards any stream that every had a message
        forwarded from it towards the disconnecting context.
        """
        if msg.is_dead:
            return

        target_id = int(msg.data)
        registered_stream = self.router.stream_by_id(target_id)
        if registered_stream is None:
            return

        stream = self.router.stream_by_id(msg.src_id)
        if registered_stream != stream:
            self._log.error(
                "received DEL_ROUTE for %d from %r, expected %r",
                target_id,
                stream,
                registered_stream,
            )
            return

        context = self.router.context_by_id(target_id, create=False)
        if context:
            self._log.debug("firing local disconnect signal for %r", context)
            mitogen.core.fire(context, "disconnect")

        self._log.debug("deleting route to %d via %r", target_id, stream)
        routes = self._routes_by_stream.get(stream)
        if routes:
            routes.discard(target_id)

        self.router.del_route(target_id)
        if stream.protocol.remote_id != mitogen.parent_id:
            self._propagate_up(mitogen.core.DEL_ROUTE, target_id)
        self._propagate_down(mitogen.core.DEL_ROUTE, target_id)


class Router(mitogen.core.Router):
    context_class = Context
    debug = False
    profiling = False

    id_allocator = None
    responder = None
    log_forwarder = None
    route_monitor = None

    def upgrade(self, importer, parent):
        LOG.debug("upgrading %r with capabilities to start new children", self)
        self.id_allocator = ChildIdAllocator(router=self)
        self.responder = ModuleForwarder(
            router=self,
            parent_context=parent,
            importer=importer,
        )
        self.route_monitor = RouteMonitor(self, parent)
        self.add_handler(
            fn=self._on_detaching,
            handle=mitogen.core.DETACHING,
            persist=True,
        )

    def _on_detaching(self, msg):
        if msg.is_dead:
            return
        stream = self.stream_by_id(msg.src_id)
        if stream.protocol.remote_id != msg.src_id or stream.conn.detached:
            LOG.warning("bad DETACHING received on %r: %r", stream, msg)
            return
        LOG.debug("%r: marking as detached", stream)
        stream.conn.detached = True
        msg.reply(None)

    def get_streams(self):
        """
        Return an atomic snapshot of all streams in existence at time of call.
        This is safe to call from any thread.
        """
        self._write_lock.acquire()
        try:
            return itervalues(self._stream_by_id)
        finally:
            self._write_lock.release()

    def disconnect(self, context):
        """
        Disconnect a context and forget its stream, assuming the context is
        directly connected.
        """
        stream = self.stream_by_id(context)
        if stream is None or stream.protocol.remote_id != context.context_id:
            return

        l = mitogen.core.Latch()
        mitogen.core.listen(stream, "disconnect", l.put)

        def disconnect():
            LOG.debug("Starting disconnect of %r", stream)
            stream.on_disconnect(self.broker)

        self.broker.defer(disconnect)
        l.get()

    def add_route(self, target_id, stream):
        """
        Arrange for messages whose `dst_id` is `target_id` to be forwarded on a
        directly connected :class:`Stream`. Safe to call from any thread.

        This is called automatically by :class:`RouteMonitor` in response to
        :data:`mitogen.core.ADD_ROUTE` messages, but remains public while the
        design has not yet settled, and situations may arise where routing is
        not fully automatic.

        :param int target_id:
            Target context ID to add a route for.
        :param mitogen.core.Stream stream:
            Stream over which messages to the target should be routed.
        """
        LOG.debug("%r: adding route to context %r via %r", self, target_id, stream)
        assert isinstance(target_id, int)
        assert isinstance(stream, mitogen.core.Stream)

        self._write_lock.acquire()
        try:
            self._stream_by_id[target_id] = stream
        finally:
            self._write_lock.release()

    def del_route(self, target_id):
        """
        Delete any route that exists for `target_id`. It is not an error to
        delete a route that does not currently exist. Safe to call from any
        thread.

        This is called automatically by :class:`RouteMonitor` in response to
        :data:`mitogen.core.DEL_ROUTE` messages, but remains public while the
        design has not yet settled, and situations may arise where routing is
        not fully automatic.

        :param int target_id:
            Target context ID to delete route for.
        """
        LOG.debug("%r: deleting route to %r", self, target_id)
        # DEL_ROUTE may be sent by a parent if it knows this context sent
        # messages to a peer that has now disconnected, to let us raise
        # 'disconnect' event on the appropriate Context instance. In that case,
        # we won't a matching _stream_by_id entry for the disappearing route,
        # so don't raise an error for a missing key here.
        self._write_lock.acquire()
        try:
            self._stream_by_id.pop(target_id, None)
        finally:
            self._write_lock.release()

    def get_module_blacklist(self):
        if mitogen.context_id == 0:
            return self.responder.blacklist
        return self.importer.master_blacklist

    def get_module_whitelist(self):
        if mitogen.context_id == 0:
            return self.responder.whitelist
        return self.importer.master_whitelist

    def allocate_id(self):
        return self.id_allocator.allocate()

    connection_timeout_msg = u"Connection timed out."

    def _connect(self, klass, **kwargs):
        context_id = self.allocate_id()
        context = self.context_class(self, context_id)
        context.name = kwargs.get("name")

        kwargs["old_router"] = self
        kwargs["max_message_size"] = self.max_message_size
        conn = klass(klass.options_class(**kwargs), self)
        try:
            conn.connect(context=context)
        except mitogen.core.TimeoutError:
            raise mitogen.core.StreamError(self.connection_timeout_msg)

        return context

    def connect(self, method_name, name=None, **kwargs):
        if name:
            name = mitogen.core.to_text(name)

        klass = get_connection_class(method_name)
        kwargs.setdefault(u"debug", self.debug)
        kwargs.setdefault(u"profiling", self.profiling)
        kwargs.setdefault(u"unidirectional", self.unidirectional)
        kwargs.setdefault(u"name", name)

        via = kwargs.pop(u"via", None)
        if via is not None:
            return self.proxy_connect(via, method_name, **mitogen.core.Kwargs(kwargs))
        return self._connect(klass, **mitogen.core.Kwargs(kwargs))

    def proxy_connect(self, via_context, method_name, name=None, **kwargs):
        resp = via_context.call(
            _proxy_connect,
            name=name,
            method_name=method_name,
            kwargs=mitogen.core.Kwargs(kwargs),
        )
        if resp["msg"] is not None:
            raise mitogen.core.StreamError(resp["msg"])

        name = f"{via_context.name}.{resp['name']}"
        context = self.context_class(self, resp["id"], name=name)
        context.via = via_context
        self._write_lock.acquire()
        try:
            self._context_by_id[context.context_id] = context
        finally:
            self._write_lock.release()
        return context

    def buildah(self, **kwargs):
        return self.connect(u"buildah", **kwargs)

    def doas(self, **kwargs):
        return self.connect(u"doas", **kwargs)

    def docker(self, **kwargs):
        return self.connect(u"docker", **kwargs)

    def kubectl(self, **kwargs):
        return self.connect(u"kubectl", **kwargs)

    def fork(self, **kwargs):
        return self.connect(u"fork", **kwargs)

    def jail(self, **kwargs):
        return self.connect(u"jail", **kwargs)

    def local(self, **kwargs):
        return self.connect(u"local", **kwargs)

    def lxc(self, **kwargs):
        return self.connect(u"lxc", **kwargs)

    def lxd(self, **kwargs):
        return self.connect(u"lxd", **kwargs)

    def setns(self, **kwargs):
        return self.connect(u"setns", **kwargs)

    def su(self, **kwargs):
        return self.connect(u"su", **kwargs)

    def sudo(self, **kwargs):
        return self.connect(u"sudo", **kwargs)

    def ssh(self, **kwargs):
        return self.connect(u"ssh", **kwargs)


class Reaper(object):
    """
    Asynchronous logic for reaping :class:`Process` objects. This is necessary
    to prevent uncontrolled buildup of zombie processes in long-lived parents
    that will eventually reach an OS limit, preventing creation of new threads
    and processes, and to log the exit status of the child in the case of an
    error.

    To avoid modifying process-global state such as with
    :func:`signal.set_wakeup_fd` or installing a :data:`signal.SIGCHLD` handler
    that might interfere with the user's ability to use those facilities,
    Reaper polls for exit with backoff using timers installed on an associated
    :class:`Broker`.

    :param mitogen.core.Broker broker:
        The :class:`Broker` on which to install timers
    :param mitogen.parent.Process proc:
        The process to reap.
    :param bool kill:
        If :data:`True`, send ``SIGTERM`` and ``SIGKILL`` to the process.
    :param bool wait_on_shutdown:
        If :data:`True`, delay :class:`Broker` shutdown if child has not yet
        exited. If :data:`False` simply forget the child.
    """

    #: :class:`Timer` that invokes :meth:`reap` after some polling delay.
    _timer = None

    def __init__(self, broker, proc, kill, wait_on_shutdown):
        self.broker = broker
        self.proc = proc
        self.kill = kill
        self.wait_on_shutdown = wait_on_shutdown
        self._tries = 0

    def _signal_child(self, signum):
        # For processes like sudo we cannot actually send sudo a signal,
        # because it is setuid, so this is best-effort only.
        LOG.debug("%r: sending %s", self.proc, SIGNAL_BY_NUM[signum])
        try:
            os.kill(self.proc.pid, signum)
        except OSError:
            e = sys.exc_info()[1]
            if e.args[0] != errno.EPERM:
                raise

    def _calc_delay(self, count):
        """
        Calculate a poll delay given `count` attempts have already been made.
        These constants have no principle, they just produce rapid but still
        relatively conservative retries.
        """
        delay = 0.05
        for _ in xrange(count):
            delay *= 1.72
        return delay

    def _on_broker_shutdown(self):
        """
        Respond to :class:`Broker` shutdown by cancelling the reap timer if
        :attr:`Router.await_children_at_shutdown` is disabled. Otherwise
        shutdown is delayed for up to :attr:`Broker.shutdown_timeout` for
        subprocesses may have no intention of exiting any time soon.
        """
        if not self.wait_on_shutdown:
            self._timer.cancel()

    def _install_timer(self, delay):
        new = self._timer is None
        self._timer = self.broker.timers.schedule(
            when=mitogen.core.now() + delay,
            func=self.reap,
        )
        if new:
            mitogen.core.listen(self.broker, "shutdown", self._on_broker_shutdown)

    def _remove_timer(self):
        if self._timer and self._timer.active:
            self._timer.cancel()
            mitogen.core.unlisten(self.broker, "shutdown", self._on_broker_shutdown)

    def reap(self):
        """
        Reap the child process during disconnection.
        """
        status = self.proc.poll()
        if status is not None:
            LOG.debug("%r: %s", self.proc, returncode_to_str(status))
            mitogen.core.fire(self.proc, "exit")
            self._remove_timer()
            return

        self._tries += 1
        if self._tries > 20:
            LOG.warning("%r: child will not exit, giving up", self)
            self._remove_timer()
            return

        delay = self._calc_delay(self._tries - 1)
        LOG.debug(
            "%r still running after IO disconnect, recheck in %.03fs", self.proc, delay
        )
        self._install_timer(delay)

        if not self.kill:
            pass
        elif self._tries == 2:
            self._signal_child(signal.SIGTERM)
        elif self._tries == 6:  # roughly 4 seconds
            self._signal_child(signal.SIGKILL)


class Process(object):
    """
    Process objects provide a uniform interface to the :mod:`subprocess` and
    :mod:`mitogen.fork`. This class is extended by :class:`PopenProcess` and
    :class:`mitogen.fork.Process`.

    :param int pid:
        The process ID.
    :param file stdin:
        File object attached to standard input.
    :param file stdout:
        File object attached to standard output.
    :param file stderr:
        File object attached to standard error, or :data:`None`.
    """

    #: Name of the process used in logs. Set to the stream/context name by
    #: :class:`Connection`.
    name = None

    def __init__(self, pid, stdin, stdout, stderr=None):
        #: The process ID.
        self.pid = pid
        #: File object attached to standard input.
        self.stdin = stdin
        #: File object attached to standard output.
        self.stdout = stdout
        #: File object attached to standard error.
        self.stderr = stderr

    def __repr__(self):
        return "%s %s pid %d" % (
            type(self).__name__,
            self.name,
            self.pid,
        )

    def poll(self):
        """
        Fetch the child process exit status, or :data:`None` if it is still
        running. This should be overridden by subclasses.

        :returns:
            Exit status in the style of the :attr:`subprocess.Popen.returncode`
            attribute, i.e. with signals represented by a negative integer.
        """
        raise NotImplementedError()


class PopenProcess(Process):
    """
    :class:`Process` subclass wrapping a :class:`subprocess.Popen` object.

    :param subprocess.Popen proc:
        The subprocess.
    """

    def __init__(self, proc, stdin, stdout, stderr=None):
        super(PopenProcess, self).__init__(proc.pid, stdin, stdout, stderr)
        #: The subprocess.
        self.proc = proc

    def poll(self):
        return self.proc.poll()


class ModuleForwarder(object):
    """
    Respond to :data:`mitogen.core.GET_MODULE` requests in a child by
    forwarding the request to our parent context, or satisfying the request
    from our local Importer cache.
    """

    def __init__(self, router, parent_context, importer):
        self.router = router
        self.parent_context = parent_context
        self.importer = importer
        router.add_handler(
            fn=self._on_forward_module,
            handle=mitogen.core.FORWARD_MODULE,
            persist=True,
            policy=mitogen.core.has_parent_authority,
        )
        router.add_handler(
            fn=self._on_get_module,
            handle=mitogen.core.GET_MODULE,
            persist=True,
            policy=is_immediate_child,
        )

    def __repr__(self):
        return "ModuleForwarder"

    def _on_forward_module(self, msg):
        if msg.is_dead:
            return

        context_id_s, _, fullname = bytes_partition(msg.data, b("\x00"))
        fullname = mitogen.core.to_text(fullname)
        context_id = int(context_id_s)
        stream = self.router.stream_by_id(context_id)
        if stream.protocol.remote_id == mitogen.parent_id:
            LOG.error(
                "%r: dropping FORWARD_MODULE(%d, %r): no route to child",
                self,
                context_id,
                fullname,
            )
            return

        if fullname in stream.protocol.sent_modules:
            return

        LOG.debug(
            "%r._on_forward_module() sending %r to %r via %r",
            self,
            fullname,
            context_id,
            stream.protocol.remote_id,
        )
        self._send_module_and_related(stream, fullname)
        if stream.protocol.remote_id != context_id:
            stream.protocol._send(
                mitogen.core.Message(
                    data=msg.data,
                    handle=mitogen.core.FORWARD_MODULE,
                    dst_id=stream.protocol.remote_id,
                )
            )

    def _on_get_module(self, msg):
        if msg.is_dead:
            return

        fullname = msg.data.decode("utf-8")
        LOG.debug("%r: %s requested by context %d", self, fullname, msg.src_id)
        callback = lambda: self._on_cache_callback(msg, fullname)
        self.importer._request_module(fullname, callback)

    def _on_cache_callback(self, msg, fullname):
        stream = self.router.stream_by_id(msg.src_id)
        LOG.debug("%r: sending %s to %r", self, fullname, stream)
        self._send_module_and_related(stream, fullname)

    def _send_module_and_related(self, stream, fullname):
        tup = self.importer._cache[fullname]
        for related in tup[4]:
            rtup = self.importer._cache.get(related)
            if rtup:
                self._send_one_module(stream, rtup)
            else:
                LOG.debug("%r: %s not in cache (for %s)", self, related, fullname)

        self._send_one_module(stream, tup)

    def _send_one_module(self, stream, tup):
        if tup[0] not in stream.protocol.sent_modules:
            stream.protocol.sent_modules.add(tup[0])
            self.router._async_route(
                mitogen.core.Message.pickled(
                    tup,
                    dst_id=stream.protocol.remote_id,
                    handle=mitogen.core.LOAD_MODULE,
                )
            )
