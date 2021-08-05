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

# future
from __future__ import absolute_import

# stdlib
import atexit
import logging
import multiprocessing
import os
import resource
import signal
import socket
import sys

try:
    # stdlib
    import faulthandler
except ImportError:
    faulthandler = None

try:
    # third party
    import setproctitle
except ImportError:
    setproctitle = None

# third party
import ansible
import ansible.constants as C
import ansible.errors
import ansible_mitogen.affinity
import ansible_mitogen.logging
import ansible_mitogen.services
import mitogen
import mitogen.core
from mitogen.core import b
import mitogen.debug
import mitogen.fork
import mitogen.master
import mitogen.parent
import mitogen.service
import mitogen.unix
import mitogen.utils

LOG = logging.getLogger(__name__)

ANSIBLE_PKG_OVERRIDE = u"__version__ = %r\n" u"__author__ = %r\n"

MAX_MESSAGE_SIZE = 4096 * 1048576

worker_model_msg = (
    "Mitogen connection types may only be instantiated when one of the "
    '"mitogen_*" or "operon_*" strategies are active.'
)

shutting_down_msg = (
    "The task worker cannot connect. Ansible may be shutting down, or "
    "the maximum open files limit may have been exceeded. If this occurs "
    "midway through a run, please retry after increasing the open file "
    "limit (ulimit -n). Original error: %s"
)


#: The worker model as configured by the currently running strategy. This is
#: managed via :func:`get_worker_model` / :func:`set_worker_model` functions by
#: :class:`StrategyMixin`.
_worker_model = None


#: A copy of the sole :class:`ClassicWorkerModel` that ever exists during a
#: classic run, as return by :func:`get_classic_worker_model`.
_classic_worker_model = None


def set_worker_model(model):
    """
    To remove process model-wiring from
    :class:`ansible_mitogen.connection.Connection`, it is necessary to track
    some idea of the configured execution environment outside the connection
    plug-in.

    That is what :func:`set_worker_model` and :func:`get_worker_model` are for.
    """
    global _worker_model
    assert model is None or _worker_model is None
    _worker_model = model


def get_worker_model():
    """
    Return the :class:`WorkerModel` currently configured by the running
    strategy.
    """
    if _worker_model is None:
        raise ansible.errors.AnsibleConnectionFailure(worker_model_msg)
    return _worker_model


def get_classic_worker_model(**kwargs):
    """
    Return the single :class:`ClassicWorkerModel` instance, constructing it if
    necessary.
    """
    global _classic_worker_model
    assert _classic_worker_model is None or (
        not kwargs
    ), "ClassicWorkerModel kwargs supplied but model already constructed"

    if _classic_worker_model is None:
        _classic_worker_model = ClassicWorkerModel(**kwargs)
    return _classic_worker_model


def getenv_int(key, default=0):
    """
    Get an integer-valued environment variable `key`, if it exists and parses
    as an integer, otherwise return `default`.
    """
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def save_pid(name):
    """
    When debugging and profiling, it is very annoying to poke through the
    process list to discover the currently running Ansible and MuxProcess IDs,
    especially when trying to catch an issue during early startup. So here, if
    a magic environment variable set, stash them in hidden files in the CWD::

        alias muxpid="cat .ansible-mux.pid"
        alias anspid="cat .ansible-controller.pid"

        gdb -p $(muxpid)
        perf top -p $(anspid)
    """
    if os.environ.get("MITOGEN_SAVE_PIDS"):
        with open(".ansible-%s.pid" % (name,), "w") as fp:
            fp.write(str(os.getpid()))


def setup_pool(pool):
    """
    Configure a connection multiplexer's :class:`mitogen.service.Pool` with
    services accessed by clients and WorkerProcesses.
    """
    pool.add(mitogen.service.FileService(router=pool.router))
    pool.add(mitogen.service.PushFileService(router=pool.router))
    pool.add(ansible_mitogen.services.ContextService(router=pool.router))
    pool.add(ansible_mitogen.services.ModuleDepService(pool.router))
    LOG.debug("Service pool configured: size=%d", pool.size)


def _setup_simplejson(responder):
    """
    We support serving simplejson for Python 2.4 targets on Ansible 2.3, at
    least so the package's own CI Docker scripts can run without external
    help, however newer versions of simplejson no longer support Python
    2.4. Therefore override any installed/loaded version with a
    2.4-compatible version we ship in the compat/ directory.
    """
    responder.whitelist_prefix("simplejson")

    # issue #536: must be at end of sys.path, in case existing newer
    # version is already loaded.
    compat_path = os.path.join(os.path.dirname(__file__), "compat")
    sys.path.append(compat_path)

    for fullname, is_pkg, suffix in (
        (u"simplejson", True, "__init__.py"),
        (u"simplejson.decoder", False, "decoder.py"),
        (u"simplejson.encoder", False, "encoder.py"),
        (u"simplejson.scanner", False, "scanner.py"),
    ):
        path = os.path.join(compat_path, "simplejson", suffix)
        fp = open(path, "rb")
        try:
            source = fp.read()
        finally:
            fp.close()

        responder.add_source_override(
            fullname=fullname,
            path=path,
            source=source,
            is_pkg=is_pkg,
        )


def _setup_responder(responder):
    """
    Configure :class:`mitogen.master.ModuleResponder` to only permit
    certain packages, and to generate custom responses for certain modules.
    """
    responder.whitelist_prefix("ansible")
    responder.whitelist_prefix("ansible_mitogen")
    _setup_simplejson(responder)

    # Ansible 2.3 is compatible with Python 2.4 targets, however
    # ansible/__init__.py is not. Instead, executor/module_common.py writes
    # out a 2.4-compatible namespace package for unknown reasons. So we
    # copy it here.
    responder.add_source_override(
        fullname="ansible",
        path=ansible.__file__,
        source=(
            ANSIBLE_PKG_OVERRIDE
            % (
                ansible.__version__,
                ansible.__author__,
            )
        ).encode(),
        is_pkg=True,
    )


def increase_open_file_limit():
    """
    #549: in order to reduce the possibility of hitting an open files limit,
    increase :data:`resource.RLIMIT_NOFILE` from its soft limit to its hard
    limit, if they differ.

    It is common that a low soft limit is configured by default, where the hard
    limit is much higher.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY:
        hard_s = "(infinity)"
        # cap in case of O(RLIMIT_NOFILE) algorithm in some subprocess.
        hard = 524288
    else:
        hard_s = str(hard)

    LOG.debug("inherited open file limits: soft=%d hard=%s", soft, hard_s)
    if soft >= hard:
        LOG.debug("max open files already set to hard limit: %d", hard)
        return

    # OS X is limited by kern.maxfilesperproc sysctl, rather than the
    # advertised unlimited hard RLIMIT_NOFILE. Just hard-wire known defaults
    # for that sysctl, to avoid the mess of querying it.
    for value in (hard, 10240):
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (value, hard))
            LOG.debug("raised soft open file limit from %d to %d", soft, value)
            break
        except ValueError as e:
            LOG.debug(
                "could not raise soft open file limit from %d to %d: %s", soft, value, e
            )


def common_setup(enable_affinity=True, _init_logging=True):
    save_pid("controller")
    ansible_mitogen.logging.set_process_name("top")

    if _init_logging:
        ansible_mitogen.logging.setup()

    if enable_affinity:
        ansible_mitogen.affinity.policy.assign_controller()

    mitogen.utils.setup_gil()
    if faulthandler is not None:
        faulthandler.enable()

    MuxProcess.profiling = getenv_int("MITOGEN_PROFILING") > 0
    if MuxProcess.profiling:
        mitogen.core.enable_profiling()

    MuxProcess.cls_original_env = dict(os.environ)
    increase_open_file_limit()


def get_cpu_count(default=None):
    """
    Get the multiplexer CPU count from the MITOGEN_CPU_COUNT environment
    variable, returning `default` if one isn't set, or is out of range.

    :param int default:
        Default CPU, or :data:`None` to use all available CPUs.
    """
    max_cpus = multiprocessing.cpu_count()
    if default is None:
        default = max_cpus

    cpu_count = getenv_int("MITOGEN_CPU_COUNT", default=default)
    if cpu_count < 1 or cpu_count > max_cpus:
        cpu_count = default

    return cpu_count


class Broker(mitogen.master.Broker):
    """
    WorkerProcess maintains at most 2 file descriptors, therefore does not need
    the exuberant syscall expense of EpollPoller, so override it and restore
    the poll() poller.
    """

    poller_class = mitogen.core.Poller


class Binding(object):
    """
    Represent a bound connection for a particular inventory hostname. When
    operating in sharded mode, the actual MuxProcess implementing a connection
    varies according to the target machine. Depending on the particular
    implementation, this class represents a binding to the correct MuxProcess.
    """

    def get_child_service_context(self):
        """
        Return the :class:`mitogen.core.Context` to which children should
        direct requests for services such as FileService, or :data:`None` for
        the local process.

        This can be different from :meth:`get_service_context` where MuxProcess
        and WorkerProcess are combined, and it is discovered a task is
        delegated after being assigned to its initial worker for the original
        un-delegated hostname. In that case, connection management and
        expensive services like file transfer must be implemented by the
        MuxProcess connected to the target, rather than routed to the
        MuxProcess responsible for executing the task.
        """
        raise NotImplementedError()

    def get_service_context(self):
        """
        Return the :class:`mitogen.core.Context` to which this process should
        direct ContextService requests, or :data:`None` for the local process.
        """
        raise NotImplementedError()

    def close(self):
        """
        Finalize any associated resources.
        """
        raise NotImplementedError()


class WorkerModel(object):
    """
    Interface used by StrategyMixin to manage various Mitogen services, by
    default running in one or more connection multiplexer subprocesses spawned
    off the top-level Ansible process.
    """

    def on_strategy_start(self):
        """
        Called prior to strategy start in the top-level process. Responsible
        for preparing any worker/connection multiplexer state.
        """
        raise NotImplementedError()

    def on_strategy_complete(self):
        """
        Called after strategy completion in the top-level process. Must place
        Ansible back in a "compatible" state where any other strategy plug-in
        may execute.
        """
        raise NotImplementedError()

    def get_binding(self, inventory_name):
        """
        Return a :class:`Binding` to access Mitogen services for
        `inventory_name`. Usually called from worker processes, but may also be
        called from top-level process to handle "meta: reset_connection".
        """
        raise NotImplementedError()


class ClassicBinding(Binding):
    """
    Only one connection may be active at a time in a classic worker, so its
    binding just provides forwarders back to :class:`ClassicWorkerModel`.
    """

    def __init__(self, model):
        self.model = model

    def get_service_context(self):
        """
        See Binding.get_service_context().
        """
        return self.model.parent

    def get_child_service_context(self):
        """
        See Binding.get_child_service_context().
        """
        return self.model.parent

    def close(self):
        """
        See Binding.close().
        """
        self.model.on_binding_close()


class ClassicWorkerModel(WorkerModel):
    #: In the top-level process, this references one end of a socketpair(),
    #: whose other end child MuxProcesses block reading from to determine when
    #: the master process dies. When the top-level exits abnormally, or
    #: normally but where :func:`_on_process_exit` has been called, this socket
    #: will be closed, causing all the children to wake.
    parent_sock = None

    #: In the mux process, this is the other end of :attr:`cls_parent_sock`.
    #: The main thread blocks on a read from it until :attr:`cls_parent_sock`
    #: is closed.
    child_sock = None

    #: mitogen.master.Router for this worker.
    router = None

    #: mitogen.master.Broker for this worker.
    broker = None

    #: Name of multiplexer process socket we are currently connected to.
    listener_path = None

    #: mitogen.parent.Context representing the parent Context, which is the
    #: connection multiplexer process when running in classic mode, or the
    #: top-level process when running a new-style mode.
    parent = None

    def __init__(self, _init_logging=True):
        """
        Arrange for classic model multiplexers to be started. The parent choses
        UNIX socket paths each child will use prior to fork, creates a
        socketpair used essentially as a semaphore, then blocks waiting for the
        child to indicate the UNIX socket is ready for use.

        :param bool _init_logging:
            For testing, if :data:`False`, don't initialize logging.
        """
        # #573: The process ID that installed the :mod:`atexit` handler. If
        # some unknown Ansible plug-in forks the Ansible top-level process and
        # later performs a graceful Python exit, it may try to wait for child
        # PIDs it never owned, causing a crash. We want to avoid that.
        self._pid = os.getpid()

        common_setup(_init_logging=_init_logging)

        self.parent_sock, self.child_sock = socket.socketpair()
        mitogen.core.set_cloexec(self.parent_sock.fileno())
        mitogen.core.set_cloexec(self.child_sock.fileno())

        self._muxes = [
            MuxProcess(self, index) for index in range(get_cpu_count(default=1))
        ]
        for mux in self._muxes:
            mux.start()

        atexit.register(self._on_process_exit)
        self.child_sock.close()
        self.child_sock = None

    def _listener_for_name(self, name):
        """
        Given an inventory hostname, return the UNIX listener that should
        communicate with it. This is a simple hash of the inventory name.
        """
        mux = self._muxes[abs(hash(name)) % len(self._muxes)]
        LOG.debug(
            'will use multiplexer %d (%s) to connect to "%s"', mux.index, mux.path, name
        )
        return mux.path

    def _reconnect(self, path):
        if self.router is not None:
            # Router can just be overwritten, but the previous parent
            # connection must explicitly be removed from the broker first.
            self.router.disconnect(self.parent)
            self.parent = None
            self.router = None

        try:
            self.router, self.parent = mitogen.unix.connect(
                path=path,
                broker=self.broker,
            )
        except mitogen.unix.ConnectError as e:
            # This is not AnsibleConnectionFailure since we want to break
            # with_items loops.
            raise ansible.errors.AnsibleError(shutting_down_msg % (e,))

        self.router.max_message_size = MAX_MESSAGE_SIZE
        self.listener_path = path

    def _on_process_exit(self):
        """
        This is an :mod:`atexit` handler installed in the top-level process.

        Shut the write end of `sock`, causing the receive side of the socket in
        every :class:`MuxProcess` to return 0-byte reads, and causing their
        main threads to wake and initiate shutdown. After shutting the socket
        down, wait on each child to finish exiting.

        This is done using :mod:`atexit` since Ansible lacks any better hook to
        run code during exit, and unless some synchronization exists with
        MuxProcess, debug logs may appear on the user's terminal *after* the
        prompt has been printed.
        """
        if self._pid != os.getpid():
            return

        try:
            self.parent_sock.shutdown(socket.SHUT_WR)
        except socket.error:
            # Already closed. This is possible when tests are running.
            LOG.debug("_on_process_exit: ignoring duplicate call")
            return

        mitogen.core.io_op(self.parent_sock.recv, 1)
        self.parent_sock.close()

        for mux in self._muxes:
            _, status = os.waitpid(mux.pid, 0)
            status = mitogen.fork._convert_exit_status(status)
            LOG.debug(
                "multiplexer %d PID %d %s",
                mux.index,
                mux.pid,
                mitogen.parent.returncode_to_str(status),
            )

    def _test_reset(self):
        """
        Used to clean up in unit tests.
        """
        self.on_binding_close()
        self._on_process_exit()
        set_worker_model(None)

        global _classic_worker_model
        _classic_worker_model = None

    def on_strategy_start(self):
        """
        See WorkerModel.on_strategy_start().
        """

    def on_strategy_complete(self):
        """
        See WorkerModel.on_strategy_complete().
        """

    def get_binding(self, inventory_name):
        """
        See WorkerModel.get_binding().
        """
        if self.broker is None:
            self.broker = Broker()

        path = self._listener_for_name(inventory_name)
        if path != self.listener_path:
            self._reconnect(path)

        return ClassicBinding(self)

    def on_binding_close(self):
        if not self.broker:
            return

        self.broker.shutdown()
        self.broker.join()
        self.router = None
        self.broker = None
        self.parent = None
        self.listener_path = None

        # #420: Ansible executes "meta" actions in the top-level process,
        # meaning "reset_connection" will cause :class:`mitogen.core.Latch` FDs
        # to be cached and erroneously shared by children on subsequent
        # WorkerProcess forks. To handle that, call on_fork() to ensure any
        # shared state is discarded.
        # #490: only attempt to clean up when it's known that some resources
        # exist to cleanup, otherwise later __del__ double-call to close() due
        # to GC at random moment may obliterate an unrelated Connection's
        # related resources.
        mitogen.fork.on_fork()


class MuxProcess(object):
    """
    Implement a subprocess forked from the Ansible top-level, as a safe place
    to contain the Mitogen IO multiplexer thread, keeping its use of the
    logging package (and the logging package's heavy use of locks) far away
    from os.fork(), which is used continuously by the multiprocessing package
    in the top-level process.

    The problem with running the multiplexer in that process is that should the
    multiplexer thread be in the process of emitting a log entry (and holding
    its lock) at the point of fork, in the child, the first attempt to log any
    log entry using the same handler will deadlock the child, as in the memory
    image the child received, the lock will always be marked held.

    See https://bugs.python.org/issue6721 for a thorough description of the
    class of problems this worker is intended to avoid.
    """

    #: A copy of :data:`os.environ` at the time the multiplexer process was
    #: started. It's used by mitogen_local.py to find changes made to the
    #: top-level environment (e.g. vars plugins -- issue #297) that must be
    #: applied to locally executed commands and modules.
    cls_original_env = None

    def __init__(self, model, index):
        #: :class:`ClassicWorkerModel` instance we were created by.
        self.model = model
        #: MuxProcess CPU index.
        self.index = index
        #: Individual path of this process.
        self.path = mitogen.unix.make_socket_path()

    def start(self):
        self.pid = os.fork()
        if self.pid:
            # Wait for child to boot before continuing.
            mitogen.core.io_op(self.model.parent_sock.recv, 1)
            return

        ansible_mitogen.logging.set_process_name("mux:" + str(self.index))
        if setproctitle:
            setproctitle.setproctitle(
                "mitogen mux:%s (%s)"
                % (
                    self.index,
                    os.path.basename(self.path),
                )
            )

        self.model.parent_sock.close()
        self.model.parent_sock = None
        try:
            try:
                self.worker_main()
            except Exception:
                LOG.exception("worker_main() crashed")
        finally:
            sys.exit()

    def worker_main(self):
        """
        The main function of the mux process: setup the Mitogen broker thread
        and ansible_mitogen services, then sleep waiting for the socket
        connected to the parent to be closed (indicating the parent has died).
        """
        save_pid("mux")

        # #623: MuxProcess ignores SIGINT because it wants to live until every
        # Ansible worker process has been cleaned up by
        # TaskQueueManager.cleanup(), otherwise harmles yet scary warnings
        # about being unable connect to MuxProess could be printed.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        ansible_mitogen.logging.set_process_name("mux")
        ansible_mitogen.affinity.policy.assign_muxprocess(self.index)

        self._setup_master()
        self._setup_services()

        try:
            # Let the parent know our listening socket is ready.
            mitogen.core.io_op(self.model.child_sock.send, b("1"))
            # Block until the socket is closed, which happens on parent exit.
            mitogen.core.io_op(self.model.child_sock.recv, 1)
        finally:
            self.broker.shutdown()
            self.broker.join()

            # Test frameworks living somewhere higher on the stack of the
            # original parent process may try to catch sys.exit(), so do a C
            # level exit instead.
            os._exit(0)

    def _enable_router_debug(self):
        if "MITOGEN_ROUTER_DEBUG" in os.environ:
            self.router.enable_debug()

    def _enable_stack_dumps(self):
        secs = getenv_int("MITOGEN_DUMP_THREAD_STACKS", default=0)
        if secs:
            mitogen.debug.dump_to_logger(secs=secs)

    def _setup_master(self):
        """
        Construct a Router, Broker, and mitogen.unix listener
        """
        self.broker = mitogen.master.Broker(install_watcher=False)
        self.router = mitogen.master.Router(
            broker=self.broker,
            max_message_size=MAX_MESSAGE_SIZE,
        )
        _setup_responder(self.router.responder)
        mitogen.core.listen(self.broker, "shutdown", self._on_broker_shutdown)
        mitogen.core.listen(self.broker, "exit", self._on_broker_exit)
        self.listener = mitogen.unix.Listener.build_stream(
            router=self.router,
            path=self.path,
            backlog=C.DEFAULT_FORKS,
        )
        self._enable_router_debug()
        self._enable_stack_dumps()

    def _setup_services(self):
        """
        Construct a ContextService and a thread to service requests for it
        arriving from worker processes.
        """
        self.pool = mitogen.service.Pool(
            router=self.router,
            size=getenv_int("MITOGEN_POOL_SIZE", default=32),
        )
        setup_pool(self.pool)

    def _on_broker_shutdown(self):
        """
        Respond to broker shutdown by shutting down the pool. Do not join on it
        yet, since that would block the broker thread which then cannot clean
        up pending handlers and connections, which is required for the threads
        to exit gracefully.
        """
        self.pool.stop(join=False)

    def _on_broker_exit(self):
        """
        Respond to the broker thread about to exit by finally joining on the
        pool. This is safe since pools only block in connection attempts, and
        connection attempts fail with CancelledError when broker shutdown
        begins.
        """
        self.pool.join()
