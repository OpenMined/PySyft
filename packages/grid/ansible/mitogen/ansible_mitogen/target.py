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
Helper functions intended to be executed on the target. These are entrypoints
for file transfer, module execution and sundry bits like changing file modes.
"""

# stdlib
import errno
import grp
import operator
import os
import pwd
import re
import signal
import stat
import subprocess
import sys
import tempfile
import traceback
import types

# Absolute imports for <2.5.
logging = __import__("logging")

# third party
import mitogen.core
from mitogen.core import b
import mitogen.fork
import mitogen.parent
import mitogen.service

try:
    # stdlib
    import json
except ImportError:
    # third party
    import simplejson as json

try:
    reduce
except NameError:
    # Python 3.x.
    # stdlib
    from functools import reduce

try:
    BaseException
except NameError:
    # Python 2.4
    BaseException = Exception


# Ansible since PR #41749 inserts "import __main__" into
# ansible.module_utils.basic. Mitogen's importer will refuse such an import, so
# we must setup a fake "__main__" before that module is ever imported. The
# str() is to cast Unicode to bytes on Python 2.6.
if not sys.modules.get(str("__main__")):
    sys.modules[str("__main__")] = types.ModuleType(str("__main__"))

# third party
import ansible.module_utils.json_utils
import ansible_mitogen.runner

LOG = logging.getLogger(__name__)

MAKE_TEMP_FAILED_MSG = (
    "Unable to find a useable temporary directory. This likely means no\n"
    "system-supplied TMP directory can be written to, or all directories\n"
    "were mounted on 'noexec' filesystems.\n"
    "\n"
    "The following paths were tried:\n"
    "    %(paths)s\n"
    "\n"
    "Please check '-vvv' output for a log of individual path errors."
)

# Python 2.4/2.5 cannot support fork+threads whatsoever, it doesn't even fix up
# interpreter state. So 2.4/2.5 interpreters start .local() contexts for
# isolation instead. Since we don't have any crazy memory sharing problems to
# avoid, there is no virginal fork parent either. The child is started directly
# from the login/become process. In future this will be default everywhere,
# fork is brainwrong from the stone age.
FORK_SUPPORTED = sys.version_info >= (2, 6)

#: Initialized to an econtext.parent.Context pointing at a pristine fork of
#: the target Python interpreter before it executes any code or imports.
_fork_parent = None

#: Set by :func:`init_child` to the name of a writeable and executable
#: temporary directory accessible by the active user account.
good_temp_dir = None


def subprocess__Popen__close_fds(self, but):
    """
    issue #362, #435: subprocess.Popen(close_fds=True) aka.
    AnsibleModule.run_command() loops the entire FD space on Python<3.2.
    CentOS>5 ships with 1,048,576 FDs by default, resulting in huge (>500ms)
    latency starting children. Therefore replace Popen._close_fds on Linux with
    a version that is O(fds) rather than O(_SC_OPEN_MAX).
    """
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        # May fail if acting on a container that does not have /proc mounted.
        self._original_close_fds(but)
        return

    for name in names:
        if not name.isdigit():
            continue

        fd = int(name, 10)
        if fd > 2 and fd != but:
            try:
                os.close(fd)
            except OSError:
                pass


if (
    sys.platform.startswith("linux")
    and sys.version < "3.0"
    and hasattr(subprocess.Popen, "_close_fds")
    and not mitogen.is_master
):
    subprocess.Popen._original_close_fds = subprocess.Popen._close_fds
    subprocess.Popen._close_fds = subprocess__Popen__close_fds


def get_small_file(context, path):
    """
    Basic in-memory caching module fetcher. This generates one roundtrip for
    every previously unseen file, so it is only a temporary solution.

    :param context:
        Context we should direct FileService requests to. For now (and probably
        forever) this is just the top-level Mitogen connection manager process.
    :param path:
        Path to fetch from FileService, must previously have been registered by
        a privileged context using the `register` command.
    :returns:
        Bytestring file data.
    """
    pool = mitogen.service.get_or_create_pool(router=context.router)
    service = pool.get_service("mitogen.service.PushFileService")
    return service.get(path)


def transfer_file(context, in_path, out_path, sync=False, set_owner=False):
    """
    Streamily download a file from the connection multiplexer process in the
    controller.

    :param mitogen.core.Context context:
        Reference to the context hosting the FileService that will transmit the
        file.
    :param bytes in_path:
        FileService registered name of the input file.
    :param bytes out_path:
        Name of the output path on the local disk.
    :param bool sync:
        If :data:`True`, ensure the file content and metadat are fully on disk
        before renaming the temporary file over the existing file. This should
        ensure in the case of system crash, either the entire old or new file
        are visible post-reboot.
    :param bool set_owner:
        If :data:`True`, look up the metadata username and group on the local
        system and file the file owner using :func:`os.fchmod`.
    """
    out_path = os.path.abspath(out_path)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=".ansible_mitogen_transfer-",
        dir=os.path.dirname(out_path),
    )
    fp = os.fdopen(fd, "wb", mitogen.core.CHUNK_SIZE)
    LOG.debug("transfer_file(%r) temporary file: %s", out_path, tmp_path)

    try:
        try:
            ok, metadata = mitogen.service.FileService.get(
                context=context,
                path=in_path,
                out_fp=fp,
            )
            if not ok:
                raise IOError(f"transfer of {in_path!r} was interrupted.")

            set_file_mode(tmp_path, metadata["mode"], fd=fp.fileno())
            if set_owner:
                set_file_owner(
                    tmp_path, metadata["owner"], metadata["group"], fd=fp.fileno()
                )
        finally:
            fp.close()

        if sync:
            os.fsync(fp.fileno())
        os.rename(tmp_path, out_path)
    except BaseException:
        os.unlink(tmp_path)
        raise

    os.utime(out_path, (metadata["atime"], metadata["mtime"]))


def prune_tree(path):
    """
    Like shutil.rmtree(), but log errors rather than discard them, and do not
    waste multiple os.stat() calls discovering whether the object can be
    deleted, just try deleting it instead.
    """
    try:
        os.unlink(path)
        return
    except OSError:
        e = sys.exc_info()[1]
        if not (os.path.isdir(path) and e.args[0] in (errno.EPERM, errno.EISDIR)):
            LOG.error("prune_tree(%r): %s", path, e)
            return

    try:
        # Ensure write access for readonly directories. Ignore error in case
        # path is on a weird filesystem (e.g. vfat).
        os.chmod(path, int("0700", 8))
    except OSError:
        e = sys.exc_info()[1]
        LOG.warning("prune_tree(%r): %s", path, e)

    try:
        for name in os.listdir(path):
            if name not in (".", ".."):
                prune_tree(os.path.join(path, name))
        os.rmdir(path)
    except OSError:
        e = sys.exc_info()[1]
        LOG.error("prune_tree(%r): %s", path, e)


def is_good_temp_dir(path):
    """
    Return :data:`True` if `path` can be used as a temporary directory, logging
    any failures that may cause it to be unsuitable. If the directory doesn't
    exist, we attempt to create it using :func:`os.makedirs`.
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path, mode=int("0700", 8))
        except OSError:
            e = sys.exc_info()[1]
            LOG.debug(
                "temp dir %r unusable: did not exist and attempting "
                "to create it failed: %s",
                path,
                e,
            )
            return False

    try:
        tmp = tempfile.NamedTemporaryFile(
            prefix="ansible_mitogen_is_good_temp_dir",
            dir=path,
        )
    except (OSError, IOError):
        e = sys.exc_info()[1]
        LOG.debug("temp dir %r unusable: %s", path, e)
        return False

    try:
        try:
            os.chmod(tmp.name, int("0700", 8))
        except OSError:
            e = sys.exc_info()[1]
            LOG.debug("temp dir %r unusable: chmod failed: %s", path, e)
            return False

        try:
            # access(.., X_OK) is sufficient to detect noexec.
            if not os.access(tmp.name, os.X_OK):
                raise OSError("filesystem appears to be mounted noexec")
        except OSError:
            e = sys.exc_info()[1]
            LOG.debug("temp dir %r unusable: %s", path, e)
            return False
    finally:
        tmp.close()

    return True


def find_good_temp_dir(candidate_temp_dirs):
    """
    Given a list of candidate temp directories extracted from ``ansible.cfg``,
    combine it with the Python-builtin list of candidate directories used by
    :mod:`tempfile`, then iteratively try each until one is found that is both
    writeable and executable.

    :param list candidate_temp_dirs:
        List of candidate $variable-expanded and tilde-expanded directory paths
        that may be usable as a temporary directory.
    """
    paths = [os.path.expandvars(os.path.expanduser(p)) for p in candidate_temp_dirs]
    paths.extend(tempfile._candidate_tempdir_list())

    for path in paths:
        if is_good_temp_dir(path):
            LOG.debug("Selected temp directory: %r (from %r)", path, paths)
            return path

    raise IOError(
        MAKE_TEMP_FAILED_MSG
        % {
            "paths": "\n    ".join(paths),
        }
    )


@mitogen.core.takes_econtext
def init_child(econtext, log_level, candidate_temp_dirs):
    """
    Called by ContextService immediately after connection; arranges for the
    (presently) spotless Python interpreter to be forked, where the newly
    forked interpreter becomes the parent of any newly forked future
    interpreters.

    This is necessary to prevent modules that are executed in-process from
    polluting the global interpreter state in a way that effects explicitly
    isolated modules.

    :param int log_level:
        Logging package level active in the master.
    :param list[str] candidate_temp_dirs:
        List of $variable-expanded and tilde-expanded directory names to add to
        candidate list of temporary directories.

    :returns:
        Dict like::

            {
                'fork_context': mitogen.core.Context or None,
                'good_temp_dir': ...
                'home_dir': str
            }

        Where `fork_context` refers to the newly forked 'fork parent' context
        the controller will use to start forked jobs, and `home_dir` is the
        home directory for the active user account.
    """
    # Copying the master's log level causes log messages to be filtered before
    # they reach LogForwarder, thus reducing an influx of tiny messges waking
    # the connection multiplexer process in the master.
    LOG.setLevel(log_level)
    logging.getLogger("ansible_mitogen").setLevel(log_level)

    # issue #536: if the json module is available, remove simplejson from the
    # importer whitelist to avoid confusing certain Ansible modules.
    if json.__name__ == "json":
        econtext.importer.whitelist.remove("simplejson")

    global _fork_parent
    if FORK_SUPPORTED:
        mitogen.parent.upgrade_router(econtext)
        _fork_parent = econtext.router.fork()

    global good_temp_dir
    good_temp_dir = find_good_temp_dir(candidate_temp_dirs)

    return {
        "fork_context": _fork_parent,
        "home_dir": mitogen.core.to_text(os.path.expanduser("~")),
        "good_temp_dir": good_temp_dir,
    }


@mitogen.core.takes_econtext
def spawn_isolated_child(econtext):
    """
    For helper functions executed in the fork parent context, arrange for
    the context's router to be upgraded as necessary and for a new child to be
    prepared.

    The actual fork occurs from the 'virginal fork parent', which does not have
    any Ansible modules loaded prior to fork, to avoid conflicts resulting from
    custom module_utils paths.
    """
    mitogen.parent.upgrade_router(econtext)
    if FORK_SUPPORTED:
        context = econtext.router.fork()
    else:
        context = econtext.router.local()
    LOG.debug("create_fork_child() -> %r", context)
    return context


def run_module(kwargs):
    """
    Set up the process environment in preparation for running an Ansible
    module. This monkey-patches the Ansible libraries in various places to
    prevent it from trying to kill the process on completion, and to prevent it
    from reading sys.stdin.
    """
    runner_name = kwargs.pop("runner_name")
    klass = getattr(ansible_mitogen.runner, runner_name)
    impl = klass(**mitogen.core.Kwargs(kwargs))
    return impl.run()


def _get_async_dir():
    return os.path.expanduser(os.environ.get("ANSIBLE_ASYNC_DIR", "~/.ansible_async"))


class AsyncRunner(object):
    def __init__(self, job_id, timeout_secs, started_sender, econtext, kwargs):
        self.job_id = job_id
        self.timeout_secs = timeout_secs
        self.started_sender = started_sender
        self.econtext = econtext
        self.kwargs = kwargs
        self._timed_out = False
        self._init_path()

    def _init_path(self):
        async_dir = _get_async_dir()
        if not os.path.exists(async_dir):
            os.makedirs(async_dir)
        self.path = os.path.join(async_dir, self.job_id)

    def _update(self, dct):
        """
        Update an async job status file.
        """
        LOG.info("%r._update(%r, %r)", self, self.job_id, dct)
        dct.setdefault("ansible_job_id", self.job_id)
        dct.setdefault("data", "")

        fp = open(self.path + ".tmp", "w")
        try:
            fp.write(json.dumps(dct))
        finally:
            fp.close()
        os.rename(self.path + ".tmp", self.path)

    def _on_sigalrm(self, signum, frame):
        """
        Respond to SIGALRM (job timeout) by updating the job file and killing
        the process.
        """
        msg = "Job reached maximum time limit of %d seconds." % (self.timeout_secs,)
        self._update(
            {
                "failed": 1,
                "finished": 1,
                "msg": msg,
            }
        )
        self._timed_out = True
        self.econtext.broker.shutdown()

    def _install_alarm(self):
        signal.signal(signal.SIGALRM, self._on_sigalrm)
        signal.alarm(self.timeout_secs)

    def _run_module(self):
        kwargs = dict(
            self.kwargs,
            **{
                "detach": True,
                "econtext": self.econtext,
                "emulate_tty": False,
            },
        )
        return run_module(kwargs)

    def _parse_result(self, dct):
        filtered, warnings = ansible.module_utils.json_utils._filter_non_json_lines(
            dct["stdout"]
        )
        result = json.loads(filtered)
        result.setdefault("warnings", []).extend(warnings)
        result["stderr"] = dct["stderr"] or result.get("stderr", "")
        self._update(result)

    def _run(self):
        """
        1. Immediately updates the status file to mark the job as started.
        2. Installs a timer/signal handler to implement the time limit.
        3. Runs as with run_module(), writing the result to the status file.

        :param dict kwargs:
            Runner keyword arguments.
        :param str job_id:
            String job ID.
        :param int timeout_secs:
            If >0, limit the task's maximum run time.
        """
        self._update({"started": 1, "finished": 0, "pid": os.getpid()})
        self.started_sender.send(True)

        if self.timeout_secs > 0:
            self._install_alarm()

        dct = self._run_module()
        if not self._timed_out:
            # After SIGALRM fires, there is a window between broker responding
            # to shutdown() by killing the process, and work continuing on the
            # main thread. If main thread was asleep in at least
            # basic.py/select.select(), an EINTR will be raised. We want to
            # discard that exception.
            try:
                self._parse_result(dct)
            except Exception:
                self._update(
                    {
                        "failed": 1,
                        "msg": traceback.format_exc(),
                        "data": dct["stdout"],  # temporary notice only
                        "stderr": dct["stderr"],
                    }
                )

    def run(self):
        try:
            try:
                self._run()
            except Exception:
                self._update(
                    {
                        "failed": 1,
                        "msg": traceback.format_exc(),
                    }
                )
        finally:
            self.econtext.broker.shutdown()


@mitogen.core.takes_econtext
def run_module_async(kwargs, job_id, timeout_secs, started_sender, econtext):
    """
    Execute a module with its run status and result written to a file,
    terminating on the process on completion. This function must run in a child
    forked using :func:`create_fork_child`.

    @param mitogen.core.Sender started_sender:
        A sender that will receive :data:`True` once the job has reached a
        point where its initial job file has been written. This is required to
        avoid a race where an overly eager controller can check for a task
        before it has reached that point in execution, which is possible at
        least on Python 2.4, where forking is not available for async tasks.
    """
    arunner = AsyncRunner(job_id, timeout_secs, started_sender, econtext, kwargs)
    arunner.run()


def get_user_shell():
    """
    For commands executed directly via an SSH command-line, SSH looks up the
    user's shell via getpwuid() and only defaults to /bin/sh if that field is
    missing or empty.
    """
    try:
        pw_shell = pwd.getpwuid(os.geteuid()).pw_shell
    except KeyError:
        pw_shell = None

    return pw_shell or "/bin/sh"


def exec_args(args, in_data="", chdir=None, shell=None, emulate_tty=False):
    """
    Run a command in a subprocess, emulating the argument handling behaviour of
    SSH.

    :param list[str]:
        Argument vector.
    :param bytes in_data:
        Optional standard input for the command.
    :param bool emulate_tty:
        If :data:`True`, arrange for stdout and stderr to be merged into the
        stdout pipe and for LF to be translated into CRLF, emulating the
        behaviour of a TTY.
    :return:
        (return code, stdout bytes, stderr bytes)
    """
    LOG.debug("exec_args(%r, ..., chdir=%r)", args, chdir)
    assert isinstance(args, list)

    if emulate_tty:
        stderr = subprocess.STDOUT
    else:
        stderr = subprocess.PIPE

    proc = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stderr=stderr,
        stdin=subprocess.PIPE,
        cwd=chdir,
    )
    stdout, stderr = proc.communicate(in_data)

    if emulate_tty:
        stdout = stdout.replace(b("\n"), b("\r\n"))
    return proc.returncode, stdout, stderr or b("")


def exec_command(cmd, in_data="", chdir=None, shell=None, emulate_tty=False):
    """
    Run a command in a subprocess, emulating the argument handling behaviour of
    SSH.

    :param bytes cmd:
        String command line, passed to user's shell.
    :param bytes in_data:
        Optional standard input for the command.
    :return:
        (return code, stdout bytes, stderr bytes)
    """
    assert isinstance(cmd, mitogen.core.UnicodeType)
    return exec_args(
        args=[get_user_shell(), "-c", cmd],
        in_data=in_data,
        chdir=chdir,
        shell=shell,
        emulate_tty=emulate_tty,
    )


def read_path(path):
    """
    Fetch the contents of a filesystem `path` as bytes.
    """
    return open(path, "rb").read()


def set_file_owner(path, owner, group=None, fd=None):
    if owner:
        uid = pwd.getpwnam(owner).pw_uid
    else:
        uid = os.geteuid()

    if group:
        gid = grp.getgrnam(group).gr_gid
    else:
        gid = os.getegid()

    if fd is not None and hasattr(os, "fchown"):
        os.fchown(fd, (uid, gid))
    else:
        # Python<2.6
        os.chown(path, (uid, gid))


def write_path(path, s, owner=None, group=None, mode=None, utimes=None, sync=False):
    """
    Writes bytes `s` to a filesystem `path`.
    """
    path = os.path.abspath(path)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=".ansible_mitogen_transfer-", dir=os.path.dirname(path)
    )
    fp = os.fdopen(fd, "wb", mitogen.core.CHUNK_SIZE)
    LOG.debug("write_path(path=%r) temporary file: %s", path, tmp_path)

    try:
        try:
            if mode:
                set_file_mode(tmp_path, mode, fd=fp.fileno())
            if owner or group:
                set_file_owner(tmp_path, owner, group, fd=fp.fileno())
            fp.write(s)
        finally:
            fp.close()

        if sync:
            os.fsync(fp.fileno())
        os.rename(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise

    if utimes:
        os.utime(path, utimes)


CHMOD_CLAUSE_PAT = re.compile(r"([uoga]*)([+\-=])([ugo]|[rwx]*)")
CHMOD_MASKS = {
    "u": stat.S_IRWXU,
    "g": stat.S_IRWXG,
    "o": stat.S_IRWXO,
    "a": (stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO),
}
CHMOD_BITS = {
    "u": {"r": stat.S_IRUSR, "w": stat.S_IWUSR, "x": stat.S_IXUSR},
    "g": {"r": stat.S_IRGRP, "w": stat.S_IWGRP, "x": stat.S_IXGRP},
    "o": {"r": stat.S_IROTH, "w": stat.S_IWOTH, "x": stat.S_IXOTH},
    "a": {
        "r": (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH),
        "w": (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH),
        "x": (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH),
    },
}


def apply_mode_spec(spec, mode):
    """
    Given a symbolic file mode change specification in the style of chmod(1)
    `spec`, apply changes in the specification to the numeric file mode `mode`.
    """
    for clause in mitogen.core.to_text(spec).split(","):
        match = CHMOD_CLAUSE_PAT.match(clause)
        who, op, perms = match.groups()
        for ch in who or "a":
            mask = CHMOD_MASKS[ch]
            bits = CHMOD_BITS[ch]
            cur_perm_bits = mode & mask
            new_perm_bits = reduce(operator.or_, (bits[p] for p in perms), 0)
            mode &= ~mask
            if op == "=":
                mode |= new_perm_bits
            elif op == "+":
                mode |= new_perm_bits | cur_perm_bits
            else:
                mode |= cur_perm_bits & ~new_perm_bits
    return mode


def set_file_mode(path, spec, fd=None):
    """
    Update the permissions of a file using the same syntax as chmod(1).
    """
    if isinstance(spec, int):
        new_mode = spec
    elif not mitogen.core.PY3 and isinstance(spec, long):
        new_mode = spec
    elif spec.isdigit():
        new_mode = int(spec, 8)
    else:
        mode = os.stat(path).st_mode
        new_mode = apply_mode_spec(spec, mode)

    if fd is not None and hasattr(os, "fchmod"):
        os.fchmod(fd, new_mode)
    else:
        os.chmod(path, new_mode)


def file_exists(path):
    """
    Return :data:`True` if `path` exists. This is a wrapper function over
    :func:`os.path.exists`, since its implementation module varies across
    Python versions.
    """
    return os.path.exists(path)
