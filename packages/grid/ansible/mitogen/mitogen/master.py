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
This module implements functionality required by master processes, such as
starting new contexts via SSH. Its size is also restricted, since it must
be sent to any context that will be used to establish additional child
contexts.
"""

# stdlib
import dis
import errno
import imp
import inspect
import itertools
import logging
import os
import pkgutil
import re
import string
import sys
import threading
import types
import zlib

try:
    # stdlib
    import sysconfig
except ImportError:
    sysconfig = None

if not hasattr(pkgutil, "find_loader"):
    # find_loader() was new in >=2.5, but the modern pkgutil.py syntax has
    # been kept intentionally 2.3 compatible so we can reuse it.
    from mitogen.compat import pkgutil

# third party
import mitogen
import mitogen.core
from mitogen.core import IOLOG
from mitogen.core import LOG
from mitogen.core import b
from mitogen.core import str_partition
from mitogen.core import str_rpartition
from mitogen.core import to_text
import mitogen.minify
import mitogen.parent

imap = getattr(itertools, "imap", map)
izip = getattr(itertools, "izip", zip)

try:
    any
except NameError:
    # third party
    from mitogen.core import any

try:
    next
except NameError:
    # third party
    from mitogen.core import next


RLOG = logging.getLogger("mitogen.ctx")


# there are some cases where modules are loaded in memory only, such as
# ansible collections, and the module "filename" doesn't actually exist
SPECIAL_FILE_PATHS = {"__synthetic__", "<ansible_synthetic_collection_package>"}


def _stdlib_paths():
    """
    Return a set of paths from which Python imports the standard library.
    """
    attr_candidates = [
        "prefix",
        "real_prefix",  # virtualenv: only set inside a virtual environment.
        "base_prefix",  # venv: always set, equal to prefix if outside.
    ]
    prefixes = (getattr(sys, a, None) for a in attr_candidates)
    version = "python%s.%s" % sys.version_info[0:2]
    s = set(
        os.path.abspath(os.path.join(p, "lib", version))
        for p in prefixes
        if p is not None
    )

    # When running 'unit2 tests/module_finder_test.py' in a Py2 venv on Ubuntu
    # 18.10, above is insufficient to catch the real directory.
    if sysconfig is not None:
        s.add(sysconfig.get_config_var("DESTLIB"))
    return s


def is_stdlib_name(modname):
    """
    Return :data:`True` if `modname` appears to come from the standard library.
    """
    if imp.is_builtin(modname) != 0:
        return True

    module = sys.modules.get(modname)
    if module is None:
        return False

    # six installs crap with no __file__
    modpath = os.path.abspath(getattr(module, "__file__", ""))
    return is_stdlib_path(modpath)


_STDLIB_PATHS = _stdlib_paths()


def is_stdlib_path(path):
    return any(
        os.path.commonprefix((libpath, path)) == libpath
        and "site-packages" not in path
        and "dist-packages" not in path
        for libpath in _STDLIB_PATHS
    )


def get_child_modules(path, fullname):
    """
    Return the suffixes of submodules directly neated beneath of the package
    directory at `path`.

    :param str path:
        Path to the module's source code on disk, or some PEP-302-recognized
        equivalent. Usually this is the module's ``__file__`` attribute, but
        is specified explicitly to avoid loading the module.
    :param str fullname:
        Name of the package we're trying to get child modules for

    :return:
        List of submodule name suffixes.
    """
    mod_path = os.path.dirname(path)
    if mod_path != "":
        return [to_text(name) for _, name, _ in pkgutil.iter_modules([mod_path])]
    else:
        # we loaded some weird package in memory, so we'll see if it has a custom loader we can use
        loader = pkgutil.find_loader(fullname)
        return (
            [to_text(name) for name, _ in loader.iter_modules(None)] if loader else []
        )


def _looks_like_script(path):
    """
    Return :data:`True` if the (possibly extensionless) file at `path`
    resembles a Python script. For now we simply verify the file contains
    ASCII text.
    """
    try:
        fp = open(path, "rb")
    except IOError:
        e = sys.exc_info()[1]
        if e.args[0] == errno.EISDIR:
            return False
        raise

    try:
        sample = fp.read(512).decode("latin-1")
        return not set(sample).difference(string.printable)
    finally:
        fp.close()


def _py_filename(path):
    """
    Returns a tuple of a Python path (if the file looks Pythonic) and whether or not
    the Python path is special. Special file paths/modules might only exist in memory
    """
    if not path:
        return None, False

    if path[-4:] in (".pyc", ".pyo"):
        path = path.rstrip("co")

    if path.endswith(".py"):
        return path, False

    if os.path.exists(path) and _looks_like_script(path):
        return path, False

    basepath = os.path.basename(path)
    if basepath in SPECIAL_FILE_PATHS:
        return path, True

    # return None, False means that the filename passed to _py_filename does not appear
    # to be python, and code later will handle when this function returns None
    # see https://github.com/dw/mitogen/pull/715#discussion_r532380528 for how this
    # decision was made to handle non-python files in this manner
    return None, False


def _get_core_source():
    """
    Master version of parent.get_core_source().
    """
    source = inspect.getsource(mitogen.core)
    return mitogen.minify.minimize_source(source)


if mitogen.is_master:
    # TODO: find a less surprising way of installing this.
    mitogen.parent._get_core_source = _get_core_source


LOAD_CONST = dis.opname.index("LOAD_CONST")
IMPORT_NAME = dis.opname.index("IMPORT_NAME")


def _getarg(nextb, c):
    if c >= dis.HAVE_ARGUMENT:
        return nextb() | (nextb() << 8)


if sys.version_info < (3, 0):

    def iter_opcodes(co):
        # Yield `(op, oparg)` tuples from the code object `co`.
        ordit = imap(ord, co.co_code)
        nextb = ordit.next
        return ((c, _getarg(nextb, c)) for c in ordit)


elif sys.version_info < (3, 6):

    def iter_opcodes(co):
        # Yield `(op, oparg)` tuples from the code object `co`.
        ordit = iter(co.co_code)
        nextb = ordit.__next__
        return ((c, _getarg(nextb, c)) for c in ordit)


else:

    def iter_opcodes(co):
        # Yield `(op, oparg)` tuples from the code object `co`.
        ordit = iter(co.co_code)
        nextb = ordit.__next__
        # https://github.com/abarnert/cpython/blob/c095a32f/Python/wordcode.md
        return ((c, nextb()) for c in ordit)


def scan_code_imports(co):
    """
    Given a code object `co`, scan its bytecode yielding any ``IMPORT_NAME``
    and associated prior ``LOAD_CONST`` instructions representing an `Import`
    statement or `ImportFrom` statement.

    :return:
        Generator producing `(level, modname, namelist)` tuples, where:

        * `level`: -1 for normal import, 0, for absolute import, and >0 for
          relative import.
        * `modname`: Name of module to import, or from where `namelist` names
          are imported.
        * `namelist`: for `ImportFrom`, the list of names to be imported from
          `modname`.
    """
    opit = iter_opcodes(co)
    opit, opit2, opit3 = itertools.tee(opit, 3)

    try:
        next(opit2)
        next(opit3)
        next(opit3)
    except StopIteration:
        return

    if sys.version_info >= (2, 5):
        for oparg1, oparg2, (op3, arg3) in izip(opit, opit2, opit3):
            if op3 == IMPORT_NAME:
                op2, arg2 = oparg2
                op1, arg1 = oparg1
                if op1 == op2 == LOAD_CONST:
                    yield (
                        co.co_consts[arg1],
                        co.co_names[arg3],
                        co.co_consts[arg2] or (),
                    )
    else:
        # Python 2.4 did not yet have 'level', so stack format differs.
        for oparg1, (op2, arg2) in izip(opit, opit2):
            if op2 == IMPORT_NAME:
                op1, arg1 = oparg1
                if op1 == LOAD_CONST:
                    yield (-1, co.co_names[arg2], co.co_consts[arg1] or ())


class ThreadWatcher(object):
    """
    Manage threads that wait for another thread to shut down, before invoking
    `on_join()` for each associated ThreadWatcher.

    In CPython it seems possible to use this method to ensure a non-main thread
    is signalled when the main thread has exited, using a third thread as a
    proxy.
    """

    #: Protects remaining _cls_* members.
    _cls_lock = threading.Lock()

    #: PID of the process that last modified the class data. If the PID
    #: changes, it means the thread watch dict refers to threads that no longer
    #: exist in the current process (since it forked), and so must be reset.
    _cls_pid = None

    #: Map watched Thread -> list of ThreadWatcher instances.
    _cls_instances_by_target = {}

    #: Map watched Thread -> watcher Thread for each watched thread.
    _cls_thread_by_target = {}

    @classmethod
    def _reset(cls):
        """
        If we have forked since the watch dictionaries were initialized, all
        that has is garbage, so clear it.
        """
        if os.getpid() != cls._cls_pid:
            cls._cls_pid = os.getpid()
            cls._cls_instances_by_target.clear()
            cls._cls_thread_by_target.clear()

    def __init__(self, target, on_join):
        self.target = target
        self.on_join = on_join

    @classmethod
    def _watch(cls, target):
        target.join()
        for watcher in cls._cls_instances_by_target[target]:
            watcher.on_join()

    def install(self):
        self._cls_lock.acquire()
        try:
            self._reset()
            lst = self._cls_instances_by_target.setdefault(self.target, [])
            lst.append(self)
            if self.target not in self._cls_thread_by_target:
                self._cls_thread_by_target[self.target] = threading.Thread(
                    name="mitogen.master.join_thread_async",
                    target=self._watch,
                    args=(self.target,),
                )
                self._cls_thread_by_target[self.target].start()
        finally:
            self._cls_lock.release()

    def remove(self):
        self._cls_lock.acquire()
        try:
            self._reset()
            lst = self._cls_instances_by_target.get(self.target, [])
            if self in lst:
                lst.remove(self)
        finally:
            self._cls_lock.release()

    @classmethod
    def watch(cls, target, on_join):
        watcher = cls(target, on_join)
        watcher.install()
        return watcher


class LogForwarder(object):
    """
    Install a :data:`mitogen.core.FORWARD_LOG` handler that delivers forwarded
    log events into the local logging framework. This is used by the master's
    :class:`Router`.

    The forwarded :class:`logging.LogRecord` objects are delivered to loggers
    under ``mitogen.ctx.*`` corresponding to their
    :attr:`mitogen.core.Context.name`, with the message prefixed with the
    logger name used in the child. The records include some extra attributes:

    * ``mitogen_message``: Unicode original message without the logger name
      prepended.
    * ``mitogen_context``: :class:`mitogen.parent.Context` reference to the
      source context.
    * ``mitogen_name``: Original logger name.

    :param mitogen.master.Router router:
        Router to install the handler on.
    """

    def __init__(self, router):
        self._router = router
        self._cache = {}
        router.add_handler(
            fn=self._on_forward_log,
            handle=mitogen.core.FORWARD_LOG,
        )

    def _on_forward_log(self, msg):
        if msg.is_dead:
            return

        context = self._router.context_by_id(msg.src_id)
        if context is None:
            LOG.error("%s: dropping log from unknown context %d", self, msg.src_id)
            return

        name, level_s, s = msg.data.decode("utf-8", "replace").split("\x00", 2)

        logger_name = f"{name}.[{context.name}]"
        logger = self._cache.get(logger_name)
        if logger is None:
            self._cache[logger_name] = logger = logging.getLogger(logger_name)

        # See logging.Handler.makeRecord()
        record = logging.LogRecord(
            name=logger.name,
            level=int(level_s),
            pathname="(unknown file)",
            lineno=0,
            msg=s,
            args=(),
            exc_info=None,
        )
        record.mitogen_message = s
        record.mitogen_context = self._router.context_by_id(msg.src_id)
        record.mitogen_name = name
        logger.handle(record)

    def __repr__(self):
        return f"LogForwarder({self._router!r})"


class FinderMethod(object):
    """
    Interface to a method for locating a Python module or package given its
    name according to the running Python interpreter. You'd think this was a
    simple task, right? Naive young fellow, welcome to the real world.
    """

    def __repr__(self):
        return f"{type(self).__name__}()"

    def find(self, fullname):
        """
        Accept a canonical module name as would be found in :data:`sys.modules`
        and return a `(path, source, is_pkg)` tuple, where:

        * `path`: Unicode string containing path to source file.
        * `source`: Bytestring containing source file's content.
        * `is_pkg`: :data:`True` if `fullname` is a package.

        :returns:
            :data:`None` if not found, or tuple as described above.
        """
        raise NotImplementedError()


class DefectivePython3xMainMethod(FinderMethod):
    """
    Recent versions of Python 3.x introduced an incomplete notion of
    importer specs, and in doing so created permanent asymmetry in the
    :mod:`pkgutil` interface handling for the :mod:`__main__` module. Therefore
    we must handle :mod:`__main__` specially.
    """

    def find(self, fullname):
        """
        Find :mod:`__main__` using its :data:`__file__` attribute.
        """
        if fullname != "__main__":
            return None

        mod = sys.modules.get(fullname)
        if not mod:
            return None

        path = getattr(mod, "__file__", None)
        if not (path is not None and os.path.exists(path) and _looks_like_script(path)):
            return None

        fp = open(path, "rb")
        try:
            source = fp.read()
        finally:
            fp.close()

        return path, source, False


class PkgutilMethod(FinderMethod):
    """
    Attempt to fetch source code via pkgutil. In an ideal world, this would
    be the only required implementation of get_module().
    """

    def find(self, fullname):
        """
        Find `fullname` using :func:`pkgutil.find_loader`.
        """
        try:
            # Pre-'import spec' this returned None, in Python3.6 it raises
            # ImportError.
            loader = pkgutil.find_loader(fullname)
        except ImportError:
            e = sys.exc_info()[1]
            LOG.debug("%r._get_module_via_pkgutil(%r): %s", self, fullname, e)
            return None

        IOLOG.debug("%r._get_module_via_pkgutil(%r) -> %r", self, fullname, loader)
        if not loader:
            return

        try:
            path, is_special = _py_filename(loader.get_filename(fullname))
            source = loader.get_source(fullname)
            is_pkg = loader.is_package(fullname)

            # workaround for special python modules that might only exist in memory
            if is_special and is_pkg and not source:
                source = "\n"
        except (AttributeError, ImportError):
            # - Per PEP-302, get_source() and is_package() are optional,
            #   calling them may throw AttributeError.
            # - get_filename() may throw ImportError if pkgutil.find_loader()
            #   picks a "parent" package's loader for some crap that's been
            #   stuffed in sys.modules, for example in the case of urllib3:
            #       "loader for urllib3.contrib.pyopenssl cannot handle
            #        requests.packages.urllib3.contrib.pyopenssl"
            e = sys.exc_info()[1]
            LOG.debug("%r: loading %r using %r failed: %s", self, fullname, loader, e)
            return

        if path is None or source is None:
            return

        if isinstance(source, mitogen.core.UnicodeType):
            # get_source() returns "string" according to PEP-302, which was
            # reinterpreted for Python 3 to mean a Unicode string.
            source = source.encode("utf-8")

        return path, source, is_pkg


class SysModulesMethod(FinderMethod):
    """
    Attempt to fetch source code via :data:`sys.modules`. This was originally
    specifically to support :mod:`__main__`, but it may catch a few more cases.
    """

    def find(self, fullname):
        """
        Find `fullname` using its :data:`__file__` attribute.
        """
        module = sys.modules.get(fullname)
        if not isinstance(module, types.ModuleType):
            LOG.debug(
                "%r: sys.modules[%r] absent or not a regular module", self, fullname
            )
            return

        LOG.debug("_get_module_via_sys_modules(%r) -> %r", fullname, module)
        alleged_name = getattr(module, "__name__", None)
        if alleged_name != fullname:
            LOG.debug(
                "sys.modules[%r].__name__ is incorrect, assuming "
                "this is a hacky module alias and ignoring it. "
                "Got %r, module object: %r",
                fullname,
                alleged_name,
                module,
            )
            return

        path, _ = _py_filename(getattr(module, "__file__", ""))
        if not path:
            return

        LOG.debug("%r: sys.modules[%r]: found %s", self, fullname, path)
        is_pkg = hasattr(module, "__path__")
        try:
            source = inspect.getsource(module)
        except IOError:
            # Work around inspect.getsourcelines() bug for 0-byte __init__.py
            # files.
            if not is_pkg:
                raise
            source = "\n"

        if isinstance(source, mitogen.core.UnicodeType):
            # get_source() returns "string" according to PEP-302, which was
            # reinterpreted for Python 3 to mean a Unicode string.
            source = source.encode("utf-8")

        return path, source, is_pkg


class ParentEnumerationMethod(FinderMethod):
    """
    Attempt to fetch source code by examining the module's (hopefully less
    insane) parent package, and if no insane parents exist, simply use
    :mod:`sys.path` to search for it from scratch on the filesystem using the
    normal Python lookup mechanism.

    This is required for older versions of :mod:`ansible.compat.six`,
    :mod:`plumbum.colors`, Ansible 2.8 :mod:`ansible.module_utils.distro` and
    its submodule :mod:`ansible.module_utils.distro._distro`.

    When some package dynamically replaces itself in :data:`sys.modules`, but
    only conditionally according to some program logic, it is possible that
    children may attempt to load modules and subpackages from it that can no
    longer be resolved by examining a (corrupted) parent.

    For cases like :mod:`ansible.module_utils.distro`, this must handle cases
    where a package transmuted itself into a totally unrelated module during
    import and vice versa, where :data:`sys.modules` is replaced with junk that
    makes it impossible to discover the loaded module using the in-memory
    module object or any parent package's :data:`__path__`, since they have all
    been overwritten. Some men just want to watch the world burn.
    """

    def _find_sane_parent(self, fullname):
        """
        Iteratively search :data:`sys.modules` for the least indirect parent of
        `fullname` that is loaded and contains a :data:`__path__` attribute.

        :return:
            `(parent_name, path, modpath)` tuple, where:

                * `modname`: canonical name of the found package, or the empty
                   string if none is found.
                * `search_path`: :data:`__path__` attribute of the least
                   indirect parent found, or :data:`None` if no indirect parent
                   was found.
                * `modpath`: list of module name components leading from `path`
                   to the target module.
        """
        path = None
        modpath = []
        while True:
            pkgname, _, modname = str_rpartition(to_text(fullname), u".")
            modpath.insert(0, modname)
            if not pkgname:
                return [], None, modpath

            pkg = sys.modules.get(pkgname)
            path = getattr(pkg, "__path__", None)
            if pkg and path:
                return pkgname.split("."), path, modpath

            LOG.debug("%r: %r lacks __path__ attribute", self, pkgname)
            fullname = pkgname

    def _found_package(self, fullname, path):
        path = os.path.join(path, "__init__.py")
        LOG.debug("%r: %r is PKG_DIRECTORY: %r", self, fullname, path)
        return self._found_module(
            fullname=fullname,
            path=path,
            fp=open(path, "rb"),
            is_pkg=True,
        )

    def _found_module(self, fullname, path, fp, is_pkg=False):
        try:
            path, _ = _py_filename(path)
            if not path:
                return

            source = fp.read()
        finally:
            if fp:
                fp.close()

        if isinstance(source, mitogen.core.UnicodeType):
            # get_source() returns "string" according to PEP-302, which was
            # reinterpreted for Python 3 to mean a Unicode string.
            source = source.encode("utf-8")
        return path, source, is_pkg

    def _find_one_component(self, modname, search_path):
        try:
            # fp, path, (suffix, _, kind) = imp.find_module(modname, search_path)
            return imp.find_module(modname, search_path)
        except ImportError:
            e = sys.exc_info()[1]
            LOG.debug(
                "%r: imp.find_module(%r, %r) -> %s", self, modname, [search_path], e
            )
            return None

    def find(self, fullname):
        """
        See implementation for a description of how this works.
        """
        # if fullname not in sys.modules:
        # Don't attempt this unless a module really exists in sys.modules,
        # else we could return junk.
        # return

        fullname = to_text(fullname)
        modname, search_path, modpath = self._find_sane_parent(fullname)
        while True:
            tup = self._find_one_component(modpath.pop(0), search_path)
            if tup is None:
                return None

            fp, path, (suffix, _, kind) = tup
            if modpath:
                # Still more components to descent. Result must be a package
                if fp:
                    fp.close()
                if kind != imp.PKG_DIRECTORY:
                    LOG.debug(
                        "%r: %r appears to be child of non-package %r",
                        self,
                        fullname,
                        path,
                    )
                    return None
                search_path = [path]
            elif kind == imp.PKG_DIRECTORY:
                return self._found_package(fullname, path)
            else:
                return self._found_module(fullname, path, fp)


class ModuleFinder(object):
    """
    Given the name of a loaded module, make a best-effort attempt at finding
    related modules likely needed by a child context requesting the original
    module.
    """

    def __init__(self):
        #: Import machinery is expensive, keep :py:meth`:get_module_source`
        #: results around.
        self._found_cache = {}

        #: Avoid repeated dependency scanning, which is expensive.
        self._related_cache = {}

    def __repr__(self):
        return "ModuleFinder()"

    def add_source_override(self, fullname, path, source, is_pkg):
        """
        Explicitly install a source cache entry, preventing usual lookup
        methods from being used.

        Beware the value of `path` is critical when `is_pkg` is specified,
        since it directs where submodules are searched for.

        :param str fullname:
            Name of the module to override.
        :param str path:
            Module's path as it will appear in the cache.
        :param bytes source:
            Module source code as a bytestring.
        :param bool is_pkg:
            :data:`True` if the module is a package.
        """
        self._found_cache[fullname] = (path, source, is_pkg)

    get_module_methods = [
        DefectivePython3xMainMethod(),
        PkgutilMethod(),
        SysModulesMethod(),
        ParentEnumerationMethod(),
    ]

    def get_module_source(self, fullname):
        """
        Given the name of a loaded module `fullname`, attempt to find its
        source code.

        :returns:
            Tuple of `(module path, source text, is package?)`, or :data:`None`
            if the source cannot be found.
        """
        tup = self._found_cache.get(fullname)
        if tup:
            return tup

        for method in self.get_module_methods:
            tup = method.find(fullname)
            if tup:
                # LOG.debug('%r returned %r', method, tup)
                break
        else:
            tup = None, None, None
            LOG.debug("get_module_source(%r): cannot find source", fullname)

        self._found_cache[fullname] = tup
        return tup

    def resolve_relpath(self, fullname, level):
        """
        Given an ImportFrom AST node, guess the prefix that should be tacked on
        to an alias name to produce a canonical name. `fullname` is the name of
        the module in which the ImportFrom appears.
        """
        mod = sys.modules.get(fullname, None)
        if hasattr(mod, "__path__"):
            fullname += ".__init__"

        if level == 0 or not fullname:
            return ""

        bits = fullname.split(".")
        if len(bits) <= level:
            # This would be an ImportError in real code.
            return ""

        return ".".join(bits[:-level]) + "."

    def generate_parent_names(self, fullname):
        while "." in fullname:
            fullname, _, _ = str_rpartition(to_text(fullname), u".")
            yield fullname

    def find_related_imports(self, fullname):
        """
        Return a list of non-stdlib modules that are directly imported by
        `fullname`, plus their parents.

        The list is determined by retrieving the source code of
        `fullname`, compiling it, and examining all IMPORT_NAME ops.

        :param fullname: Fully qualified name of an *already imported* module
            for which source code can be retrieved
        :type fullname: str
        """
        related = self._related_cache.get(fullname)
        if related is not None:
            return related

        modpath, src, _ = self.get_module_source(fullname)
        if src is None:
            return []

        maybe_names = list(self.generate_parent_names(fullname))

        co = compile(src, modpath, "exec")
        for level, modname, namelist in scan_code_imports(co):
            if level == -1:
                modnames = [modname, f"{fullname}.{modname}"]
            else:
                modnames = [f"{self.resolve_relpath(fullname, level)}{modname}"]

            maybe_names.extend(modnames)
            maybe_names.extend(
                f"{mname}.{name}" for mname in modnames for name in namelist
            )

        return self._related_cache.setdefault(
            fullname,
            sorted(
                set(
                    mitogen.core.to_text(name)
                    for name in maybe_names
                    if sys.modules.get(name) is not None
                    and not is_stdlib_name(name)
                    and u"six.moves" not in name  # TODO: crap
                )
            ),
        )

    def find_related(self, fullname):
        """
        Return a list of non-stdlib modules that are imported directly or
        indirectly by `fullname`, plus their parents.

        This method is like :py:meth:`find_related_imports`, but also
        recursively searches any modules which are imported by `fullname`.

        :param fullname: Fully qualified name of an *already imported* module
            for which source code can be retrieved
        :type fullname: str
        """
        stack = [fullname]
        found = set()

        while stack:
            name = stack.pop(0)
            names = self.find_related_imports(name)
            stack.extend(set(names).difference(set(found).union(stack)))
            found.update(names)

        found.discard(fullname)
        return sorted(found)


class ModuleResponder(object):
    def __init__(self, router):
        self._log = logging.getLogger("mitogen.responder")
        self._router = router
        self._finder = ModuleFinder()
        self._cache = {}  # fullname -> pickled
        self.blacklist = []
        self.whitelist = [""]

        #: Context -> set([fullname, ..])
        self._forwarded_by_context = {}

        #: Number of GET_MODULE messages received.
        self.get_module_count = 0
        #: Total time spent in uncached GET_MODULE.
        self.get_module_secs = 0.0
        #: Total time spent minifying modules.
        self.minify_secs = 0.0
        #: Number of successful LOAD_MODULE messages sent.
        self.good_load_module_count = 0
        #: Total bytes in successful LOAD_MODULE payloads.
        self.good_load_module_size = 0
        #: Number of negative LOAD_MODULE messages sent.
        self.bad_load_module_count = 0

        router.add_handler(
            fn=self._on_get_module,
            handle=mitogen.core.GET_MODULE,
        )

    def __repr__(self):
        return "ModuleResponder"

    def add_source_override(self, fullname, path, source, is_pkg):
        """
        See :meth:`ModuleFinder.add_source_override`.
        """
        self._finder.add_source_override(fullname, path, source, is_pkg)

    MAIN_RE = re.compile(b(r"^if\s+__name__\s*==\s*.__main__.\s*:"), re.M)
    main_guard_msg = (
        "A child context attempted to import __main__, however the main "
        "module present in the master process lacks an execution guard. "
        "Update %r to prevent unintended execution, using a guard like:\n"
        "\n"
        "    if __name__ == '__main__':\n"
        "        # your code here.\n"
    )

    def whitelist_prefix(self, fullname):
        if self.whitelist == [""]:
            self.whitelist = ["mitogen"]
        self.whitelist.append(fullname)

    def blacklist_prefix(self, fullname):
        self.blacklist.append(fullname)

    def neutralize_main(self, path, src):
        """
        Given the source for the __main__ module, try to find where it begins
        conditional execution based on a "if __name__ == '__main__'" guard, and
        remove any code after that point.
        """
        match = self.MAIN_RE.search(src)
        if match:
            return src[: match.start()]

        if b("mitogen.main(") in src:
            return src

        self._log.error(self.main_guard_msg, path)
        raise ImportError("refused")

    def _make_negative_response(self, fullname):
        return (fullname, None, None, None, ())

    minify_safe_re = re.compile(b(r"\s+#\s*!mitogen:\s*minify_safe"))

    def _build_tuple(self, fullname):
        if fullname in self._cache:
            return self._cache[fullname]

        if mitogen.core.is_blacklisted_import(self, fullname):
            raise ImportError("blacklisted")

        path, source, is_pkg = self._finder.get_module_source(fullname)
        if path and is_stdlib_path(path):
            # Prevent loading of 2.x<->3.x stdlib modules! This costs one
            # RTT per hit, so a client-side solution is also required.
            self._log.debug("refusing to serve stdlib module %r", fullname)
            tup = self._make_negative_response(fullname)
            self._cache[fullname] = tup
            return tup

        if source is None:
            # TODO: make this .warning() or similar again once importer has its
            # own logging category.
            self._log.debug("could not find source for %r", fullname)
            tup = self._make_negative_response(fullname)
            self._cache[fullname] = tup
            return tup

        if self.minify_safe_re.search(source):
            # If the module contains a magic marker, it's safe to minify.
            t0 = mitogen.core.now()
            source = mitogen.minify.minimize_source(source).encode("utf-8")
            self.minify_secs += mitogen.core.now() - t0

        if is_pkg:
            pkg_present = get_child_modules(path, fullname)
            self._log.debug(
                "%s is a package at %s with submodules %r", fullname, path, pkg_present
            )
        else:
            pkg_present = None

        if fullname == "__main__":
            source = self.neutralize_main(path, source)
        compressed = mitogen.core.Blob(zlib.compress(source, 9))
        related = [
            to_text(name)
            for name in self._finder.find_related(fullname)
            if not mitogen.core.is_blacklisted_import(self, name)
        ]
        # 0:fullname 1:pkg_present 2:path 3:compressed 4:related
        tup = (to_text(fullname), pkg_present, to_text(path), compressed, related)
        self._cache[fullname] = tup
        return tup

    def _send_load_module(self, stream, fullname):
        if fullname not in stream.protocol.sent_modules:
            tup = self._build_tuple(fullname)
            msg = mitogen.core.Message.pickled(
                tup,
                dst_id=stream.protocol.remote_id,
                handle=mitogen.core.LOAD_MODULE,
            )
            self._log.debug(
                "sending %s (%.2f KiB) to %s",
                fullname,
                len(msg.data) / 1024.0,
                stream.name,
            )
            self._router._async_route(msg)
            stream.protocol.sent_modules.add(fullname)
            if tup[2] is not None:
                self.good_load_module_count += 1
                self.good_load_module_size += len(msg.data)
            else:
                self.bad_load_module_count += 1

    def _send_module_load_failed(self, stream, fullname):
        self.bad_load_module_count += 1
        stream.protocol.send(
            mitogen.core.Message.pickled(
                self._make_negative_response(fullname),
                dst_id=stream.protocol.remote_id,
                handle=mitogen.core.LOAD_MODULE,
            )
        )

    def _send_module_and_related(self, stream, fullname):
        if fullname in stream.protocol.sent_modules:
            return

        try:
            tup = self._build_tuple(fullname)
            for name in tup[4]:  # related
                parent, _, _ = str_partition(name, ".")
                if parent != fullname and parent not in stream.protocol.sent_modules:
                    # Parent hasn't been sent, so don't load submodule yet.
                    continue

                self._send_load_module(stream, name)
            self._send_load_module(stream, fullname)
        except Exception:
            LOG.debug("While importing %r", fullname, exc_info=True)
            self._send_module_load_failed(stream, fullname)

    def _on_get_module(self, msg):
        if msg.is_dead:
            return

        stream = self._router.stream_by_id(msg.src_id)
        if stream is None:
            return

        fullname = msg.data.decode()
        self._log.debug("%s requested module %s", stream.name, fullname)
        self.get_module_count += 1
        if fullname in stream.protocol.sent_modules:
            LOG.warning(
                "_on_get_module(): dup request for %r from %r", fullname, stream
            )

        t0 = mitogen.core.now()
        try:
            self._send_module_and_related(stream, fullname)
        finally:
            self.get_module_secs += mitogen.core.now() - t0

    def _send_forward_module(self, stream, context, fullname):
        if stream.protocol.remote_id != context.context_id:
            stream.protocol._send(
                mitogen.core.Message(
                    data=b(f"{context.context_id}\x00{fullname}"),
                    handle=mitogen.core.FORWARD_MODULE,
                    dst_id=stream.protocol.remote_id,
                )
            )

    def _forward_one_module(self, context, fullname):
        forwarded = self._forwarded_by_context.get(context)
        if forwarded is None:
            forwarded = set()
            self._forwarded_by_context[context] = forwarded

        if fullname in forwarded:
            return

        path = []
        while fullname:
            path.append(fullname)
            fullname, _, _ = str_rpartition(fullname, u".")

        stream = self._router.stream_by_id(context.context_id)
        if stream is None:
            LOG.debug(
                "%r: dropping forward of %s to no longer existent " "%r",
                self,
                path[0],
                context,
            )
            return

        for fullname in reversed(path):
            self._send_module_and_related(stream, fullname)
            self._send_forward_module(stream, context, fullname)

    def _forward_modules(self, context, fullnames):
        IOLOG.debug("%r._forward_modules(%r, %r)", self, context, fullnames)
        for fullname in fullnames:
            self._forward_one_module(context, mitogen.core.to_text(fullname))

    def forward_modules(self, context, fullnames):
        self._router.broker.defer(self._forward_modules, context, fullnames)


class Broker(mitogen.core.Broker):
    """
    .. note::

        You may construct as many brokers as desired, and use the same broker
        for multiple routers, however usually only one broker need exist.
        Multiple brokers may be useful when dealing with sets of children with
        differing lifetimes. For example, a subscription service where
        non-payment results in termination for one customer.

    :param bool install_watcher:
        If :data:`True`, an additional thread is started to monitor the
        lifetime of the main thread, triggering :meth:`shutdown`
        automatically in case the user forgets to call it, or their code
        crashed.

        You should not rely on this functionality in your program, it is only
        intended as a fail-safe and to simplify the API for new users. In
        particular, alternative Python implementations may not be able to
        support watching the main thread.
    """

    shutdown_timeout = 5.0
    _watcher = None
    poller_class = mitogen.parent.PREFERRED_POLLER

    def __init__(self, install_watcher=True):
        if install_watcher:
            self._watcher = ThreadWatcher.watch(
                target=threading.currentThread(),
                on_join=self.shutdown,
            )
        super(Broker, self).__init__()
        self.timers = mitogen.parent.TimerList()

    def shutdown(self):
        super(Broker, self).shutdown()
        if self._watcher:
            self._watcher.remove()


class Router(mitogen.parent.Router):
    """
    Extend :class:`mitogen.core.Router` with functionality useful to masters,
    and child contexts who later become masters. Currently when this class is
    required, the target context's router is upgraded at runtime.

    .. note::

        You may construct as many routers as desired, and use the same broker
        for multiple routers, however usually only one broker and router need
        exist. Multiple routers may be useful when dealing with separate trust
        domains, for example, manipulating infrastructure belonging to separate
        customers or projects.

    :param mitogen.master.Broker broker:
        Broker to use. If not specified, a private :class:`Broker` is created.

    :param int max_message_size:
        Override the maximum message size this router is willing to receive or
        transmit. Any value set here is automatically inherited by any children
        created by the router.

        This has a liberal default of 128 MiB, but may be set much lower.
        Beware that setting it below 64KiB may encourage unexpected failures as
        parents and children can no longer route large Python modules that may
        be required by your application.
    """

    broker_class = Broker

    #: When :data:`True`, cause the broker thread and any subsequent broker and
    #: main threads existing in any child to write
    #: ``/tmp/mitogen.stats.<pid>.<thread_name>.log`` containing a
    #: :mod:`cProfile` dump on graceful exit. Must be set prior to construction
    #: of any :class:`Broker`, e.g. via::
    #:
    #:      mitogen.master.Router.profiling = True
    profiling = os.environ.get("MITOGEN_PROFILING") is not None

    def __init__(self, broker=None, max_message_size=None):
        if broker is None:
            broker = self.broker_class()
        if max_message_size:
            self.max_message_size = max_message_size
        super(Router, self).__init__(broker)
        self.upgrade()

    def upgrade(self):
        self.id_allocator = IdAllocator(self)
        self.responder = ModuleResponder(self)
        self.log_forwarder = LogForwarder(self)
        self.route_monitor = mitogen.parent.RouteMonitor(router=self)
        self.add_handler(  # TODO: cutpaste.
            fn=self._on_detaching,
            handle=mitogen.core.DETACHING,
            persist=True,
        )

    def _on_broker_exit(self):
        super(Router, self)._on_broker_exit()
        dct = self.get_stats()
        dct["self"] = self
        dct["minify_ms"] = 1000 * dct["minify_secs"]
        dct["get_module_ms"] = 1000 * dct["get_module_secs"]
        dct["good_load_module_size_kb"] = dct["good_load_module_size"] / 1024.0
        dct["good_load_module_size_avg"] = (
            dct["good_load_module_size"] / (float(dct["good_load_module_count"]) or 1.0)
        ) / 1024.0

        LOG.debug(
            "%(self)r: stats: "
            "%(get_module_count)d module requests in "
            "%(get_module_ms)d ms, "
            "%(good_load_module_count)d sent "
            "(%(minify_ms)d ms minify time), "
            "%(bad_load_module_count)d negative responses. "
            "Sent %(good_load_module_size_kb).01f kb total, "
            "%(good_load_module_size_avg).01f kb avg." % dct
        )

    def get_stats(self):
        """
        Return performance data for the module responder.

        :returns:

            Dict containing keys:

            * `get_module_count`: Integer count of
              :data:`mitogen.core.GET_MODULE` messages received.
            * `get_module_secs`: Floating point total seconds spent servicing
              :data:`mitogen.core.GET_MODULE` requests.
            * `good_load_module_count`: Integer count of successful
              :data:`mitogen.core.LOAD_MODULE` messages sent.
            * `good_load_module_size`: Integer total bytes sent in
              :data:`mitogen.core.LOAD_MODULE` message payloads.
            * `bad_load_module_count`: Integer count of negative
              :data:`mitogen.core.LOAD_MODULE` messages sent.
            * `minify_secs`: CPU seconds spent minifying modules marked
               minify-safe.
        """
        return {
            "get_module_count": self.responder.get_module_count,
            "get_module_secs": self.responder.get_module_secs,
            "good_load_module_count": self.responder.good_load_module_count,
            "good_load_module_size": self.responder.good_load_module_size,
            "bad_load_module_count": self.responder.bad_load_module_count,
            "minify_secs": self.responder.minify_secs,
        }

    def enable_debug(self):
        """
        Cause this context and any descendant child contexts to write debug
        logs to ``/tmp/mitogen.<pid>.log``.
        """
        mitogen.core.enable_debug_logging()
        self.debug = True

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, tb):
        self.broker.shutdown()
        self.broker.join()

    def disconnect_stream(self, stream):
        self.broker.defer(stream.on_disconnect, self.broker)

    def disconnect_all(self):
        # making stream_by_id python3-safe by converting stream_by_id values iter to list
        for stream in list(self._stream_by_id.values()):
            self.disconnect_stream(stream)


class IdAllocator(object):
    """
    Allocate IDs for new contexts constructed locally, and blocks of IDs for
    children to allocate their own IDs using
    :class:`mitogen.parent.ChildIdAllocator` without risk of conflict, and
    without necessitating network round-trips for each new context.

    This class responds to :data:`mitogen.core.ALLOCATE_ID` messages received
    from children by replying with fresh block ID allocations.

    The master's :class:`IdAllocator` instance can be accessed via
    :attr:`mitogen.master.Router.id_allocator`.
    """

    #: Block allocations are made in groups of 1000 by default.
    BLOCK_SIZE = 1000

    def __init__(self, router):
        self.router = router
        self.next_id = 1
        self.lock = threading.Lock()
        router.add_handler(
            fn=self.on_allocate_id,
            handle=mitogen.core.ALLOCATE_ID,
        )

    def __repr__(self):
        return f"IdAllocator({self.router!r})"

    def allocate(self):
        """
        Allocate a context ID by directly incrementing an internal counter.

        :returns:
            The new context ID.
        """
        self.lock.acquire()
        try:
            id_ = self.next_id
            self.next_id += 1
            return id_
        finally:
            self.lock.release()

    def allocate_block(self):
        """
        Allocate a block of IDs for use in a child context.

        This function is safe to call from any thread.

        :returns:
            Tuple of the form `(id, end_id)` where `id` is the first usable ID
            and `end_id` is the last usable ID.
        """
        self.lock.acquire()
        try:
            id_ = self.next_id
            self.next_id += self.BLOCK_SIZE
            end_id = id_ + self.BLOCK_SIZE
            LOG.debug("%r: allocating [%d..%d)", self, id_, end_id)
            return id_, end_id
        finally:
            self.lock.release()

    def on_allocate_id(self, msg):
        if msg.is_dead:
            return

        id_, last_id = self.allocate_block()
        requestee = self.router.context_by_id(msg.src_id)
        LOG.debug("%r: allocating [%r..%r) to %r", self, id_, last_id, requestee)
        msg.reply((id_, last_id))
