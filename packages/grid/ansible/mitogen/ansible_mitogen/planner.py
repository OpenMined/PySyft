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

"""
Classes to detect each case from [0] and prepare arguments necessary for the
corresponding Runner class within the target, including preloading requisite
files/modules known missing.

[0] "Ansible Module Architecture", developing_program_flow_modules.html
"""

# future
from __future__ import absolute_import
from __future__ import unicode_literals

# stdlib
import json
import logging
import os
import random

# third party
from ansible.collections.list import list_collection_dirs
import ansible.errors
from ansible.executor import module_common
import ansible.module_utils
import ansible.release
import ansible_mitogen.loaders
import ansible_mitogen.parsing
import ansible_mitogen.target
import mitogen.core
import mitogen.select

LOG = logging.getLogger(__name__)
NO_METHOD_MSG = "Mitogen: no invocation method found for: "
NO_INTERPRETER_MSG = "module (%s) is missing interpreter line"
# NOTE: Ansible 2.10 no longer has a `.` at the end of NO_MODULE_MSG error
NO_MODULE_MSG = "The module %s was not found in configured module paths"

_planner_by_path = {}


class Invocation(object):
    """
    Collect up a module's execution environment then use it to invoke
    target.run_module() or helpers.run_module_async() in the target context.
    """

    def __init__(
        self,
        action,
        connection,
        module_name,
        module_args,
        task_vars,
        templar,
        env,
        wrap_async,
        timeout_secs,
    ):
        #: ActionBase instance invoking the module. Required to access some
        #: output postprocessing methods that don't belong in ActionBase at
        #: all.
        self.action = action
        #: Ansible connection to use to contact the target. Must be an
        #: ansible_mitogen connection.
        self.connection = connection
        #: Name of the module ('command', 'shell', etc.) to execute.
        self.module_name = module_name
        #: Final module arguments.
        self.module_args = module_args
        #: Task variables, needed to extract ansible_*_interpreter.
        self.task_vars = task_vars
        #: Templar, needed to extract ansible_*_interpreter.
        self.templar = templar
        #: Final module environment.
        self.env = env
        #: Boolean, if :py:data:`True`, launch the module asynchronously.
        self.wrap_async = wrap_async
        #: Integer, if >0, limit the time an asynchronous job may run for.
        self.timeout_secs = timeout_secs
        #: Initially ``None``, but set by :func:`invoke`. The path on the
        #: master to the module's implementation file.
        self.module_path = None
        #: Initially ``None``, but set by :func:`invoke`. The raw source or
        #: binary contents of the module.
        self._module_source = None
        #: Initially ``{}``, but set by :func:`invoke`. Optional source to send
        #: to :func:`propagate_paths_and_modules` to fix Python3.5 relative import errors
        self._overridden_sources = {}
        #: Initially ``set()``, but set by :func:`invoke`. Optional source paths to send
        #: to :func:`propagate_paths_and_modules` to handle loading source dependencies from
        #: places outside of the main source path, such as collections
        self._extra_sys_paths = set()

    def get_module_source(self):
        if self._module_source is None:
            self._module_source = read_file(self.module_path)
        return self._module_source

    def __repr__(self):
        return "Invocation(module_name=%s)" % (self.module_name,)


class Planner(object):
    """
    A Planner receives a module name and the contents of its implementation
    file, indicates whether or not it understands how to run the module, and
    exports a method to run the module.
    """

    def __init__(self, invocation):
        self._inv = invocation

    @classmethod
    def detect(cls, path, source):
        """
        Return true if the supplied `invocation` matches the module type
        implemented by this planner.
        """
        raise NotImplementedError()

    def should_fork(self):
        """
        Asynchronous tasks must always be forked.
        """
        return self._inv.wrap_async

    def get_push_files(self):
        """
        Return a list of files that should be propagated to the target context
        using PushFileService. The default implementation pushes nothing.
        """
        return []

    def get_module_deps(self):
        """
        Return a list of the Python module names imported by the module.
        """
        return []

    def get_kwargs(self, **kwargs):
        """
        If :meth:`detect` returned :data:`True`, plan for the module's
        execution, including granting access to or delivering any files to it
        that are known to be absent, and finally return a dict::

            {
                # Name of the class from runners.py that implements the
                # target-side execution of this module type.
                "runner_name": "...",

                # Remaining keys are passed to the constructor of the class
                # named by `runner_name`.
            }
        """
        binding = self._inv.connection.get_binding()

        new = dict((mitogen.core.UnicodeType(k), kwargs[k]) for k in kwargs)
        new.setdefault("good_temp_dir", self._inv.connection.get_good_temp_dir())
        new.setdefault("cwd", self._inv.connection.get_default_cwd())
        new.setdefault("extra_env", self._inv.connection.get_default_env())
        new.setdefault("emulate_tty", True)
        new.setdefault("service_context", binding.get_child_service_context())
        return new

    def __repr__(self):
        return "%s()" % (type(self).__name__,)


class BinaryPlanner(Planner):
    """
    Binary modules take their arguments and will return data to Ansible in the
    same way as want JSON modules.
    """

    runner_name = "BinaryRunner"

    @classmethod
    def detect(cls, path, source):
        return module_common._is_binary(source)

    def get_push_files(self):
        return [mitogen.core.to_text(self._inv.module_path)]

    def get_kwargs(self, **kwargs):
        return super(BinaryPlanner, self).get_kwargs(
            runner_name=self.runner_name,
            module=self._inv.module_name,
            path=self._inv.module_path,
            json_args=json.dumps(self._inv.module_args),
            env=self._inv.env,
            **kwargs
        )


class ScriptPlanner(BinaryPlanner):
    """
    Common functionality for script module planners -- handle interpreter
    detection and rewrite.
    """

    def _rewrite_interpreter(self, path):
        """
        Given the original interpreter binary extracted from the script's
        interpreter line, look up the associated `ansible_*_interpreter`
        variable, render it and return it.

        :param str path:
            Absolute UNIX path to original interpreter.

        :returns:
            Shell fragment prefix used to execute the script via "/bin/sh -c".
            While `ansible_*_interpreter` documentation suggests shell isn't
            involved here, the vanilla implementation uses it and that use is
            exploited in common playbooks.
        """
        key = "ansible_%s_interpreter" % os.path.basename(path).strip()
        try:
            template = self._inv.task_vars[key]
        except KeyError:
            return path

        return mitogen.utils.cast(self._inv.templar.template(template))

    def _get_interpreter(self):
        path, arg = ansible_mitogen.parsing.parse_hashbang(
            self._inv.get_module_source()
        )
        if path is None:
            raise ansible.errors.AnsibleError(
                NO_INTERPRETER_MSG % (self._inv.module_name,)
            )

        fragment = self._rewrite_interpreter(path)
        if arg:
            fragment += " " + arg

        return fragment, path.startswith("python")

    def get_kwargs(self, **kwargs):
        interpreter_fragment, is_python = self._get_interpreter()
        return super(ScriptPlanner, self).get_kwargs(
            interpreter_fragment=interpreter_fragment, is_python=is_python, **kwargs
        )


class JsonArgsPlanner(ScriptPlanner):
    """
    Script that has its interpreter directive and the task arguments
    substituted into its source as a JSON string.
    """

    runner_name = "JsonArgsRunner"

    @classmethod
    def detect(cls, path, source):
        return module_common.REPLACER_JSONARGS in source


class WantJsonPlanner(ScriptPlanner):
    """
    If a module has the string WANT_JSON in it anywhere, Ansible treats it as a
    non-native module that accepts a filename as its only command line
    parameter. The filename is for a temporary file containing a JSON string
    containing the module's parameters. The module needs to open the file, read
    and parse the parameters, operate on the data, and print its return data as
    a JSON encoded dictionary to stdout before exiting.

    These types of modules are self-contained entities. As of Ansible 2.1,
    Ansible only modifies them to change a shebang line if present.
    """

    runner_name = "WantJsonRunner"

    @classmethod
    def detect(cls, path, source):
        return b"WANT_JSON" in source


class NewStylePlanner(ScriptPlanner):
    """
    The Ansiballz framework differs from module replacer in that it uses real
    Python imports of things in ansible/module_utils instead of merely
    preprocessing the module.
    """

    runner_name = "NewStyleRunner"
    marker = b"from ansible.module_utils."

    @classmethod
    def detect(cls, path, source):
        return cls.marker in source

    def _get_interpreter(self):
        return None, None

    def get_push_files(self):
        return super(NewStylePlanner, self).get_push_files() + [
            mitogen.core.to_text(path)
            for fullname, path, is_pkg in self.get_module_map()["custom"]
        ]

    def get_module_deps(self):
        return self.get_module_map()["builtin"]

    #: Module names appearing in this set always require forking, usually due
    #: to some terminal leakage that cannot be worked around in any sane
    #: manner.
    ALWAYS_FORK_MODULES = frozenset(
        [
            "dnf",  # issue #280; py-dnf/hawkey need therapy
            "firewalld",  # issue #570: ansible module_utils caches dbus conn
        ]
    )

    def should_fork(self):
        """
        In addition to asynchronous tasks, new-style modules should be forked
        if:

        * the user specifies mitogen_task_isolation=fork, or
        * the new-style module has a custom module search path, or
        * the module is known to leak like a sieve.
        """
        return (
            super(NewStylePlanner, self).should_fork()
            or (self._inv.task_vars.get("mitogen_task_isolation") == "fork")
            or (self._inv.module_name in self.ALWAYS_FORK_MODULES)
            or (len(self.get_module_map()["custom"]) > 0)
        )

    def get_search_path(self):
        return tuple(
            path
            for path in ansible_mitogen.loaders.module_utils_loader._get_paths(
                subdirs=False
            )
        )

    _module_map = None

    def get_module_map(self):
        if self._module_map is None:
            binding = self._inv.connection.get_binding()
            self._module_map = mitogen.service.call(
                call_context=binding.get_service_context(),
                service_name="ansible_mitogen.services.ModuleDepService",
                method_name="scan",
                module_name="ansible_module_%s" % (self._inv.module_name,),
                module_path=self._inv.module_path,
                search_path=self.get_search_path(),
                builtin_path=module_common._MODULE_UTILS_PATH,
                context=self._inv.connection.context,
            )
        return self._module_map

    def get_kwargs(self):
        return super(NewStylePlanner, self).get_kwargs(
            module_map=self.get_module_map(),
            py_module_name=py_modname_from_path(
                self._inv.module_name,
                self._inv.module_path,
            ),
        )


class ReplacerPlanner(NewStylePlanner):
    """
    The Module Replacer framework is the original framework implementing
    new-style modules. It is essentially a preprocessor (like the C
    Preprocessor for those familiar with that programming language). It does
    straight substitutions of specific substring patterns in the module file.
    There are two types of substitutions.

    * Replacements that only happen in the module file. These are public
      replacement strings that modules can utilize to get helpful boilerplate
      or access to arguments.

      "from ansible.module_utils.MOD_LIB_NAME import *" is replaced with the
      contents of the ansible/module_utils/MOD_LIB_NAME.py. These should only
      be used with new-style Python modules.

      "#<<INCLUDE_ANSIBLE_MODULE_COMMON>>" is equivalent to
      "from ansible.module_utils.basic import *" and should also only apply to
      new-style Python modules.

      "# POWERSHELL_COMMON" substitutes the contents of
      "ansible/module_utils/powershell.ps1". It should only be used with
      new-style Powershell modules.
    """

    runner_name = "ReplacerRunner"

    @classmethod
    def detect(cls, path, source):
        return module_common.REPLACER in source


class OldStylePlanner(ScriptPlanner):
    runner_name = "OldStyleRunner"

    @classmethod
    def detect(cls, path, source):
        # Everything else.
        return True


_planners = [
    BinaryPlanner,
    # ReplacerPlanner,
    NewStylePlanner,
    JsonArgsPlanner,
    WantJsonPlanner,
    OldStylePlanner,
]


try:
    _get_ansible_module_fqn = module_common._get_ansible_module_fqn
except AttributeError:
    _get_ansible_module_fqn = None


def py_modname_from_path(name, path):
    """
    Fetch the logical name of a new-style module as it might appear in
    :data:`sys.modules` of the target's Python interpreter.

    * Since Ansible 2.9, modules appearing within a package have the original
      package hierarchy approximated on the target, enabling relative imports
      to function correctly. For example, "ansible.modules.system.setup".
    """
    if _get_ansible_module_fqn:
        try:
            return _get_ansible_module_fqn(path)
        except ValueError:
            pass

    return "ansible.modules." + name


def read_file(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        bits = []
        chunk = True
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            bits.append(chunk)
    finally:
        os.close(fd)

    return mitogen.core.b("").join(bits)


def _propagate_deps(invocation, planner, context):
    binding = invocation.connection.get_binding()
    mitogen.service.call(
        call_context=binding.get_service_context(),
        service_name="mitogen.service.PushFileService",
        method_name="propagate_paths_and_modules",
        context=context,
        paths=planner.get_push_files(),
        # modules=planner.get_module_deps(), TODO
        overridden_sources=invocation._overridden_sources,
        # needs to be a list because can't unpickle() a set()
        extra_sys_paths=list(invocation._extra_sys_paths),
    )


def _invoke_async_task(invocation, planner):
    job_id = "%016x" % random.randint(0, 2 ** 64)
    context = invocation.connection.spawn_isolated_child()
    _propagate_deps(invocation, planner, context)

    with mitogen.core.Receiver(context.router) as started_recv:
        call_recv = context.call_async(
            ansible_mitogen.target.run_module_async,
            job_id=job_id,
            timeout_secs=invocation.timeout_secs,
            started_sender=started_recv.to_sender(),
            kwargs=planner.get_kwargs(),
        )

        # Wait for run_module_async() to crash, or for AsyncRunner to indicate
        # the job file has been written.
        for msg in mitogen.select.Select([started_recv, call_recv]):
            if msg.receiver is call_recv:
                # It can only be an exception.
                raise msg.unpickle()
            break

        return {
            "stdout": json.dumps(
                {
                    # modules/utilities/logic/async_wrapper.py::_run_module().
                    "changed": True,
                    "started": 1,
                    "finished": 0,
                    "ansible_job_id": job_id,
                }
            )
        }


def _invoke_isolated_task(invocation, planner):
    context = invocation.connection.spawn_isolated_child()
    _propagate_deps(invocation, planner, context)
    try:
        return context.call(
            ansible_mitogen.target.run_module,
            kwargs=planner.get_kwargs(),
        )
    finally:
        context.shutdown()


def _get_planner(name, path, source):
    for klass in _planners:
        if klass.detect(path, source):
            LOG.debug("%r accepted %r (filename %r)", klass, name, path)
            return klass
        LOG.debug("%r rejected %r", klass, name)
    raise ansible.errors.AnsibleError(NO_METHOD_MSG + repr(invocation))


def _fix_py35(invocation, module_source):
    """
    super edge case with a relative import error in Python 3.5.1-3.5.3
    in Ansible's setup module when using Mitogen
    https://github.com/dw/mitogen/issues/672#issuecomment-636408833
    We replace a relative import in the setup module with the actual full file path
    This works in vanilla Ansible but not in Mitogen otherwise
    """
    if (
        invocation.module_name
        in {"ansible.builtin.setup", "ansible.legacy.setup", "setup"}
        and invocation.module_path not in invocation._overridden_sources
    ):
        # in-memory replacement of setup module's relative import
        # would check for just python3.5 and run this then but we don't know the
        # target python at this time yet
        # NOTE: another ansible 2.10-specific fix: `from ..module_utils` used to be `from ...module_utils`
        module_source = module_source.replace(
            b"from ..module_utils.basic import AnsibleModule",
            b"from ansible.module_utils.basic import AnsibleModule",
        )
        invocation._overridden_sources[invocation.module_path] = module_source


def _load_collections(invocation):
    """
    Special loader that ensures that `ansible_collections` exist as a module path for import
    Goes through all collection path possibilities and stores paths to installed collections
    Stores them on the current invocation to later be passed to the master service
    """
    for collection_path in list_collection_dirs():
        invocation._extra_sys_paths.add(collection_path.decode("utf-8"))


def invoke(invocation):
    """
    Find a Planner subclass corresponding to `invocation` and use it to invoke
    the module.

    :param Invocation invocation:
    :returns:
        Module return dict.
    :raises ansible.errors.AnsibleError:
        Unrecognized/unsupported module type.
    """
    path = ansible_mitogen.loaders.module_loader.find_plugin(
        invocation.module_name,
        "",
    )
    if path is None:
        raise ansible.errors.AnsibleError(NO_MODULE_MSG % (invocation.module_name,))

    invocation.module_path = mitogen.core.to_text(path)
    if invocation.module_path not in _planner_by_path:
        if "ansible_collections" in invocation.module_path:
            _load_collections(invocation)

        module_source = invocation.get_module_source()
        _fix_py35(invocation, module_source)
        _planner_by_path[invocation.module_path] = _get_planner(
            invocation.module_name, invocation.module_path, module_source
        )

    planner = _planner_by_path[invocation.module_path](invocation)
    if invocation.wrap_async:
        response = _invoke_async_task(invocation, planner)
    elif planner.should_fork():
        response = _invoke_isolated_task(invocation, planner)
    else:
        _propagate_deps(invocation, planner, invocation.connection.context)
        response = invocation.connection.get_chain().call(
            ansible_mitogen.target.run_module,
            kwargs=planner.get_kwargs(),
        )

    return invocation.action._postprocess_response(response)
