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
from __future__ import unicode_literals

# stdlib
import errno
import logging
import os
import pprint
import stat
import sys
import time

# third party
import ansible.constants as C
import ansible.errors
import ansible.plugins.connection
import ansible.utils.shlex
import ansible_mitogen.mixins
import ansible_mitogen.parsing
import ansible_mitogen.process
import ansible_mitogen.services
import ansible_mitogen.target
import ansible_mitogen.transport_config
import mitogen.core
import mitogen.fork
import mitogen.utils

LOG = logging.getLogger(__name__)

task_vars_msg = (
    "could not recover task_vars. This means some connection "
    "settings may erroneously be reset to their defaults. "
    "Please report a bug if you encounter this message."
)


def get_remote_name(spec):
    """
    Return the value to use for the "remote_name" parameter.
    """
    if spec.mitogen_mask_remote_name():
        return "ansible"
    return None


def optional_int(value):
    """
    Convert `value` to an integer if it is not :data:`None`, otherwise return
    :data:`None`.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def convert_bool(obj):
    if isinstance(obj, bool):
        return obj
    if str(obj).lower() in ("no", "false", "0"):
        return False
    if str(obj).lower() not in ("yes", "true", "1"):
        raise ansible.errors.AnsibleConnectionFailure(
            f"expected yes/no/true/false/0/1, got {obj!r}"
        )
    return True


def default(value, default):
    """
    Return `default` is `value` is :data:`None`, otherwise return `value`.
    """
    if value is None:
        return default
    return value


def _connect_local(spec):
    """
    Return ContextService arguments for a local connection.
    """
    return {
        "method": "local",
        "kwargs": {
            "python_path": spec.python_path(),
        },
    }


def _connect_ssh(spec):
    """
    Return ContextService arguments for an SSH connection.
    """
    if C.HOST_KEY_CHECKING:
        check_host_keys = "enforce"
    else:
        check_host_keys = "ignore"

    # #334: tilde-expand private_key_file to avoid implementation difference
    # between Python and OpenSSH.
    private_key_file = spec.private_key_file()
    if private_key_file is not None:
        private_key_file = os.path.expanduser(private_key_file)

    return {
        "method": "ssh",
        "kwargs": {
            "check_host_keys": check_host_keys,
            "hostname": spec.remote_addr(),
            "username": spec.remote_user(),
            "compression": convert_bool(default(spec.mitogen_ssh_compression(), True)),
            "password": spec.password(),
            "port": spec.port(),
            "python_path": spec.python_path(),
            "identity_file": private_key_file,
            "identities_only": False,
            "ssh_path": spec.ssh_executable(),
            "connect_timeout": spec.ansible_ssh_timeout(),
            "ssh_args": spec.ssh_args(),
            "ssh_debug_level": spec.mitogen_ssh_debug_level(),
            "remote_name": get_remote_name(spec),
            "keepalive_count": (spec.mitogen_ssh_keepalive_count() or 10),
            "keepalive_interval": (spec.mitogen_ssh_keepalive_interval() or 30),
        },
    }


def _connect_buildah(spec):
    """
    Return ContextService arguments for a Buildah connection.
    """
    return {
        "method": "buildah",
        "kwargs": {
            "username": spec.remote_user(),
            "container": spec.remote_addr(),
            "python_path": spec.python_path(),
            "connect_timeout": spec.ansible_ssh_timeout() or spec.timeout(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_docker(spec):
    """
    Return ContextService arguments for a Docker connection.
    """
    return {
        "method": "docker",
        "kwargs": {
            "username": spec.remote_user(),
            "container": spec.remote_addr(),
            "python_path": spec.python_path(rediscover_python=True),
            "connect_timeout": spec.ansible_ssh_timeout() or spec.timeout(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_kubectl(spec):
    """
    Return ContextService arguments for a Kubernetes connection.
    """
    return {
        "method": "kubectl",
        "kwargs": {
            "pod": spec.remote_addr(),
            "python_path": spec.python_path(),
            "connect_timeout": spec.ansible_ssh_timeout() or spec.timeout(),
            "kubectl_path": spec.mitogen_kubectl_path(),
            "kubectl_args": spec.extra_args(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_jail(spec):
    """
    Return ContextService arguments for a FreeBSD jail connection.
    """
    return {
        "method": "jail",
        "kwargs": {
            "username": spec.remote_user(),
            "container": spec.remote_addr(),
            "python_path": spec.python_path(),
            "connect_timeout": spec.ansible_ssh_timeout() or spec.timeout(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_lxc(spec):
    """
    Return ContextService arguments for an LXC Classic container connection.
    """
    return {
        "method": "lxc",
        "kwargs": {
            "container": spec.remote_addr(),
            "python_path": spec.python_path(),
            "lxc_attach_path": spec.mitogen_lxc_attach_path(),
            "connect_timeout": spec.ansible_ssh_timeout() or spec.timeout(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_lxd(spec):
    """
    Return ContextService arguments for an LXD container connection.
    """
    return {
        "method": "lxd",
        "kwargs": {
            "container": spec.remote_addr(),
            "python_path": spec.python_path(),
            "lxc_path": spec.mitogen_lxc_path(),
            "connect_timeout": spec.ansible_ssh_timeout() or spec.timeout(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_machinectl(spec):
    """
    Return ContextService arguments for a machinectl connection.
    """
    return _connect_setns(spec, kind="machinectl")


def _connect_setns(spec, kind=None):
    """
    Return ContextService arguments for a mitogen_setns connection.
    """
    return {
        "method": "setns",
        "kwargs": {
            "container": spec.remote_addr(),
            "username": spec.remote_user(),
            "python_path": spec.python_path(),
            "kind": kind or spec.mitogen_kind(),
            "docker_path": spec.mitogen_docker_path(),
            "lxc_path": spec.mitogen_lxc_path(),
            "lxc_info_path": spec.mitogen_lxc_info_path(),
            "machinectl_path": spec.mitogen_machinectl_path(),
        },
    }


def _connect_su(spec):
    """
    Return ContextService arguments for su as a become method.
    """
    return {
        "method": "su",
        "enable_lru": True,
        "kwargs": {
            "username": spec.become_user(),
            "password": spec.become_pass(),
            "python_path": spec.python_path(),
            "su_path": spec.become_exe(),
            "connect_timeout": spec.timeout(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_sudo(spec):
    """
    Return ContextService arguments for sudo as a become method.
    """
    return {
        "method": "sudo",
        "enable_lru": True,
        "kwargs": {
            "username": spec.become_user(),
            "password": spec.become_pass(),
            "python_path": spec.python_path(),
            "sudo_path": spec.become_exe(),
            "connect_timeout": spec.timeout(),
            "sudo_args": spec.sudo_args(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_doas(spec):
    """
    Return ContextService arguments for doas as a become method.
    """
    return {
        "method": "doas",
        "enable_lru": True,
        "kwargs": {
            "username": spec.become_user(),
            "password": spec.become_pass(),
            "python_path": spec.python_path(),
            "doas_path": spec.become_exe(),
            "connect_timeout": spec.timeout(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_mitogen_su(spec):
    """
    Return ContextService arguments for su as a first class connection.
    """
    return {
        "method": "su",
        "kwargs": {
            "username": spec.remote_user(),
            "password": spec.password(),
            "python_path": spec.python_path(),
            "su_path": spec.become_exe(),
            "connect_timeout": spec.timeout(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_mitogen_sudo(spec):
    """
    Return ContextService arguments for sudo as a first class connection.
    """
    return {
        "method": "sudo",
        "kwargs": {
            "username": spec.remote_user(),
            "password": spec.password(),
            "python_path": spec.python_path(),
            "sudo_path": spec.become_exe(),
            "connect_timeout": spec.timeout(),
            "sudo_args": spec.sudo_args(),
            "remote_name": get_remote_name(spec),
        },
    }


def _connect_mitogen_doas(spec):
    """
    Return ContextService arguments for doas as a first class connection.
    """
    return {
        "method": "doas",
        "kwargs": {
            "username": spec.remote_user(),
            "password": spec.password(),
            "python_path": spec.python_path(),
            "doas_path": spec.ansible_doas_exe(),
            "connect_timeout": spec.timeout(),
            "remote_name": get_remote_name(spec),
        },
    }


#: Mapping of connection method names to functions invoked as `func(spec)`
#: generating ContextService keyword arguments matching a connection
#: specification.
CONNECTION_METHOD = {
    "buildah": _connect_buildah,
    "docker": _connect_docker,
    "kubectl": _connect_kubectl,
    "jail": _connect_jail,
    "local": _connect_local,
    "lxc": _connect_lxc,
    "lxd": _connect_lxd,
    "machinectl": _connect_machinectl,
    "setns": _connect_setns,
    "ssh": _connect_ssh,
    "smart": _connect_ssh,  # issue #548.
    "su": _connect_su,
    "sudo": _connect_sudo,
    "doas": _connect_doas,
    "mitogen_su": _connect_mitogen_su,
    "mitogen_sudo": _connect_mitogen_sudo,
    "mitogen_doas": _connect_mitogen_doas,
}


class CallChain(mitogen.parent.CallChain):
    """
    Extend :class:`mitogen.parent.CallChain` to additionally cause the
    associated :class:`Connection` to be reset if a ChannelError occurs.

    This only catches failures that occur while a call is pending, it is a
    stop-gap until a more general method is available to notice connection in
    every situation.
    """

    call_aborted_msg = (
        "Mitogen was disconnected from the remote environment while a call "
        "was in-progress. If you feel this is in error, please file a bug. "
        "Original error was: %s"
    )

    def __init__(self, connection, context, pipelined=False):
        super(CallChain, self).__init__(context, pipelined)
        #: The connection to reset on CallError.
        self._connection = connection

    def _rethrow(self, recv):
        try:
            return recv.get().unpickle()
        except mitogen.core.ChannelError as e:
            self._connection.reset()
            raise ansible.errors.AnsibleConnectionFailure(self.call_aborted_msg % (e,))

    def call(self, func, *args, **kwargs):
        """
        Like :meth:`mitogen.parent.CallChain.call`, but log timings.
        """
        t0 = time.time()
        try:
            recv = self.call_async(func, *args, **kwargs)
            return self._rethrow(recv)
        finally:
            LOG.debug(
                "Call took %d ms: %r",
                1000 * (time.time() - t0),
                mitogen.parent.CallSpec(func, args, kwargs),
            )


class Connection(ansible.plugins.connection.ConnectionBase):
    #: The :class:`ansible_mitogen.process.Binding` representing the connection
    #: multiplexer this connection's target is assigned to. :data:`None` when
    #: disconnected.
    binding = None

    #: mitogen.parent.Context for the target account on the target, possibly
    #: reached via become.
    context = None

    #: Context for the login account on the target. This is always the login
    #: account, even when become=True.
    login_context = None

    #: Only sudo, su, and doas are supported for now.
    become_methods = ["sudo", "su", "doas"]

    #: Dict containing init_child() return value as recorded at startup by
    #: ContextService. Contains:
    #:
    #:  fork_context:   Context connected to the fork parent : process in the
    #:                  target account.
    #:  home_dir:       Target context's home directory.
    #:  good_temp_dir:  A writeable directory where new temporary directories
    #:                  can be created.
    init_child_result = None

    #: A :class:`mitogen.parent.CallChain` for calls made to the target
    #: account, to ensure subsequent calls fail with the original exception if
    #: pipelined directory creation or file transfer fails.
    chain = None

    #
    # Note: any of the attributes below may be :data:`None` if the connection
    # plugin was constructed directly by a non-cooperative action, such as in
    # the case of the synchronize module.
    #

    #: Set to task_vars by on_action_run().
    _task_vars = None

    #: Set by on_action_run()
    delegate_to_hostname = None

    #: Set to '_loader.get_basedir()' by on_action_run(). Used by mitogen_local
    #: to change the working directory to that of the current playbook,
    #: matching vanilla Ansible behaviour.
    loader_basedir = None

    # set by `_get_task_vars()` for interpreter discovery
    _action = None

    def __del__(self):
        """
        Ansible cannot be trusted to always call close() e.g. the synchronize
        action constructs a local connection like this. So provide a destructor
        in the hopes of catching these cases.
        """
        # https://github.com/dw/mitogen/issues/140
        self.close()

    def on_action_run(self, task_vars, delegate_to_hostname, loader_basedir):
        """
        Invoked by ActionModuleMixin to indicate a new task is about to start
        executing. We use the opportunity to grab relevant bits from the
        task-specific data.

        :param dict task_vars:
            Task variable dictionary.
        :param str delegate_to_hostname:
            :data:`None`, or the template-expanded inventory hostname this task
            is being delegated to. A similar variable exists on PlayContext
            when ``delegate_to:`` is active, however it is unexpanded.
        :param str loader_basedir:
            Loader base directory; see :attr:`loader_basedir`.
        """
        self._task_vars = task_vars
        self.delegate_to_hostname = delegate_to_hostname
        self.loader_basedir = loader_basedir
        self._put_connection()

    def _get_task_vars(self):
        """
        More information is needed than normally provided to an Ansible
        connection.  For proxied connections, intermediary configuration must
        be inferred, and for any connection the configured Python interpreter
        must be known.

        There is no clean way to access this information that would not deviate
        from the running Ansible version. The least invasive method known is to
        reuse the running task's task_vars dict.

        This method walks the stack to find task_vars of the Action plugin's
        run(), or if no Action is present, from Strategy's _execute_meta(), as
        in the case of 'meta: reset_connection'. The stack is walked in
        addition to subclassing Action.run()/on_action_run(), as it is possible
        for new connections to be constructed in addition to the preconstructed
        connection passed into any running action.
        """
        if self._task_vars is not None:
            # check for if self._action has already been set or not
            # there are some cases where the ansible executor passes in task_vars
            # so we don't walk the stack to find them
            # TODO: is there a better way to get the ActionModuleMixin object?
            # ansible python discovery needs it to run discover_interpreter()
            if not isinstance(self._action, ansible_mitogen.mixins.ActionModuleMixin):
                f = sys._getframe()
                while f:
                    if f.f_code.co_name == "run":
                        f_self = f.f_locals.get("self")
                        if isinstance(f_self, ansible_mitogen.mixins.ActionModuleMixin):
                            self._action = f_self
                            break
                    elif f.f_code.co_name == "_execute_meta":
                        break
                    f = f.f_back

            return self._task_vars

        f = sys._getframe()
        while f:
            if f.f_code.co_name == "run":
                f_locals = f.f_locals
                f_self = f_locals.get("self")
                if isinstance(f_self, ansible_mitogen.mixins.ActionModuleMixin):
                    # backref for python interpreter discovery, should be safe because _get_task_vars
                    # is always called before running interpreter discovery
                    self._action = f_self
                    task_vars = f_locals.get("task_vars")
                    if task_vars:
                        LOG.debug("recovered task_vars from Action")
                        return task_vars
            elif f.f_code.co_name == "_execute_meta":
                f_all_vars = f.f_locals.get("all_vars")
                if isinstance(f_all_vars, dict):
                    LOG.debug("recovered task_vars from meta:")
                    return f_all_vars

            f = f.f_back

        raise ansible.errors.AnsibleConnectionFailure(task_vars_msg)

    def get_host_vars(self, inventory_hostname):
        """
        Fetch the HostVars for a host.

        :returns:
            Variables dictionary or :data:`None`.
        :raises ansible.errors.AnsibleConnectionFailure:
            Task vars unavailable.
        """
        task_vars = self._get_task_vars()
        hostvars = task_vars.get("hostvars")
        if hostvars:
            return hostvars.get(inventory_hostname)

        raise ansible.errors.AnsibleConnectionFailure(task_vars_msg)

    def get_task_var(self, key, default=None):
        """
        Fetch the value of a task variable related to connection configuration,
        or, if delegate_to is active, fetch the same variable via HostVars for
        the delegated-to machine.

        When running with delegate_to, Ansible tasks have variables associated
        with the original machine, not the delegated-to machine, therefore it
        does not make sense to extract connection-related configuration for the
        delegated-to machine from them.
        """

        def _fetch_task_var(task_vars, key):
            """
            Special helper func in case vars can be templated
            """
            SPECIAL_TASK_VARS = ["ansible_python_interpreter"]
            if key in task_vars:
                val = task_vars[key]
                if "{" in str(val) and key in SPECIAL_TASK_VARS:
                    # template every time rather than storing in a cache
                    # in case a different template value is used in a different task
                    val = self.templar.template(
                        val, preserve_trailing_newlines=True, escape_backslashes=False
                    )
                return val

        task_vars = self._get_task_vars()
        if self.delegate_to_hostname is None:
            return _fetch_task_var(task_vars, key)
        else:
            delegated_vars = task_vars["ansible_delegated_vars"]
            if self.delegate_to_hostname in delegated_vars:
                task_vars = delegated_vars[self.delegate_to_hostname]
                return _fetch_task_var(task_vars, key)

        return default

    @property
    def homedir(self):
        self._connect()
        return self.init_child_result["home_dir"]

    def get_binding(self):
        """
        Return the :class:`ansible_mitogen.process.Binding` representing the
        process that hosts the physical connection and services (context
        establishment, file transfer, ..) for our desired target.
        """
        assert self.binding is not None
        return self.binding

    @property
    def connected(self):
        return self.context is not None

    def _spec_from_via(self, proxied_inventory_name, via_spec):
        """
        Produce a dict connection specifiction given a string `via_spec`, of
        the form `[[become_method:]become_user@]inventory_hostname`.
        """
        become_user, _, inventory_name = via_spec.rpartition("@")
        become_method, _, become_user = become_user.rpartition(":")

        # must use __contains__ to avoid a TypeError for a missing host on
        # Ansible 2.3.
        via_vars = self.get_host_vars(inventory_name)
        if via_vars is None:
            raise ansible.errors.AnsibleConnectionFailure(
                self.unknown_via_msg
                % (
                    via_spec,
                    proxied_inventory_name,
                )
            )

        return ansible_mitogen.transport_config.MitogenViaSpec(
            inventory_name=inventory_name,
            play_context=self._play_context,
            host_vars=dict(via_vars),  # TODO: make it lazy
            task_vars=self._get_task_vars(),  # needed for interpreter discovery in parse_python_path
            action=self._action,
            become_method=become_method or None,
            become_user=become_user or None,
        )

    unknown_via_msg = "mitogen_via=%s of %s specifies an unknown hostname"
    via_cycle_msg = "mitogen_via=%s of %s creates a cycle (%s)"

    def _stack_from_spec(self, spec, stack=(), seen_names=()):
        """
        Return a tuple of ContextService parameter dictionaries corresponding
        to the connection described by `spec`, and any connection referenced by
        its `mitogen_via` or `become` fields. Each element is a dict of the
        form::

            {
                # Optional. If present and `True`, this hop is elegible for
                # interpreter recycling.
                "enable_lru": True,
                # mitogen.master.Router method name.
                "method": "ssh",
                # mitogen.master.Router method kwargs.
                "kwargs": {
                    "hostname": "..."
                }
            }

        :param ansible_mitogen.transport_config.Spec spec:
            Connection specification.
        :param tuple stack:
            Stack elements from parent call (used for recursion).
        :param tuple seen_names:
            Inventory hostnames from parent call (cycle detection).
        :returns:
            Tuple `(stack, seen_names)`.
        """
        if spec.inventory_name() in seen_names:
            raise ansible.errors.AnsibleConnectionFailure(
                self.via_cycle_msg
                % (
                    spec.mitogen_via(),
                    spec.inventory_name(),
                    " -> ".join(reversed(seen_names + (spec.inventory_name(),))),
                )
            )

        if spec.mitogen_via():
            stack = self._stack_from_spec(
                self._spec_from_via(spec.inventory_name(), spec.mitogen_via()),
                stack=stack,
                seen_names=seen_names + (spec.inventory_name(),),
            )

        stack += (CONNECTION_METHOD[spec.transport()](spec),)
        if spec.become() and (
            (spec.become_user() != spec.remote_user()) or C.BECOME_ALLOW_SAME_USER
        ):
            stack += (CONNECTION_METHOD[spec.become_method()](spec),)

        return stack

    def _build_stack(self):
        """
        Construct a list of dictionaries representing the connection
        configuration between the controller and the target. This is
        additionally used by the integration tests "mitogen_get_stack" action
        to fetch the would-be connection configuration.
        """
        spec = ansible_mitogen.transport_config.PlayContextSpec(
            connection=self,
            play_context=self._play_context,
            transport=self.transport,
            inventory_name=self.get_task_var("inventory_hostname"),
        )
        stack = self._stack_from_spec(spec)
        return spec.inventory_name(), stack

    def _connect_stack(self, stack):
        """
        Pass `stack` to ContextService, requesting a copy of the context object
        representing the last tuple element. If no connection exists yet,
        ContextService will recursively establish it before returning it or
        throwing an error.

        See :meth:`ansible_mitogen.services.ContextService.get` docstring for
        description of the returned dictionary.
        """
        try:
            dct = mitogen.service.call(
                call_context=self.binding.get_service_context(),
                service_name="ansible_mitogen.services.ContextService",
                method_name="get",
                stack=mitogen.utils.cast(list(stack)),
            )
        except mitogen.core.CallError:
            LOG.warning(
                "Connection failed; stack configuration was:\n%s", pprint.pformat(stack)
            )
            raise

        if dct["msg"]:
            if dct["method_name"] in self.become_methods:
                raise ansible.errors.AnsibleModuleError(dct["msg"])
            raise ansible.errors.AnsibleConnectionFailure(dct["msg"])

        self.context = dct["context"]
        self.chain = CallChain(self, self.context, pipelined=True)
        if self._play_context.become:
            self.login_context = dct["via"]
        else:
            self.login_context = self.context

        self.init_child_result = dct["init_child_result"]

    def get_good_temp_dir(self):
        """
        Return the 'good temporary directory' as discovered by
        :func:`ansible_mitogen.target.init_child` immediately after
        ContextService constructed the target context.
        """
        self._connect()
        return self.init_child_result["good_temp_dir"]

    def _connect(self):
        """
        Establish a connection to the master process's UNIX listener socket,
        constructing a mitogen.master.Router to communicate with the master,
        and a mitogen.parent.Context to represent it.

        Depending on the original transport we should emulate, trigger one of
        the _connect_*() service calls defined above to cause the master
        process to establish the real connection on our behalf, or return a
        reference to the existing one.
        """
        if self.connected:
            return

        inventory_name, stack = self._build_stack()
        worker_model = ansible_mitogen.process.get_worker_model()
        self.binding = worker_model.get_binding(mitogen.utils.cast(inventory_name))
        self._connect_stack(stack)

    def _put_connection(self):
        """
        Forget everything we know about the connected context. This function
        cannot be called _reset() since that name is used as a public API by
        Ansible 2.4 wait_for_connection plug-in.
        """
        if not self.context:
            return

        self.chain.reset()
        mitogen.service.call(
            call_context=self.binding.get_service_context(),
            service_name="ansible_mitogen.services.ContextService",
            method_name="put",
            context=self.context,
        )

        self.context = None
        self.login_context = None
        self.init_child_result = None
        self.chain = None

    def close(self):
        """
        Arrange for the mitogen.master.Router running in the worker to
        gracefully shut down, and wait for shutdown to complete. Safe to call
        multiple times.
        """
        self._put_connection()
        if self.binding:
            self.binding.close()
            self.binding = None

    reset_compat_msg = (
        'Mitogen only supports "reset_connection" on Ansible 2.5.6 or later'
    )

    def reset(self):
        """
        Explicitly terminate the connection to the remote host. This discards
        any local state we hold for the connection, returns the Connection to
        the 'disconnected' state, and informs ContextService the connection is
        bad somehow, and should be shut down and discarded.
        """
        if self._play_context.remote_addr is None:
            # <2.5.6 incorrectly populate PlayContext for reset_connection
            # https://github.com/ansible/ansible/issues/27520
            raise ansible.errors.AnsibleConnectionFailure(self.reset_compat_msg)

        # Strategy's _execute_meta doesn't have an action obj but we'll need one for
        # running interpreter_discovery
        # will create a new temporary action obj for this purpose
        self._action = ansible_mitogen.mixins.ActionModuleMixin(
            task=0,
            connection=self,
            play_context=self._play_context,
            loader=0,
            templar=0,
            shared_loader_obj=0,
        )

        # Clear out state in case we were ever connected.
        self.close()

        inventory_name, stack = self._build_stack()
        if self._play_context.become:
            stack = stack[:-1]

        worker_model = ansible_mitogen.process.get_worker_model()
        binding = worker_model.get_binding(inventory_name)
        try:
            mitogen.service.call(
                call_context=binding.get_service_context(),
                service_name="ansible_mitogen.services.ContextService",
                method_name="reset",
                stack=mitogen.utils.cast(list(stack)),
            )
        finally:
            binding.close()

    # Compatibility with Ansible 2.4 wait_for_connection plug-in.
    _reset = reset

    def get_chain(self, use_login=False, use_fork=False):
        """
        Return the :class:`mitogen.parent.CallChain` to use for executing
        function calls.

        :param bool use_login:
            If :data:`True`, always return the chain for the login account
            rather than any active become user.
        :param bool use_fork:
            If :data:`True`, return the chain for the fork parent.
        :returns mitogen.parent.CallChain:
        """
        self._connect()
        if use_login:
            return self.login_context.default_call_chain
        # See FORK_SUPPORTED comments in target.py.
        if use_fork and self.init_child_result["fork_context"] is not None:
            return self.init_child_result["fork_context"].default_call_chain
        return self.chain

    def spawn_isolated_child(self):
        """
        Fork or launch a new child off the target context.

        :returns:
            mitogen.core.Context of the new child.
        """
        return self.get_chain(use_fork=True).call(
            ansible_mitogen.target.spawn_isolated_child
        )

    def get_extra_args(self):
        """
        Overridden by connections/mitogen_kubectl.py to a list of additional
        arguments for the command.
        """
        # TODO: maybe use this for SSH too.
        return []

    def get_default_cwd(self):
        """
        Overridden by connections/mitogen_local.py to emulate behaviour of CWD
        being fixed to that of ActionBase._loader.get_basedir().
        """
        return None

    def get_default_env(self):
        """
        Overridden by connections/mitogen_local.py to emulate behaviour of
        WorkProcess environment inherited from WorkerProcess.
        """
        return None

    def exec_command(self, cmd, in_data="", sudoable=True, mitogen_chdir=None):
        """
        Implement exec_command() by calling the corresponding
        ansible_mitogen.target function in the target.

        :param str cmd:
            Shell command to execute.
        :param bytes in_data:
            Data to supply on ``stdin`` of the process.
        :returns:
            (return code, stdout bytes, stderr bytes)
        """
        emulate_tty = not in_data and sudoable
        rc, stdout, stderr = self.get_chain().call(
            ansible_mitogen.target.exec_command,
            cmd=mitogen.utils.cast(cmd),
            in_data=mitogen.utils.cast(in_data),
            chdir=mitogen_chdir or self.get_default_cwd(),
            emulate_tty=emulate_tty,
        )

        stderr += b"Shared connection to %s closed.%s" % (
            self._play_context.remote_addr.encode(),
            (b"\r\n" if emulate_tty else b"\n"),
        )
        return rc, stdout, stderr

    def fetch_file(self, in_path, out_path):
        """
        Implement fetch_file() by calling the corresponding
        ansible_mitogen.target function in the target.

        :param str in_path:
            Remote filesystem path to read.
        :param str out_path:
            Local filesystem path to write.
        """
        self._connect()
        ansible_mitogen.target.transfer_file(
            context=self.context,
            # in_path may be AnsibleUnicode
            in_path=mitogen.utils.cast(in_path),
            out_path=out_path,
        )

    def put_data(self, out_path, data, mode=None, utimes=None):
        """
        Implement put_file() by caling the corresponding ansible_mitogen.target
        function in the target, transferring small files inline. This is
        pipelined and will return immediately; failed transfers are reported as
        exceptions in subsequent functon calls.

        :param str out_path:
            Remote filesystem path to write.
        :param byte data:
            File contents to put.
        """
        self.get_chain().call_no_reply(
            ansible_mitogen.target.write_path,
            mitogen.utils.cast(out_path),
            mitogen.core.Blob(data),
            mode=mode,
            utimes=utimes,
        )

    #: Maximum size of a small file before switching to streaming
    #: transfer. This should really be the same as
    #: mitogen.services.FileService.IO_SIZE, however the message format has
    #: slightly more overhead, so just randomly subtract 4KiB.
    SMALL_FILE_LIMIT = mitogen.core.CHUNK_SIZE - 4096

    def _throw_io_error(self, e, path):
        if e.args[0] == errno.ENOENT:
            s = "file or module does not exist: " + path
            raise ansible.errors.AnsibleFileNotFound(s)

    def put_file(self, in_path, out_path):
        """
        Implement put_file() by streamily transferring the file via
        FileService.

        :param str in_path:
            Local filesystem path to read.
        :param str out_path:
            Remote filesystem path to write.
        """
        try:
            st = os.stat(in_path)
        except OSError as e:
            self._throw_io_error(e, in_path)
            raise

        if not stat.S_ISREG(st.st_mode):
            raise IOError(f"{in_path!r} is not a regular file.")

        # If the file is sufficiently small, just ship it in the argument list
        # rather than introducing an extra RTT for the child to request it from
        # FileService.
        if st.st_size <= self.SMALL_FILE_LIMIT:
            try:
                fp = open(in_path, "rb")
                try:
                    s = fp.read(self.SMALL_FILE_LIMIT + 1)
                finally:
                    fp.close()
            except OSError:
                self._throw_io_error(e, in_path)
                raise

            # Ensure did not grow during read.
            if len(s) == st.st_size:
                return self.put_data(
                    out_path, s, mode=st.st_mode, utimes=(st.st_atime, st.st_mtime)
                )

        self._connect()
        mitogen.service.call(
            call_context=self.binding.get_service_context(),
            service_name="mitogen.service.FileService",
            method_name="register",
            path=mitogen.utils.cast(in_path),
        )

        # For now this must remain synchronous, as the action plug-in may have
        # passed us a temporary file to transfer. A future FileService could
        # maintain an LRU list of open file descriptors to keep the temporary
        # file alive, but that requires more work.
        self.get_chain().call(
            ansible_mitogen.target.transfer_file,
            context=self.binding.get_child_service_context(),
            in_path=in_path,
            out_path=out_path,
        )
