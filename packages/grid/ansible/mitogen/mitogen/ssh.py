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
Construct new children via the OpenSSH client.
"""

# stdlib
import logging
import re

try:
    # stdlib
    from shlex import quote as shlex_quote
except ImportError:
    from pipes import quote as shlex_quote

# third party
from mitogen.core import b
import mitogen.parent

try:
    any
except NameError:
    # third party
    from mitogen.core import any


LOG = logging.getLogger(__name__)

auth_incorrect_msg = "SSH authentication is incorrect"
password_incorrect_msg = "SSH password is incorrect"
password_required_msg = "SSH password was requested, but none specified"
hostkey_config_msg = (
    "SSH requested permission to accept unknown host key, but "
    "check_host_keys=ignore. This is likely due to ssh_args=  "
    "conflicting with check_host_keys=. Please correct your "
    "configuration."
)
hostkey_failed_msg = (
    "Host key checking is enabled, and SSH reported an unrecognized or "
    "mismatching host key."
)

# sshpass uses 'assword' because it doesn't lowercase the input.
PASSWORD_PROMPT_PATTERN = re.compile(b("password"), re.I)

HOSTKEY_REQ_PATTERN = re.compile(
    b(
        r"are you sure you want to continue connecting "
        r"\(yes/no(?:/\[fingerprint\])?\)\?"
    ),
    re.I,
)

HOSTKEY_FAIL_PATTERN = re.compile(b(r"host key verification failed\."), re.I)

# [user@host: ] permission denied
# issue #271: work around conflict with user shell reporting 'permission
# denied' e.g. during chdir($HOME) by only matching it at the start of the
# line.
PERMDENIED_PATTERN = re.compile(
    b("^(?:[^@]+@[^:]+: )?" "Permission denied"), re.I  # Absent in OpenSSH <7.5
)

DEBUG_PATTERN = re.compile(b("^debug[123]:"))


class PasswordError(mitogen.core.StreamError):
    pass


class HostKeyError(mitogen.core.StreamError):
    pass


class SetupProtocol(mitogen.parent.RegexProtocol):
    """
    This protocol is attached to stderr of the SSH client. It responds to
    various interactive prompts as required.
    """

    password_sent = False

    def _on_host_key_request(self, line, match):
        if self.stream.conn.options.check_host_keys == "accept":
            LOG.debug("%s: accepting host key", self.stream.name)
            self.stream.transmit_side.write(b("yes\n"))
            return

        # _host_key_prompt() should never be reached with ignore or enforce
        # mode, SSH should have handled that. User's ssh_args= is conflicting
        # with ours.
        self.stream.conn._fail_connection(HostKeyError(hostkey_config_msg))

    def _on_host_key_failed(self, line, match):
        self.stream.conn._fail_connection(HostKeyError(hostkey_failed_msg))

    def _on_permission_denied(self, line, match):
        if self.stream.conn.options.password is not None and self.password_sent:
            self.stream.conn._fail_connection(PasswordError(password_incorrect_msg))
        elif (
            PASSWORD_PROMPT_PATTERN.search(line)
            and self.stream.conn.options.password is None
        ):
            # Permission denied (password,pubkey)
            self.stream.conn._fail_connection(PasswordError(password_required_msg))
        else:
            self.stream.conn._fail_connection(PasswordError(auth_incorrect_msg))

    def _on_password_prompt(self, line, match):
        LOG.debug("%s: (password prompt): %s", self.stream.name, line)
        if self.stream.conn.options.password is None:
            self.stream.conn._fail(PasswordError(password_required_msg))

        self.stream.transmit_side.write(
            (self.stream.conn.options.password + "\n").encode("utf-8")
        )
        self.password_sent = True

    def _on_debug_line(self, line, match):
        text = mitogen.core.to_text(line.rstrip())
        LOG.debug("%s: %s", self.stream.name, text)

    PATTERNS = [
        (DEBUG_PATTERN, _on_debug_line),
        (HOSTKEY_FAIL_PATTERN, _on_host_key_failed),
        (PERMDENIED_PATTERN, _on_permission_denied),
    ]

    PARTIAL_PATTERNS = [
        (PASSWORD_PROMPT_PATTERN, _on_password_prompt),
        (HOSTKEY_REQ_PATTERN, _on_host_key_request),
    ]


class Options(mitogen.parent.Options):
    #: Default to whatever is available as 'python' on the remote machine,
    #: overriding sys.executable use.
    python_path = "python"

    #: Number of -v invocations to pass on command line.
    ssh_debug_level = 0

    #: The path to the SSH binary.
    ssh_path = "ssh"

    hostname = None
    username = None
    port = None
    identity_file = None
    password = None
    ssh_args = None

    check_host_keys_msg = "check_host_keys= must be set to accept, enforce or ignore"

    def __init__(
        self,
        hostname,
        username=None,
        ssh_path=None,
        port=None,
        check_host_keys="enforce",
        password=None,
        identity_file=None,
        compression=True,
        ssh_args=None,
        keepalive_enabled=True,
        keepalive_count=3,
        keepalive_interval=15,
        identities_only=True,
        ssh_debug_level=None,
        **kwargs,
    ):
        super(Options, self).__init__(**kwargs)

        if check_host_keys not in ("accept", "enforce", "ignore"):
            raise ValueError(self.check_host_keys_msg)

        self.hostname = hostname
        self.username = username
        self.port = port
        self.check_host_keys = check_host_keys
        self.password = password
        self.identity_file = identity_file
        self.identities_only = identities_only
        self.compression = compression
        self.keepalive_enabled = keepalive_enabled
        self.keepalive_count = keepalive_count
        self.keepalive_interval = keepalive_interval
        if ssh_path:
            self.ssh_path = ssh_path
        if ssh_args:
            self.ssh_args = ssh_args
        if ssh_debug_level:
            self.ssh_debug_level = ssh_debug_level


class Connection(mitogen.parent.Connection):
    options_class = Options
    diag_protocol_class = SetupProtocol

    child_is_immediate_subprocess = False

    # strings that, if escaped, cause problems creating connections
    # example: `source /opt/rh/rh-python36/enable && python`
    # is an acceptable ansible_python_version but shlex would quote the &&
    # and prevent python from executing
    SHLEX_IGNORE = ["&&"]

    def _get_name(self):
        s = "ssh." + mitogen.core.to_text(self.options.hostname)
        if self.options.port and self.options.port != 22:
            s += f":{self.options.port}"
        return s

    def _requires_pty(self):
        """
        Return :data:`True` if a PTY to is required for this configuration,
        because it must interactively accept host keys or type a password.
        """
        return (
            self.options.check_host_keys == "accept"
            or self.options.password is not None
        )

    def create_child(self, **kwargs):
        """
        Avoid PTY use when possible to avoid a scaling limitation.
        """
        if self._requires_pty():
            return mitogen.parent.hybrid_tty_create_child(**kwargs)
        else:
            return mitogen.parent.create_child(stderr_pipe=True, **kwargs)

    def get_boot_command(self):
        bits = [self.options.ssh_path]
        if self.options.ssh_debug_level:
            bits += ["-" + ("v" * min(3, self.options.ssh_debug_level))]
        else:
            # issue #307: suppress any login banner, as it may contain the
            # password prompt, and there is no robust way to tell the
            # difference.
            bits += ["-o", "LogLevel ERROR"]
        if self.options.username:
            bits += ["-l", self.options.username]
        if self.options.port is not None:
            bits += ["-p", str(self.options.port)]
        if self.options.identities_only and (
            self.options.identity_file or self.options.password
        ):
            bits += ["-o", "IdentitiesOnly yes"]
        if self.options.identity_file:
            bits += ["-i", self.options.identity_file]
        if self.options.compression:
            bits += ["-o", "Compression yes"]
        if self.options.keepalive_enabled:
            bits += [
                "-o",
                f"ServerAliveInterval {self.options.keepalive_interval}",
                "-o",
                f"ServerAliveCountMax {self.options.keepalive_count}",
            ]
        if not self._requires_pty():
            bits += ["-o", "BatchMode yes"]
        if self.options.check_host_keys == "enforce":
            bits += ["-o", "StrictHostKeyChecking yes"]
        if self.options.check_host_keys == "accept":
            bits += ["-o", "StrictHostKeyChecking ask"]
        elif self.options.check_host_keys == "ignore":
            bits += [
                "-o",
                "StrictHostKeyChecking no",
                "-o",
                "UserKnownHostsFile /dev/null",
                "-o",
                "GlobalKnownHostsFile /dev/null",
            ]
        if self.options.ssh_args:
            bits += self.options.ssh_args
        bits.append(self.options.hostname)
        base = super(Connection, self).get_boot_command()

        base_parts = []
        for s in base:
            val = s if s in self.SHLEX_IGNORE else shlex_quote(s).strip()
            base_parts.append(val)
        return bits + base_parts
