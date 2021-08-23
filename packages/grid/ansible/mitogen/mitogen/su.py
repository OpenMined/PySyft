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

# stdlib
import logging
import re

# third party
import mitogen.core
import mitogen.parent

try:
    any
except NameError:
    # third party
    from mitogen.core import any


LOG = logging.getLogger(__name__)

password_incorrect_msg = "su password is incorrect"
password_required_msg = "su password is required"


class PasswordError(mitogen.core.StreamError):
    pass


class SetupBootstrapProtocol(mitogen.parent.BootstrapProtocol):
    password_sent = False

    def setup_patterns(self, conn):
        """
        su options cause the regexes used to vary. This is a mess, requires
        reworking.
        """
        incorrect_pattern = re.compile(
            mitogen.core.b("|").join(
                re.escape(s.encode("utf-8")) for s in conn.options.incorrect_prompts
            ),
            re.I,
        )
        prompt_pattern = re.compile(
            re.escape(conn.options.password_prompt.encode("utf-8")), re.I
        )

        self.PATTERNS = mitogen.parent.BootstrapProtocol.PATTERNS + [
            (incorrect_pattern, type(self)._on_password_incorrect),
        ]
        self.PARTIAL_PATTERNS = mitogen.parent.BootstrapProtocol.PARTIAL_PATTERNS + [
            (prompt_pattern, type(self)._on_password_prompt),
        ]

    def _on_password_prompt(self, line, match):
        LOG.debug(
            "%r: (password prompt): %r",
            self.stream.name,
            line.decode("utf-8", "replace"),
        )

        if self.stream.conn.options.password is None:
            self.stream.conn._fail_connection(PasswordError(password_required_msg))
            return

        if self.password_sent:
            self.stream.conn._fail_connection(PasswordError(password_incorrect_msg))
            return

        self.stream.transmit_side.write(
            (self.stream.conn.options.password + "\n").encode("utf-8")
        )
        self.password_sent = True

    def _on_password_incorrect(self, line, match):
        self.stream.conn._fail_connection(PasswordError(password_incorrect_msg))


class Options(mitogen.parent.Options):
    username = u"root"
    password = None
    su_path = "su"
    password_prompt = u"password:"
    incorrect_prompts = (
        u"su: sorry",  # BSD
        u"su: authentication failure",  # Linux
        u"su: incorrect password",  # CentOS 6
        u"authentication is denied",  # AIX
    )

    def __init__(
        self,
        username=None,
        password=None,
        su_path=None,
        password_prompt=None,
        incorrect_prompts=None,
        **kwargs
    ):
        super(Options, self).__init__(**kwargs)
        if username is not None:
            self.username = mitogen.core.to_text(username)
        if password is not None:
            self.password = mitogen.core.to_text(password)
        if su_path is not None:
            self.su_path = su_path
        if password_prompt is not None:
            self.password_prompt = password_prompt
        if incorrect_prompts is not None:
            self.incorrect_prompts = [
                mitogen.core.to_text(p) for p in incorrect_prompts
            ]


class Connection(mitogen.parent.Connection):
    options_class = Options
    stream_protocol_class = SetupBootstrapProtocol

    # TODO: BSD su cannot handle stdin being a socketpair, but it does let the
    # child inherit fds from the parent. So we can still pass a socketpair in
    # for hybrid_tty_create_child(), there just needs to be either a shell
    # snippet or bootstrap support for fixing things up afterwards.
    create_child = staticmethod(mitogen.parent.tty_create_child)
    child_is_immediate_subprocess = False

    def _get_name(self):
        return u"su." + self.options.username

    def stream_factory(self):
        stream = super(Connection, self).stream_factory()
        stream.protocol.setup_patterns(self)
        return stream

    def get_boot_command(self):
        argv = mitogen.parent.Argv(super(Connection, self).get_boot_command())
        return [self.options.su_path, self.options.username, "-c", str(argv)]
