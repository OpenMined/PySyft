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

LOG = logging.getLogger(__name__)

password_incorrect_msg = "doas password is incorrect"
password_required_msg = "doas password is required"


class PasswordError(mitogen.core.StreamError):
    pass


class Options(mitogen.parent.Options):
    username = u"root"
    password = None
    doas_path = "doas"
    password_prompt = u"Password:"
    incorrect_prompts = (
        u"doas: authentication failed",  # slicer69/doas
        u"doas: Authorization failed",  # openbsd/src
    )

    def __init__(
        self,
        username=None,
        password=None,
        doas_path=None,
        password_prompt=None,
        incorrect_prompts=None,
        **kwargs
    ):
        super(Options, self).__init__(**kwargs)
        if username is not None:
            self.username = mitogen.core.to_text(username)
        if password is not None:
            self.password = mitogen.core.to_text(password)
        if doas_path is not None:
            self.doas_path = doas_path
        if password_prompt is not None:
            self.password_prompt = mitogen.core.to_text(password_prompt)
        if incorrect_prompts is not None:
            self.incorrect_prompts = [
                mitogen.core.to_text(p) for p in incorrect_prompts
            ]


class BootstrapProtocol(mitogen.parent.RegexProtocol):
    password_sent = False

    def setup_patterns(self, conn):
        prompt_pattern = re.compile(
            re.escape(conn.options.password_prompt).encode("utf-8"), re.I
        )
        incorrect_prompt_pattern = re.compile(
            u"|".join(re.escape(s) for s in conn.options.incorrect_prompts).encode(
                "utf-8"
            ),
            re.I,
        )

        self.PATTERNS = [
            (incorrect_prompt_pattern, type(self)._on_incorrect_password),
        ]
        self.PARTIAL_PATTERNS = [
            (prompt_pattern, type(self)._on_password_prompt),
        ]

    def _on_incorrect_password(self, line, match):
        if self.password_sent:
            self.stream.conn._fail_connection(PasswordError(password_incorrect_msg))

    def _on_password_prompt(self, line, match):
        if self.stream.conn.options.password is None:
            self.stream.conn._fail_connection(PasswordError(password_required_msg))
            return

        if self.password_sent:
            self.stream.conn._fail_connection(PasswordError(password_incorrect_msg))
            return

        LOG.debug("sending password")
        self.stream.transmit_side.write(
            (self.stream.conn.options.password + "\n").encode("utf-8")
        )
        self.password_sent = True


class Connection(mitogen.parent.Connection):
    options_class = Options
    diag_protocol_class = BootstrapProtocol

    create_child = staticmethod(mitogen.parent.hybrid_tty_create_child)
    child_is_immediate_subprocess = False

    def _get_name(self):
        return u"doas." + self.options.username

    def stderr_stream_factory(self):
        stream = super(Connection, self).stderr_stream_factory()
        stream.protocol.setup_patterns(self)
        return stream

    def get_boot_command(self):
        bits = [self.options.doas_path, "-u", self.options.username, "--"]
        return bits + super(Connection, self).get_boot_command()
