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
import base64
import logging
import optparse
import re

# third party
import mitogen.core
import mitogen.parent

LOG = logging.getLogger(__name__)

password_incorrect_msg = "sudo password is incorrect"
password_required_msg = "sudo password is required"

# These are base64-encoded UTF-8 as our existing minifier/module server
# struggles with Unicode Python source in some (forgotten) circumstances.
PASSWORD_PROMPTS = [
    "cGFzc3dvcmQ=",  # english
    "bG96aW5rYQ==",  # sr@latin.po
    "44OR44K544Ov44O844OJ",  # ja.po
    "4Kaq4Ka+4Ka44KaT4Kef4Ka+4Kaw4KeN4Kah",  # bn.po
    "2YPZhNmF2Kkg2KfZhNiz2LE=",  # ar.po
    "cGFzYWhpdHph",  # eu.po
    "0L/QsNGA0L7Qu9GM",  # uk.po
    "cGFyb29s",  # et.po
    "c2FsYXNhbmE=",  # fi.po
    "4Kiq4Ki+4Ki44Ki14Kiw4Kih",  # pa.po
    "Y29udHJhc2lnbm8=",  # ia.po
    "Zm9jYWwgZmFpcmU=",  # ga.po
    "16HXodee15Q=",  # he.po
    "4Kqq4Kq+4Kq44Kq14Kqw4KuN4Kqh",  # gu.po
    "0L/QsNGA0L7Qu9Cw",  # bg.po
    "4Kyq4K2N4Kyw4Kys4K2H4Ky2IOCsuOCsmeCtjeCsleCth+CspA==",  # or.po
    "4K6V4K6f4K614K+B4K6a4K+N4K6a4K+K4K6y4K+N",  # ta.po
    "cGFzc3dvcnQ=",  # de.po
    "7JWU7Zi4",  # ko.po
    "0LvQvtC30LjQvdC60LA=",  # sr.po
    "beG6rXQga2jhuql1",  # vi.po
    "c2VuaGE=",  # pt_BR.po
    "cGFzc3dvcmQ=",  # it.po
    "aGVzbG8=",  # cs.po
    "5a+G56K877ya",  # zh_TW.po
    "aGVzbG8=",  # sk.po
    "4LC44LCC4LCV4LGH4LCk4LCq4LCm4LCu4LGB",  # te.po
    "0L/QsNGA0L7Qu9GM",  # kk.po
    "aGFzxYJv",  # pl.po
    "Y29udHJhc2VueWE=",  # ca.po
    "Y29udHJhc2XDsWE=",  # es.po
    "4LSF4LSf4LSv4LS+4LSz4LS14LS+4LSV4LWN4LSV4LWN",  # ml.po
    "c2VuaGE=",  # pt.po
    "5a+G56CB77ya",  # zh_CN.po
    "4KSX4KWB4KSq4KWN4KSk4KS24KSs4KWN4KSm",  # mr.po
    "bMO2c2Vub3Jk",  # sv.po
    "4YOe4YOQ4YOg4YOd4YOa4YOY",  # ka.po
    "4KS24KSs4KWN4KSm4KSV4KWC4KSf",  # hi.po
    "YWRnYW5nc2tvZGU=",  # da.po
    "4La74LeE4LeD4LeK4La04Lav4La6",  # si.po
    "cGFzc29yZA==",  # nb.po
    "d2FjaHR3b29yZA==",  # nl.po
    "4Kaq4Ka+4Ka44KaT4Kef4Ka+4Kaw4KeN4Kah",  # bn_IN.po
    "cGFyb2xh",  # tr.po
    "4LKX4LOB4LKq4LON4LKk4LKq4LKm",  # kn.po
    "c2FuZGk=",  # id.po
    "0L/QsNGA0L7Qu9GM",  # ru.po
    "amVsc3rDsw==",  # hu.po
    "bW90IGRlIHBhc3Nl",  # fr.po
    "aXBoYXNpd2VkaQ==",  # zu.po
    "4Z6W4Z624Z6A4Z+S4Z6Z4Z6f4Z6Y4Z+S4Z6E4Z624Z6P4Z+LwqDhn5Y=",  # km.po
    "4KaX4KeB4Kaq4KeN4Kak4Ka24Kas4KeN4Kam",  # as.po
]


PASSWORD_PROMPT_RE = re.compile(
    mitogen.core.b("|").join(base64.b64decode(s) for s in PASSWORD_PROMPTS), re.I
)

SUDO_OPTIONS = [
    # (False, 'bool', '--askpass', '-A')
    # (False, 'str', '--auth-type', '-a')
    # (False, 'bool', '--background', '-b')
    # (False, 'str', '--close-from', '-C')
    # (False, 'str', '--login-class', 'c')
    (True, "bool", "--preserve-env", "-E"),
    # (False, 'bool', '--edit', '-e')
    # (False, 'str', '--group', '-g')
    (True, "bool", "--set-home", "-H"),
    # (False, 'str', '--host', '-h')
    (False, "bool", "--login", "-i"),
    # (False, 'bool', '--remove-timestamp', '-K')
    # (False, 'bool', '--reset-timestamp', '-k')
    # (False, 'bool', '--list', '-l')
    # (False, 'bool', '--preserve-groups', '-P')
    # (False, 'str', '--prompt', '-p')
    # SELinux options. Passed through as-is.
    (False, "str", "--role", "-r"),
    (False, "str", "--type", "-t"),
    # These options are supplied by default by Ansible, but are ignored, as
    # sudo always runs under a TTY with Mitogen.
    (True, "bool", "--stdin", "-S"),
    (True, "bool", "--non-interactive", "-n"),
    # (False, 'str', '--shell', '-s')
    # (False, 'str', '--other-user', '-U')
    (False, "str", "--user", "-u"),
    # (False, 'bool', '--version', '-V')
    # (False, 'bool', '--validate', '-v')
]


class OptionParser(optparse.OptionParser):
    def help(self):
        self.exit()

    def error(self, msg):
        self.exit(msg=msg)

    def exit(self, status=0, msg=None):
        msg = "sudo: " + (msg or "unsupported option")
        raise mitogen.core.StreamError(msg)


def make_sudo_parser():
    parser = OptionParser()
    for supported, kind, longopt, shortopt in SUDO_OPTIONS:
        if kind == "bool":
            parser.add_option(longopt, shortopt, action="store_true")
        else:
            parser.add_option(longopt, shortopt)
    return parser


def parse_sudo_flags(args):
    parser = make_sudo_parser()
    opts, args = parser.parse_args(args)
    if len(args):
        raise mitogen.core.StreamError("unsupported sudo arguments:" + str(args))
    return opts


class PasswordError(mitogen.core.StreamError):
    pass


def option(default, *args):
    for arg in args:
        if arg is not None:
            return arg
    return default


class Options(mitogen.parent.Options):
    sudo_path = "sudo"
    username = "root"
    password = None
    preserve_env = False
    set_home = False
    login = False

    selinux_role = None
    selinux_type = None

    def __init__(
        self,
        username=None,
        sudo_path=None,
        password=None,
        preserve_env=None,
        set_home=None,
        sudo_args=None,
        login=None,
        selinux_role=None,
        selinux_type=None,
        **kwargs
    ):
        super(Options, self).__init__(**kwargs)
        opts = parse_sudo_flags(sudo_args or [])

        self.username = option(self.username, username, opts.user)
        self.sudo_path = option(self.sudo_path, sudo_path)
        if password:
            self.password = mitogen.core.to_text(password)
        self.preserve_env = option(self.preserve_env, preserve_env, opts.preserve_env)
        self.set_home = option(self.set_home, set_home, opts.set_home)
        self.login = option(self.login, login, opts.login)
        self.selinux_role = option(self.selinux_role, selinux_role, opts.role)
        self.selinux_type = option(self.selinux_type, selinux_type, opts.type)


class SetupProtocol(mitogen.parent.RegexProtocol):
    password_sent = False

    def _on_password_prompt(self, line, match):
        LOG.debug(
            "%s: (password prompt): %s",
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

    PARTIAL_PATTERNS = [
        (PASSWORD_PROMPT_RE, _on_password_prompt),
    ]


class Connection(mitogen.parent.Connection):
    diag_protocol_class = SetupProtocol
    options_class = Options
    create_child = staticmethod(mitogen.parent.hybrid_tty_create_child)
    create_child_args = {
        "escalates_privilege": True,
    }
    child_is_immediate_subprocess = False

    def _get_name(self):
        return u"sudo." + mitogen.core.to_text(self.options.username)

    def get_boot_command(self):
        # Note: sudo did not introduce long-format option processing until July
        # 2013, so even though we parse long-format options, supply short-form
        # to the sudo command.
        boot_cmd = super(Connection, self).get_boot_command()

        bits = [self.options.sudo_path, "-u", self.options.username]
        if self.options.preserve_env:
            bits += ["-E"]
        if self.options.set_home:
            bits += ["-H"]
        if self.options.login:
            bits += ["-i"]
        if self.options.selinux_role:
            bits += ["-r", self.options.selinux_role]
        if self.options.selinux_type:
            bits += ["-t", self.options.selinux_type]

        # special handling for bash builtins
        # TODO: more efficient way of doing this, at least
        # it's only 1 iteration of boot_cmd to go through
        source_found = False
        for cmd in boot_cmd[:]:
            # rip `source` from boot_cmd if it exists; sudo.py can't run this
            # even with -i or -s options
            # since we've already got our ssh command working we shouldn't
            # need to source anymore
            # couldn't figure out how to get this to work using sudo flags
            if "source" == cmd:
                boot_cmd.remove(cmd)
                source_found = True
                continue
            if source_found:
                # remove words until we hit the python interpreter call
                if not cmd.endswith("python"):
                    boot_cmd.remove(cmd)
                else:
                    break

        return bits + ["--"] + boot_cmd
