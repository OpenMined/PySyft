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

# third party
import mitogen.core
import mitogen.parent

LOG = logging.getLogger(__name__)


class Options(mitogen.parent.Options):
    container = None
    image = None
    username = None
    docker_path = u"docker"

    def __init__(
        self, container=None, image=None, docker_path=None, username=None, **kwargs
    ):
        super(Options, self).__init__(**kwargs)
        assert container or image
        if container:
            self.container = mitogen.core.to_text(container)
        if image:
            self.image = mitogen.core.to_text(image)
        if docker_path:
            self.docker_path = mitogen.core.to_text(docker_path)
        if username:
            self.username = mitogen.core.to_text(username)


class Connection(mitogen.parent.Connection):
    options_class = Options
    child_is_immediate_subprocess = False

    # TODO: better way of capturing errors such as "No such container."
    create_child_args = {"merge_stdio": True}

    def _get_name(self):
        return u"docker." + (self.options.container or self.options.image)

    def get_boot_command(self):
        args = ["--interactive"]
        if self.options.username:
            args += ["--user=" + self.options.username]

        bits = [self.options.docker_path]
        if self.options.container:
            bits += ["exec"] + args + [self.options.container]
        elif self.options.image:
            bits += ["run"] + args + ["--rm", self.options.image]

        return bits + super(Connection, self).get_boot_command()
