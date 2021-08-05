# Copyright 2018, Yannig Perre
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

# third party
import mitogen.core
import mitogen.parent


class Options(mitogen.parent.Options):
    pod = None
    kubectl_path = "kubectl"
    kubectl_args = None

    def __init__(self, pod, kubectl_path=None, kubectl_args=None, **kwargs):
        super(Options, self).__init__(**kwargs)
        assert pod
        self.pod = pod
        if kubectl_path:
            self.kubectl_path = kubectl_path
        self.kubectl_args = kubectl_args or []


class Connection(mitogen.parent.Connection):
    options_class = Options
    child_is_immediate_subprocess = True

    # TODO: better way of capturing errors such as "No such container."
    create_child_args = {"merge_stdio": True}

    def _get_name(self):
        return u"kubectl.%s%s" % (self.options.pod, self.options.kubectl_args)

    def get_boot_command(self):
        bits = (
            [self.options.kubectl_path]
            + self.options.kubectl_args
            + ["exec", "-it", self.options.pod]
        )
        return bits + ["--"] + super(Connection, self).get_boot_command()
