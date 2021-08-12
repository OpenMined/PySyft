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

# stdlib
import os.path
import sys

try:
    # third party
    import ansible_mitogen.connection
except ImportError:
    base_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.abspath(os.path.join(base_dir, "../../..")))
    del base_dir

# third party
import ansible_mitogen.connection
import ansible_mitogen.process

if sys.version_info > (3,):
    viewkeys = dict.keys
elif sys.version_info > (2, 7):
    viewkeys = dict.viewkeys
else:
    viewkeys = lambda dct: set(dct)


def dict_diff(old, new):
    """
    Return a dict representing the differences between the dicts `old` and
    `new`. Deleted keys appear as a key with the value :data:`None`, added and
    changed keys appear as a key with the new value.
    """
    old_keys = viewkeys(old)
    new_keys = viewkeys(dict(new))
    out = {}
    for key in new_keys - old_keys:
        out[key] = new[key]
    for key in old_keys - new_keys:
        out[key] = None
    for key in old_keys & new_keys:
        if old[key] != new[key]:
            out[key] = new[key]
    return out


class Connection(ansible_mitogen.connection.Connection):
    transport = "local"

    def get_default_cwd(self):
        # https://github.com/ansible/ansible/issues/14489
        return self.loader_basedir

    def get_default_env(self):
        """
        Vanilla Ansible local commands execute with an environment inherited
        from WorkerProcess, we must emulate that.
        """
        return dict_diff(
            old=ansible_mitogen.process.MuxProcess.cls_original_env,
            new=os.environ,
        )
