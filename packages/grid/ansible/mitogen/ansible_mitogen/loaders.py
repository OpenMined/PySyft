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
Stable names for PluginLoader instances across Ansible versions.
"""

# future
from __future__ import absolute_import

# stdlib
import distutils.version

__all__ = [
    "action_loader",
    "connection_loader",
    "module_loader",
    "module_utils_loader",
    "shell_loader",
    "strategy_loader",
]

# third party
import ansible

ANSIBLE_VERSION_MIN = (2, 10)
ANSIBLE_VERSION_MAX = (2, 11)

NEW_VERSION_MSG = (
    "Your Ansible version (%s) is too recent. The most recent version\n"
    "supported by Mitogen for Ansible is %s.x. Please check the Mitogen\n"
    "release notes to see if a new version is available, otherwise\n"
    "subscribe to the corresponding GitHub issue to be notified when\n"
    "support becomes available.\n"
    "\n"
    "    https://mitogen.rtfd.io/en/latest/changelog.html\n"
    "    https://github.com/mitogen-hq/mitogen/issues/\n"
)
OLD_VERSION_MSG = (
    "Your version of Ansible (%s) is too old. The oldest version supported by "
    "Mitogen for Ansible is %s."
)


def assert_supported_release():
    """
    Throw AnsibleError with a descriptive message in case of being loaded into
    an unsupported Ansible release.
    """
    v = ansible.__version__
    if not isinstance(v, tuple):
        v = tuple(distutils.version.LooseVersion(v).version)

    if v[:2] < ANSIBLE_VERSION_MIN:
        raise ansible.errors.AnsibleError(OLD_VERSION_MSG % (v, ANSIBLE_VERSION_MIN))

    if v[:2] > ANSIBLE_VERSION_MAX:
        raise ansible.errors.AnsibleError(
            NEW_VERSION_MSG % (ansible.__version__, ANSIBLE_VERSION_MAX)
        )


# this is the first file our strategy plugins import, so we need to check this here
# in prior Ansible versions, connection_loader.get_with_context didn't exist, so if a user
# is trying to load an old Ansible version, we'll fail and error gracefully
assert_supported_release()


# third party
from ansible.plugins.loader import action_loader
from ansible.plugins.loader import connection_loader
from ansible.plugins.loader import module_loader
from ansible.plugins.loader import module_utils_loader
from ansible.plugins.loader import shell_loader
from ansible.plugins.loader import strategy_loader

# These are original, unwrapped implementations
action_loader__get = action_loader.get
connection_loader__get = connection_loader.get_with_context
