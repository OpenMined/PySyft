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
import logging
import os

# third party
import mitogen.core
import mitogen.utils

try:
    # third party
    from __main__ import display
except ImportError:
    # third party
    from ansible.utils.display import Display

    display = Display()


#: The process name set via :func:`set_process_name`.
_process_name = None

#: The PID of the process that last called :func:`set_process_name`, so its
#: value can be ignored in unknown fork children.
_process_pid = None


def set_process_name(name):
    """
    Set a name to adorn log messages with.
    """
    global _process_name
    _process_name = name

    global _process_pid
    _process_pid = os.getpid()


class Handler(logging.Handler):
    """
    Use Mitogen's log format, but send the result to a Display method.
    """

    def __init__(self, normal_method):
        logging.Handler.__init__(self)
        self.formatter = mitogen.utils.log_get_formatter()
        self.normal_method = normal_method

    #: Set of target loggers that produce warnings and errors that spam the
    #: console needlessly. Their log level is forced to INFO. A better strategy
    #: may simply be to bury all target logs in DEBUG output, but not by
    #: overriding their log level as done here.
    NOISY_LOGGERS = frozenset(
        [
            "dnf",  # issue #272; warns when a package is already installed.
            "boto",  # issue #541; normal boto retry logic can cause ERROR logs.
        ]
    )

    def emit(self, record):
        mitogen_name = getattr(record, "mitogen_name", "")
        if mitogen_name == "stderr":
            record.levelno = logging.ERROR
        if mitogen_name in self.NOISY_LOGGERS and record.levelno >= logging.WARNING:
            record.levelno = logging.DEBUG

        if _process_pid == os.getpid():
            process_name = _process_name
        else:
            process_name = "?"

        s = "[%-4s %d] %s" % (process_name, os.getpid(), self.format(record))
        if record.levelno >= logging.ERROR:
            display.error(s, wrap_text=False)
        elif record.levelno >= logging.WARNING:
            display.warning(s, formatted=True)
        else:
            self.normal_method(s)


def setup():
    """
    Install handlers for Mitogen loggers to redirect them into the Ansible
    display framework. Ansible installs its own logging framework handlers when
    C.DEFAULT_LOG_PATH is set, therefore disable propagation for our handlers.
    """
    l_mitogen = logging.getLogger("mitogen")
    l_mitogen_io = logging.getLogger("mitogen.io")
    l_ansible_mitogen = logging.getLogger("ansible_mitogen")
    l_operon = logging.getLogger("operon")

    for logger in l_mitogen, l_mitogen_io, l_ansible_mitogen, l_operon:
        logger.handlers = [Handler(display.vvv)]
        logger.propagate = False

    if display.verbosity > 2:
        l_ansible_mitogen.setLevel(logging.DEBUG)
        l_mitogen.setLevel(logging.DEBUG)
    else:
        # Mitogen copies the active log level into new children, allowing them
        # to filter tiny messages before they hit the network, and therefore
        # before they wake the IO loop. Explicitly setting INFO saves ~4%
        # running against just the local machine.
        l_mitogen.setLevel(logging.ERROR)
        l_ansible_mitogen.setLevel(logging.ERROR)

    if display.verbosity > 3:
        l_mitogen_io.setLevel(logging.DEBUG)
