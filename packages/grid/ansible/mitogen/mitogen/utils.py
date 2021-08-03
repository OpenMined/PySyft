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
import datetime
import logging
import os
import sys

# third party
import mitogen
import mitogen.core
import mitogen.master
import mitogen.parent

iteritems = getattr(dict, "iteritems", dict.items)

if mitogen.core.PY3:
    iteritems = dict.items
else:
    iteritems = dict.iteritems


def setup_gil():
    """
    Set extremely long GIL release interval to let threads naturally progress
    through CPU-heavy sequences without forcing the wake of another thread that
    may contend trying to run the same CPU-heavy code. For the new-style
    Ansible work, this drops runtime ~33% and involuntary context switches by
    >80%, essentially making threads cooperatively scheduled.
    """
    try:
        # Python 2.
        sys.setcheckinterval(100000)
    except AttributeError:
        pass

    try:
        # Python 3.
        sys.setswitchinterval(10)
    except AttributeError:
        pass


def disable_site_packages():
    """
    Remove all entries mentioning ``site-packages`` or ``Extras`` from
    :attr:sys.path. Used primarily for testing on OS X within a virtualenv,
    where OS X bundles some ancient version of the :mod:`six` module.
    """
    for entry in sys.path[:]:
        if "site-packages" in entry or "Extras" in entry:
            sys.path.remove(entry)


def _formatTime(record, datefmt=None):
    dt = datetime.datetime.fromtimestamp(record.created)
    return dt.strftime(datefmt)


def log_get_formatter():
    datefmt = "%H:%M:%S"
    if sys.version_info > (2, 6):
        datefmt += ".%f"
    fmt = "%(asctime)s %(levelname).1s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt)
    formatter.formatTime = _formatTime
    return formatter


def log_to_file(path=None, io=False, level="INFO"):
    """
    Install a new :class:`logging.Handler` writing applications logs to the
    filesystem. Useful when debugging slave IO problems.

    Parameters to this function may be overridden at runtime using environment
    variables. See :ref:`logging-env-vars`.

    :param str path:
        If not :data:`None`, a filesystem path to write logs to. Otherwise,
        logs are written to :data:`sys.stderr`.

    :param bool io:
        If :data:`True`, include extremely verbose IO logs in the output.
        Useful for debugging hangs, less useful for debugging application code.

    :param str level:
        Name of the :mod:`logging` package constant that is the minimum level
        to log at. Useful levels are ``DEBUG``, ``INFO``, ``WARNING``, and
        ``ERROR``.
    """
    log = logging.getLogger("")
    if path:
        fp = open(path, "w", 1)
        mitogen.core.set_cloexec(fp.fileno())
    else:
        fp = sys.stderr

    level = os.environ.get("MITOGEN_LOG_LEVEL", level).upper()
    io = level == "IO"
    if io:
        level = "DEBUG"
        logging.getLogger("mitogen.io").setLevel(level)

    level = getattr(logging, level, logging.INFO)
    log.setLevel(level)

    # Prevent accidental duplicate log_to_file() calls from generating
    # duplicate output.
    for handler_ in reversed(log.handlers):
        if getattr(handler_, "is_mitogen", None):
            log.handlers.remove(handler_)

    handler = logging.StreamHandler(fp)
    handler.is_mitogen = True
    handler.formatter = log_get_formatter()
    log.handlers.insert(0, handler)


def run_with_router(func, *args, **kwargs):
    """
    Arrange for `func(router, *args, **kwargs)` to run with a temporary
    :class:`mitogen.master.Router`, ensuring the Router and Broker are
    correctly shut down during normal or exceptional return.

    :returns:
        `func`'s return value.
    """
    broker = mitogen.master.Broker()
    router = mitogen.master.Router(broker)
    try:
        return func(router, *args, **kwargs)
    finally:
        broker.shutdown()
        broker.join()


def with_router(func):
    """
    Decorator version of :func:`run_with_router`. Example:

    .. code-block:: python

        @with_router
        def do_stuff(router, arg):
            pass

        do_stuff(blah, 123)
    """

    def wrapper(*args, **kwargs):
        return run_with_router(func, *args, **kwargs)

    if mitogen.core.PY3:
        wrapper.func_name = func.__name__
    else:
        wrapper.func_name = func.func_name
    return wrapper


PASSTHROUGH = (
    int,
    float,
    bool,
    type(None),
    mitogen.core.Context,
    mitogen.core.CallError,
    mitogen.core.Blob,
    mitogen.core.Secret,
)


def cast(obj):
    """
    Many tools love to subclass built-in types in order to implement useful
    functionality, such as annotating the safety of a Unicode string, or adding
    additional methods to a dict. However, cPickle loves to preserve those
    subtypes during serialization, resulting in CallError during :meth:`call
    <mitogen.parent.Context.call>` in the target when it tries to deserialize
    the data.

    This function walks the object graph `obj`, producing a copy with any
    custom sub-types removed. The functionality is not default since the
    resulting walk may be computationally expensive given a large enough graph.

    See :ref:`serialization-rules` for a list of supported types.

    :param obj:
        Object to undecorate.
    :returns:
        Undecorated object.
    """
    if isinstance(obj, dict):
        return dict((cast(k), cast(v)) for k, v in iteritems(obj))
    if isinstance(obj, (list, tuple)):
        return [cast(v) for v in obj]
    if isinstance(obj, PASSTHROUGH):
        return obj
    if isinstance(obj, mitogen.core.UnicodeType):
        return mitogen.core.UnicodeType(obj)
    if isinstance(obj, mitogen.core.BytesType):
        return mitogen.core.BytesType(obj)

    raise TypeError(f"Cannot serialize: {type(obj)!r}: {obj!r}")
