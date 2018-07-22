"""A module to handle generic operations.
"""

from __future__ import absolute_import

import mimetypes
import os
from functools import wraps

import six


def guess_mimetype(filename):
    """Guesses the mimetype of a file based on the given ``filename``.

    .. code-block:: python

        >>> guess_mimetype('example.txt')
        'text/plain'
        >>> guess_mimetype('/foo/bar/example')
        'application/octet-stream'

    Parameters
    ----------
    filename : str
        The file name or path for which the mimetype is to be guessed
    """
    fn = os.path.basename(filename)
    return mimetypes.guess_type(fn)[0] or 'application/octet-stream'


def ls_dir(dirname):
    """Returns files and subdirectories within a given directory.

    Returns a pair of lists, containing the names of directories and files
    in ``dirname``.

    Raises
    ------
    OSError : Accessing the given directory path failed

    Parameters
    ----------
    dirname : str
        The path of the directory to be listed
    """
    ls = os.listdir(dirname)
    files = [p for p in ls if os.path.isfile(os.path.join(dirname, p))]
    dirs = [p for p in ls if os.path.isdir(os.path.join(dirname, p))]
    return files, dirs


def clean_file(file):
    """Returns a tuple containing a ``file``-like object and a close indicator.

    This ensures the given file is opened and keeps track of files that should
    be closed after use (files that were not open prior to this function call).

    Raises
    ------
    OSError : Accessing the given file path failed

    Parameters
    ----------
    file : str | io.IOBase
        A filepath or ``file``-like object that may or may not need to be
        opened
    """
    if not hasattr(file, 'read'):
        return open(file, 'rb'), True
    else:
        return file, False


def clean_files(files):
    """Generates tuples with a ``file``-like object and a close indicator.

    This is a generator of tuples, where the first element is the file object
    and the second element is a boolean which is True if this module opened the
    file (and thus should close it).

    Raises
    ------
    OSError : Accessing the given file path failed

    Parameters
    ----------
    files : list | io.IOBase | str
        Collection or single instance of a filepath and file-like object
    """
    if isinstance(files, (list, tuple)):
        for f in files:
            yield clean_file(f)
    else:
        yield clean_file(files)


def file_size(f):
    """Returns the size of a file in bytes.

    Raises
    ------
    OSError : Accessing the given file path failed

    Parameters
    ----------
    f : io.IOBase | str
        The file path or object for which the size should be determined
    """
    if isinstance(f, (six.string_types, six.text_type)):
        return os.path.getsize(f)
    else:
        cur = f.tell()
        f.seek(0, 2)
        size = f.tell()
        f.seek(cur)
        return size


class return_field(object):
    """Decorator that returns the given field of a json response.

    Parameters
    ----------
    field : object
        The response field to be returned for all invocations
    """
    def __init__(self, field):
        self.field = field

    def __call__(self, cmd):
        """Wraps a command so that only a specified field is returned.

        Parameters
        ----------
        cmd : callable
            A command that is intended to be wrapped
        """
        @wraps(cmd)
        def wrapper(*args, **kwargs):
            """Returns the specified field of the command invocation.

            Parameters
            ----------
            args : list
                Positional parameters to pass to the wrapped callable
            kwargs : dict
                Named parameter to pass to the wrapped callable
            """
            res = cmd(*args, **kwargs)
            return res[self.field]
        return wrapper
