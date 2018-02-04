"""HTTP :mimetype:`multipart/*`-encoded file streaming.
"""
from __future__ import absolute_import

import re
import requests
import io
import os
from inspect import isgenerator
from uuid import uuid4

import six

from six.moves.urllib.parse import quote

from . import utils

if six.PY3:
    from builtins import memoryview as buffer


CRLF = b'\r\n'

default_chunk_size = 4096


def content_disposition(fn, disptype='file'):
    """Returns a dict containing the MIME content-disposition header for a file.

    .. code-block:: python

        >>> content_disposition('example.txt')
        {'Content-Disposition': 'file; filename="example.txt"'}

        >>> content_disposition('example.txt', 'attachment')
        {'Content-Disposition': 'attachment; filename="example.txt"'}

    Parameters
    ----------
    fn : str
        Filename to retrieve the MIME content-disposition for
    disptype : str
        Rhe disposition type to use for the file
    """
    disp = '%s; filename="%s"' % (
        disptype,
        quote(fn, safe='')
    )
    return {'Content-Disposition': disp}


def content_type(fn):
    """Returns a dict with the content-type header for a file.

    Guesses the mimetype for a filename and returns a dict
    containing the content-type header.

    .. code-block:: python

        >>> content_type('example.txt')
        {'Content-Type': 'text/plain'}

        >>> content_type('example.jpeg')
        {'Content-Type': 'image/jpeg'}

        >>> content_type('example')
        {'Content-Type': 'application/octet-stream'}

    Parameters
    ----------
    fn : str
        Filename to guess the content-type for
    """
    return {'Content-Type': utils.guess_mimetype(fn)}


def multipart_content_type(boundary, subtype='mixed'):
    """Creates a MIME multipart header with the given configuration.

    Returns a dict containing a MIME multipart header with the given
    boundary.

    .. code-block:: python

        >>> multipart_content_type('8K5rNKlLQVyreRNncxOTeg')
        {'Content-Type': 'multipart/mixed; boundary="8K5rNKlLQVyreRNncxOTeg"'}

        >>> multipart_content_type('8K5rNKlLQVyreRNncxOTeg', 'alt')
        {'Content-Type': 'multipart/alt; boundary="8K5rNKlLQVyreRNncxOTeg"'}

    Parameters
    ----------
    boundry : str
        The content delimiter to put into the header
    subtype : str
        The subtype in :mimetype:`multipart/*`-domain to put into the header
    """
    ctype = 'multipart/%s; boundary="%s"' % (
        subtype,
        boundary
    )
    return {'Content-Type': ctype}


class BodyGenerator(object):
    """Generators for creating the body of a :mimetype:`multipart/*`
    HTTP request.

    Parameters
    ----------
    name : str
        The filename of the file(s)/content being encoded
    disptype : str
        The ``Content-Disposition`` of the content
    subtype : str
        The :mimetype:`multipart/*`-subtype of the content
    boundary : str
        An identifier used as a delimiter for the content's body
    """

    def __init__(self, name, disptype='file', subtype='mixed', boundary=None):
        # If the boundary is unspecified, make a random one
        if boundary is None:
            boundary = self._make_boundary()
        self.boundary = boundary

        headers = content_disposition(name, disptype=disptype)
        headers.update(multipart_content_type(boundary, subtype=subtype))
        self.headers = headers

    def _make_boundary(self):
        """Returns a random hexadecimal string (UUID 4).

        The HTTP multipart request body spec requires a boundary string to
        separate different content chunks within a request, and this is
        usually a random string. Using a UUID is an easy way to generate
        a random string of appropriate length as this content separator.
        """
        return uuid4().hex

    def _write_headers(self, headers):
        """Yields the HTTP header text for some content.

        Parameters
        ----------
        headers : dict
            The headers to yield
        """
        if headers:
            for name in sorted(headers.keys()):
                yield name.encode("ascii")
                yield b': '
                yield headers[name].encode("ascii")
                yield CRLF
        yield CRLF

    def write_headers(self):
        """Yields the HTTP header text for the content."""
        for c in self._write_headers(self.headers):
            yield c

    def open(self, **kwargs):
        """Yields the body section for the content.
        """
        yield b'--'
        yield self.boundary.encode()
        yield CRLF

    def file_open(self, fn):
        """Yields the opening text of a file section in multipart HTTP.

        Parameters
        ----------
        fn : str
            Filename for the file being opened and added to the HTTP body
        """
        yield b'--'
        yield self.boundary.encode()
        yield CRLF
        headers = content_disposition(fn)
        headers.update(content_type(fn))
        for c in self._write_headers(headers):
            yield c

    def file_close(self):
        """Yields the end text of a file section in HTTP multipart encoding."""
        yield CRLF

    def close(self):
        """Yields the ends of the content area in a HTTP multipart body."""
        yield b'--'
        yield self.boundary.encode()
        yield b'--'
        yield CRLF


class BufferedGenerator(object):
    """Generator that encodes multipart/form-data.

    An abstract buffered generator class which encodes
    :mimetype:`multipart/form-data`.

    Parameters
    ----------
    name : str
        The name of the file to encode
    chunk_size : int
        The maximum size that any single file chunk may have in bytes
    """

    def __init__(self, name, chunk_size=default_chunk_size):
        self.chunk_size = chunk_size
        self._internal = bytearray(chunk_size)
        self.buf = buffer(self._internal)

        self.name = name
        self.envelope = BodyGenerator(self.name,
                                      disptype='form-data',
                                      subtype='form-data')
        self.headers = self.envelope.headers

    def file_chunks(self, fp):
        """Yields chunks of a file.

        Parameters
        ----------
        fp : io.RawIOBase
            The file to break into chunks
            (must be an open file or have the ``readinto`` method)
        """
        fsize = utils.file_size(fp)
        offset = 0
        if hasattr(fp, 'readinto'):
            while offset < fsize:
                nb = fp.readinto(self._internal)
                yield self.buf[:nb]
                offset += nb
        else:
            while offset < fsize:
                nb = min(self.chunk_size, fsize - offset)
                yield fp.read(nb)
                offset += nb

    def gen_chunks(self, gen):
        """Generates byte chunks of a given size.

        Takes a bytes generator and yields chunks of a maximum of
        ``chunk_size`` bytes.

        Parameters
        ----------
        gen : generator
            The bytes generator that produces the bytes
        """
        for data in gen:
            size = len(data)
            if size < self.chunk_size:
                yield data
            else:
                mv = buffer(data)
                offset = 0
                while offset < size:
                    nb = min(self.chunk_size, size - offset)
                    yield mv[offset:offset + nb]
                    offset += nb

    def body(self, *args, **kwargs):
        """Returns the body of the buffered file.

        .. note:: This function is not actually implemented.
        """
        raise NotImplementedError

    def close(self):
        """Yields the closing text of a multipart envelope."""
        for chunk in self.gen_chunks(self.envelope.close()):
            yield chunk


class FileStream(BufferedGenerator):
    """Generator that encodes multiples files into HTTP multipart.

    A buffered generator that encodes an array of files as
    :mimetype:`multipart/form-data`. This is a concrete implementation of
    :class:`~ipfsapi.multipart.BufferedGenerator`.

    Parameters
    ----------
    name : str
        The filename of the file to encode
    chunk_size : int
        The maximum size that any single file chunk may have in bytes
    """

    def __init__(self, files, chunk_size=default_chunk_size):
        BufferedGenerator.__init__(self, 'files', chunk_size=chunk_size)

        self.files = utils.clean_files(files)

    def body(self):
        """Yields the body of the buffered file."""
        for fp, need_close in self.files:
            try:
                name = os.path.basename(fp.name)
            except AttributeError:
                name = ''
            for chunk in self.gen_chunks(self.envelope.file_open(name)):
                yield chunk
            for chunk in self.file_chunks(fp):
                yield chunk
            for chunk in self.gen_chunks(self.envelope.file_close()):
                yield chunk
            if need_close:
                fp.close()
        for chunk in self.close():
            yield chunk


def glob_compile(pat):
    """Translate a shell glob PATTERN to a regular expression.

    This is almost entirely based on `fnmatch.translate` source-code from the
    python 3.5 standard-library.
    """

    i, n = 0, len(pat)
    res = ''
    while i < n:
        c = pat[i]
        i = i + 1
        if c == '/' and len(pat) > (i + 2) and pat[i:(i + 3)] == '**/':
            # Special-case for "any number of sub-directories" operator since
            # may also expand to no entries:
            #  Otherwise `a/**/b` would expand to `a[/].*[/]b` which wouldn't
            #  match the immediate sub-directories of `a`, like `a/b`.
            i = i + 3
            res = res + '[/]([^/]*[/])*'
        elif c == '*':
            if len(pat) > i and pat[i] == '*':
                i = i + 1
                res = res + '.*'
            else:
                res = res + '[^/]*'
        elif c == '?':
            res = res + '[^/]'
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j = j + 1
            if j < n and pat[j] == ']':
                j = j + 1
            while j < n and pat[j] != ']':
                j = j + 1
            if j >= n:
                res = res + '\\['
            else:
                stuff = pat[i:j].replace('\\', '\\\\')
                i = j + 1
                if stuff[0] == '!':
                    stuff = '^' + stuff[1:]
                elif stuff[0] == '^':
                    stuff = '\\' + stuff
                res = '%s[%s]' % (res, stuff)
        else:
            res = res + re.escape(c)
    return re.compile('^' + res + '\Z(?ms)' + '$')


class DirectoryStream(BufferedGenerator):
    """Generator that encodes a directory into HTTP multipart.

    A buffered generator that encodes an array of files as
    :mimetype:`multipart/form-data`. This is a concrete implementation of
    :class:`~ipfsapi.multipart.BufferedGenerator`.

    Parameters
    ----------
    directory : str
        The filepath of the directory to encode
    patterns : str | list
        A single glob pattern or a list of several glob patterns and
        compiled regular expressions used to determine which filepaths to match
    chunk_size : int
        The maximum size that any single file chunk may have in bytes
    """

    def __init__(self,
                 directory,
                 recursive=False,
                 patterns='**',
                 chunk_size=default_chunk_size):
        BufferedGenerator.__init__(self, directory, chunk_size=chunk_size)

        self.patterns = []
        patterns = [patterns] if isinstance(patterns, str) else patterns
        for pattern in patterns:
            if isinstance(pattern, str):
                self.patterns.append(glob_compile(pattern))
            else:
                self.patterns.append(pattern)

        self.directory = os.path.normpath(directory)
        self.recursive = recursive
        self._request = self._prepare()
        self.headers = self._request.headers

    def body(self):
        """Returns the HTTP headers for this directory upload request."""
        return self._request.body

    def headers(self):
        """Returns the HTTP body for this directory upload request."""
        return self._request.headers

    def _prepare(self):
        """Pre-formats the multipart HTTP request to transmit the directory."""
        names = []

        added_directories = set()

        def add_directory(short_path):
            # Do not continue if this directory has already been added
            if short_path in added_directories:
                return

            # Scan for first super-directory that has already been added
            dir_base  = short_path
            dir_parts = []
            while dir_base:
                dir_base, dir_name = os.path.split(dir_base)
                dir_parts.append(dir_name)
                if dir_base in added_directories:
                    break

            # Add missing intermediate directory nodes in the right order
            while dir_parts:
                dir_base = os.path.join(dir_base, dir_parts.pop())

                # Create an empty, fake file to represent the directory
                mock_file = io.StringIO()
                mock_file.write(u'')
                # Add this directory to those that will be sent
                names.append(('files',
                             (dir_base, mock_file, 'application/x-directory')))
                # Remember that this directory has already been sent
                added_directories.add(dir_base)

        def add_file(short_path, full_path):
            try:
                # Always add files in wildcard directories
                names.append(('files', (short_name,
                                        open(full_path, 'rb'),
                                        'application/octet-stream')))
            except OSError:
                # File might have disappeared between `os.walk()` and `open()`
                pass

        def match_short_path(short_path):
            # Remove initial path component so that all files are based in
            # the target directory itself (not one level above)
            if os.sep in short_path:
                path = short_path.split(os.sep, 1)[1]
            else:
                return False

            # Convert all path seperators to POSIX style
            path = path.replace(os.sep, '/')

            # Do the matching and the simplified path
            for pattern in self.patterns:
                if pattern.match(path):
                    return True
            return False

        # Identify the unecessary portion of the relative path
        truncate = os.path.dirname(self.directory)
        # Traverse the filesystem downward from the target directory's uri
        # Errors: `os.walk()` will simply return an empty generator if the
        #         target directory does not exist.
        wildcard_directories = set()
        for curr_dir, _, files in os.walk(self.directory):
            # find the path relative to the directory being added
            if len(truncate) > 0:
                _, _, short_path = curr_dir.partition(truncate)
            else:
                short_path = curr_dir
            # remove leading / or \ if it is present
            if short_path.startswith(os.sep):
                short_path = short_path[1:]

            wildcard_directory = False
            if os.path.split(short_path)[0] in wildcard_directories:
                # Parent directory has matched a pattern, all sub-nodes should
                # be added too
                wildcard_directories.add(short_path)
                wildcard_directory = True
            else:
                # Check if directory path matches one of the patterns
                if match_short_path(short_path):
                    # Directory matched pattern and it should therefor
                    # be added along with all of its contents
                    wildcard_directories.add(short_path)
                    wildcard_directory = True

            # Always add directories within wildcard directories - even if they
            # are empty
            if wildcard_directory:
                add_directory(short_path)

            # Iterate across the files in the current directory
            for filename in files:
                # Find the filename relative to the directory being added
                short_name = os.path.join(short_path, filename)
                filepath = os.path.join(curr_dir, filename)

                if wildcard_directory:
                    # Always add files in wildcard directories
                    add_file(short_name, filepath)
                else:
                    # Add file (and all missing intermediary directories)
                    # if it matches one of the patterns
                    if match_short_path(short_name):
                        add_directory(short_path)
                        add_file(short_name, filepath)
        # Send the request and present the response body to the user
        req = requests.Request("POST", 'http://localhost', files=names)
        prep = req.prepare()
        return prep


class BytesStream(BufferedGenerator):
    """A buffered generator that encodes bytes as
    :mimetype:`multipart/form-data`.

    Parameters
    ----------
    data : bytes
        The binary data to stream to the daemon
    chunk_size : int
        The maximum size of a single data chunk
    """

    def __init__(self, data, chunk_size=default_chunk_size):
        BufferedGenerator.__init__(self, 'bytes', chunk_size=chunk_size)

        self.data = data if isgenerator(data) else (data,)

    def body(self):
        """Yields the encoded body."""
        for chunk in self.gen_chunks(self.envelope.file_open(self.name)):
            yield chunk
        for chunk in self.gen_chunks(self.data):
            yield chunk
        for chunk in self.gen_chunks(self.envelope.file_close()):
            yield chunk
        for chunk in self.close():
            yield chunk


def stream_files(files, chunk_size=default_chunk_size):
    """Gets a buffered generator for streaming files.

    Returns a buffered generator which encodes a file or list of files as
    :mimetype:`multipart/form-data` with the corresponding headers.

    Parameters
    ----------
    files : str
        The file(s) to stream
    chunk_size : int
        Maximum size of each stream chunk
    """
    stream = FileStream(files, chunk_size=chunk_size)

    return stream.body(), stream.headers


def stream_directory(directory,
                     recursive=False,
                     patterns='**',
                     chunk_size=default_chunk_size):
    """Gets a buffered generator for streaming directories.

    Returns a buffered generator which encodes a directory as
    :mimetype:`multipart/form-data` with the corresponding headers.

    Parameters
    ----------
    directory : str
        The filepath of the directory to stream
    recursive : bool
        Stream all content within the directory recursively?
    patterns : str | list
        Single *glob* pattern or list of *glob* patterns and compiled
        regular expressions to match the names of the filepaths to keep
    chunk_size : int
        Maximum size of each stream chunk
    """
    stream = DirectoryStream(directory,
                             recursive=recursive,
                             patterns=patterns,
                             chunk_size=chunk_size)

    return stream.body(), stream.headers


def stream_filesystem_node(path,
                           recursive=False,
                           patterns='**',
                           chunk_size=default_chunk_size):
    """Gets a buffered generator for streaming either files or directories.

    Returns a buffered generator which encodes the file or directory at the
    given path as :mimetype:`multipart/form-data` with the corresponding
    headers.

    Parameters
    ----------
    path : str
        The filepath of the directory or file to stream
    recursive : bool
        Stream all content within the directory recursively?
    patterns : str | list
        Single *glob* pattern or list of *glob* patterns and compiled
        regular expressions to match the names of the filepaths to keep
    chunk_size : int
        Maximum size of each stream chunk
    """
    is_dir = isinstance(path, six.string_types) and os.path.isdir(path)
    if recursive or is_dir:
        return stream_directory(path, recursive, patterns, chunk_size)
    else:
        return stream_files(path, chunk_size)


def stream_bytes(data, chunk_size=default_chunk_size):
    """Gets a buffered generator for streaming binary data.

    Returns a buffered generator which encodes binary data as
    :mimetype:`multipart/form-data` with the corresponding headers.

    Parameters
    ----------
    data : bytes
        The data bytes to stream
    chunk_size : int
        The maximum size of each stream chunk

    Returns
    -------
        (generator, dict)
    """
    stream = BytesStream(data, chunk_size=chunk_size)

    return stream.body(), stream.headers


def stream_text(text, chunk_size=default_chunk_size):
    """Gets a buffered generator for streaming text.

    Returns a buffered generator which encodes a string as
    :mimetype:`multipart/form-data` with the corresponding headers.

    Parameters
    ----------
    text : str
        The data bytes to stream
    chunk_size : int
        The maximum size of each stream chunk

    Returns
    -------
        (generator, dict)
    """
    if isgenerator(text):
        def binary_stream():
            for item in text:
                if six.PY2 and isinstance(text, six.binary_type):
                    #PY2: Allow binary strings under Python 2 since
                    # Python 2 code is not expected to always get the
                    # distinction between text and binary strings right.
                    yield text
                else:
                    yield text.encode("utf-8")
        data = binary_stream()
    elif six.PY2 and isinstance(text, six.binary_type):
        #PY2: See above.
        data = text
    else:
        data = text.encode("utf-8")

    return stream_bytes(data, chunk_size)
