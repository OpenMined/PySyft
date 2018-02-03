# -*- coding: utf-8 -*-
"""IPFS API Bindings for Python.

Classes:

 * Client – a TCP client for interacting with an IPFS daemon
"""
from __future__ import absolute_import

import os
import warnings

from . import http, multipart, utils, exceptions, encoding

DEFAULT_HOST = str(os.environ.get("PY_IPFSAPI_DEFAULT_HOST", 'localhost'))
DEFAULT_PORT = int(os.environ.get("PY_IPFSAPI_DEFAULT_PORT", 5001))
DEFAULT_BASE = str(os.environ.get("PY_IPFSAPI_DEFAULT_BASE", 'api/v0'))

VERSION_MINIMUM = "0.4.3"
VERSION_MAXIMUM = "0.5.0"


def assert_version(version, minimum=VERSION_MINIMUM, maximum=VERSION_MAXIMUM):
    """Make sure that the given daemon version is supported by this client
    version.

    Raises
    ------
    ~ipfsapi.exceptions.VersionMismatch

    Parameters
    ----------
    version : str
        The version of an IPFS daemon.
    minimum : str
        The minimal IPFS version to allow.
    maximum : str
        The maximum IPFS version to allow.
    """
    # Convert version strings to integer tuples
    version = list(map(int, version.split('-', 1)[0].split('.')))
    minimum = list(map(int, minimum.split('-', 1)[0].split('.')))
    maximum = list(map(int, maximum.split('-', 1)[0].split('.')))

    if minimum > version or version >= maximum:
        raise exceptions.VersionMismatch(version, minimum, maximum)


def connect(host=DEFAULT_HOST, port=DEFAULT_PORT, base=DEFAULT_BASE,
            chunk_size=multipart.default_chunk_size, **defaults):
    """Create a new :class:`~ipfsapi.Client` instance and connect to the
    daemon to validate that its version is supported.

    Raises
    ------
    ~ipfsapi.exceptions.VersionMismatch
    ~ipfsapi.exceptions.ErrorResponse
    ~ipfsapi.exceptions.ConnectionError
    ~ipfsapi.exceptions.ProtocolError
    ~ipfsapi.exceptions.StatusError
    ~ipfsapi.exceptions.TimeoutError


    All parameters are identical to those passed to the constructor of the
    :class:`~ipfsapi.Client` class.

    Returns
    -------
        ~ipfsapi.Client
    """
    # Create client instance
    client = Client(host, port, base, chunk_size, **defaults)

    # Query version number from daemon and validate it
    assert_version(client.version()['Version'])

    return client


class Client(object):
    """A TCP client for interacting with an IPFS daemon.

    A :class:`~ipfsapi.Client` instance will not actually establish a
    connection to the daemon until at least one of it's methods is called.

    Parameters
    ----------
    host : str
        Hostname or IP address of the computer running the ``ipfs daemon``
        node (defaults to the local system)
    port : int
        The API port of the IPFS deamon (usually 5001)
    base : str
        Path of the deamon's API (currently always ``api/v0``)
    chunk_size : int
        The size of the chunks to break uploaded files and text content into
    """

    _clientfactory = http.HTTPClient

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 base=DEFAULT_BASE, chunk_size=multipart.default_chunk_size,
                 **defaults):
        """Connects to the API port of an IPFS node."""

        self.chunk_size = chunk_size

        self._client = self._clientfactory(host, port, base, **defaults)

    def add(self, files, recursive=False, pattern='**', *args, **kwargs):
        """Add a file, or directory of files to IPFS.

        .. code-block:: python

            >>> with io.open('nurseryrhyme.txt', 'w', encoding='utf-8') as f:
            ...     numbytes = f.write('Mary had a little lamb')
            >>> c.add('nurseryrhyme.txt')
            {'Hash': 'QmZfF6C9j4VtoCsTp4KSrhYH47QMd3DNXVZBKaxJdhaPab',
             'Name': 'nurseryrhyme.txt'}

        Parameters
        ----------
        files : str
            A filepath to either a file or directory
        recursive : bool
            Controls if files in subdirectories are added or not
        pattern : str | list
            Single `*glob* <https://docs.python.org/3/library/glob.html>`_
            pattern or list of *glob* patterns and compiled regular expressions
            to match the names of the filepaths to keep
        trickle : bool
            Use trickle-dag format (optimized for streaming) when generating
            the dag; see `the FAQ <https://github.com/ipfs/faq/issues/218>` for
            more information (Default: ``False``)
        only_hash : bool
            Only chunk and hash, but do not write to disk (Default: ``False``)
        wrap_with_directory : bool
            Wrap files with a directory object to preserve their filename
            (Default: ``False``)
        chunker : str
            The chunking algorithm to use
        pin : bool
            Pin this object when adding (Default: ``True``)

        Returns
        -------
            dict: File name and hash of the added file node
        """
        #PY2: No support for kw-only parameters after glob parameters
        opts = {
            "trickle": kwargs.pop("trickle", False),
            "only-hash": kwargs.pop("only_hash", False),
            "wrap-with-directory": kwargs.pop("wrap_with_directory", False),
            "pin": kwargs.pop("pin", True)
        }
        if "chunker" in kwargs:
            opts["chunker"] = kwargs.pop("chunker")
        kwargs.setdefault("opts", opts)

        body, headers = multipart.stream_filesystem_node(
            files, recursive, pattern, self.chunk_size
        )
        return self._client.request('/add', decoder='json',
                                    data=body, headers=headers, **kwargs)

    def get(self, multihash, **kwargs):
        """Downloads a file, or directory of files from IPFS.

        Files are placed in the current working directory.

        Parameters
        ----------
        multihash : str
            The path to the IPFS object(s) to be outputted
        """
        args = (multihash,)
        return self._client.download('/get', args, **kwargs)

    def cat(self, multihash, **kwargs):
        r"""Retrieves the contents of a file identified by hash.

        .. code-block:: python

            >>> c.cat('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D')
            Traceback (most recent call last):
              ...
            ipfsapi.exceptions.Error: this dag node is a directory
            >>> c.cat('QmeKozNssnkJ4NcyRidYgDY2jfRZqVEoRGfipkgath71bX')
            b'<!DOCTYPE html>\n<html>\n\n<head>\n<title>ipfs example viewer</…'

        Parameters
        ----------
        multihash : str
            The path to the IPFS object(s) to be retrieved

        Returns
        -------
            str : File contents
        """
        args = (multihash,)
        return self._client.request('/cat', args, **kwargs)

    def ls(self, multihash, **kwargs):
        """Returns a list of objects linked to by the given hash.

        .. code-block:: python

            >>> c.ls('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D')
            {'Objects': [
              {'Hash': 'QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D',
               'Links': [
                {'Hash': 'Qmd2xkBfEwEs9oMTk77A6jrsgurpF3ugXSg7dtPNFkcNMV',
                 'Name': 'Makefile',          'Size': 174, 'Type': 2},
                 …
                {'Hash': 'QmSY8RfVntt3VdxWppv9w5hWgNrE31uctgTiYwKir8eXJY',
                 'Name': 'published-version', 'Size': 55,  'Type': 2}
                ]}
              ]}

        Parameters
        ----------
        multihash : str
            The path to the IPFS object(s) to list links from

        Returns
        -------
            dict : Directory information and contents
        """
        args = (multihash,)
        return self._client.request('/ls', args, decoder='json', **kwargs)

    def refs(self, multihash, **kwargs):
        """Returns a list of hashes of objects referenced by the given hash.

        .. code-block:: python

            >>> c.refs('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D')
            [{'Ref': 'Qmd2xkBfEwEs9oMTk77A6jrsgurpF3ugXSg7 … cNMV', 'Err': ''},
             …
             {'Ref': 'QmSY8RfVntt3VdxWppv9w5hWgNrE31uctgTi … eXJY', 'Err': ''}]

        Parameters
        ----------
        multihash : str
            Path to the object(s) to list refs from

        Returns
        -------
            list
        """
        args = (multihash,)
        return self._client.request('/refs', args, decoder='json', **kwargs)

    def refs_local(self, **kwargs):
        """Displays the hashes of all local objects.

        .. code-block:: python

            >>> c.refs_local()
            [{'Ref': 'Qmd2xkBfEwEs9oMTk77A6jrsgurpF3ugXSg7 … cNMV', 'Err': ''},
             …
             {'Ref': 'QmSY8RfVntt3VdxWppv9w5hWgNrE31uctgTi … eXJY', 'Err': ''}]

        Returns
        -------
            list
        """
        return self._client.request('/refs/local', decoder='json', **kwargs)

    def block_stat(self, multihash, **kwargs):
        """Returns a dict with the size of the block with the given hash.

        .. code-block:: python

            >>> c.block_stat('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D')
            {'Key':  'QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D',
             'Size': 258}

        Parameters
        ----------
        multihash : str
            The base58 multihash of an existing block to stat

        Returns
        -------
            dict : Information about the requested block
        """
        args = (multihash,)
        return self._client.request('/block/stat', args,
                                    decoder='json', **kwargs)

    def block_get(self, multihash, **kwargs):
        r"""Returns the raw contents of a block.

        .. code-block:: python

            >>> c.block_get('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D')
            b'\x121\n"\x12 \xdaW>\x14\xe5\xc1\xf6\xe4\x92\xd1 … \n\x02\x08\x01'

        Parameters
        ----------
        multihash : str
            The base58 multihash of an existing block to get

        Returns
        -------
            str : Value of the requested block
        """
        args = (multihash,)
        return self._client.request('/block/get', args, **kwargs)

    def block_put(self, file, **kwargs):
        """Stores the contents of the given file object as an IPFS block.

        .. code-block:: python

            >>> c.block_put(io.BytesIO(b'Mary had a little lamb'))
                {'Key':  'QmeV6C6XVt1wf7V7as7Yak3mxPma8jzpqyhtRtCvpKcfBb',
                 'Size': 22}

        Parameters
        ----------
        file : io.RawIOBase
            The data to be stored as an IPFS block

        Returns
        -------
            dict : Information about the new block

                   See :meth:`~ipfsapi.Client.block_stat`
        """
        body, headers = multipart.stream_files(file, self.chunk_size)
        return self._client.request('/block/put', decoder='json',
                                    data=body, headers=headers, **kwargs)

    def bitswap_wantlist(self, peer=None, **kwargs):
        """Returns blocks currently on the bitswap wantlist.

        .. code-block:: python

            >>> c.bitswap_wantlist()
            {'Keys': [
                'QmeV6C6XVt1wf7V7as7Yak3mxPma8jzpqyhtRtCvpKcfBb',
                'QmdCWFLDXqgdWQY9kVubbEHBbkieKd3uo7MtCm7nTZZE9K',
                'QmVQ1XvYGF19X4eJqz1s7FJYJqAxFC4oqh3vWJJEXn66cp'
            ]}

        Parameters
        ----------
        peer : str
            Peer to show wantlist for.

        Returns
        -------
            dict : List of wanted blocks
        """
        args = (peer,)
        return self._client.request('/bitswap/wantlist', args,
                                    decoder='json', **kwargs)

    def bitswap_stat(self, **kwargs):
        """Returns some diagnostic information from the bitswap agent.

        .. code-block:: python

            >>> c.bitswap_stat()
            {'BlocksReceived': 96,
             'DupBlksReceived': 73,
             'DupDataReceived': 2560601,
             'ProviderBufLen': 0,
             'Peers': [
                'QmNZFQRxt9RMNm2VVtuV2Qx7q69bcMWRVXmr5CEkJEgJJP',
                'QmNfCubGpwYZAQxX8LQDsYgB48C4GbfZHuYdexpX9mbNyT',
                'QmNfnZ8SCs3jAtNPc8kf3WJqJqSoX7wsX7VqkLdEYMao4u',
                …
             ],
             'Wantlist': [
                'QmeV6C6XVt1wf7V7as7Yak3mxPma8jzpqyhtRtCvpKcfBb',
                'QmdCWFLDXqgdWQY9kVubbEHBbkieKd3uo7MtCm7nTZZE9K',
                'QmVQ1XvYGF19X4eJqz1s7FJYJqAxFC4oqh3vWJJEXn66cp'
             ]
            }

        Returns
        -------
            dict : Statistics, peers and wanted blocks
        """
        return self._client.request('/bitswap/stat', decoder='json', **kwargs)

    def bitswap_unwant(self, key, **kwargs):
        """
        Remove a given block from wantlist.

        Parameters
        ----------
        key : str
            Key to remove from wantlist.
        """
        args = (key,)
        return self._client.request('/bitswap/unwant', args, **kwargs)

    def object_data(self, multihash, **kwargs):
        r"""Returns the raw bytes in an IPFS object.

        .. code-block:: python

            >>> c.object_data('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D')
            b'\x08\x01'

        Parameters
        ----------
        multihash : str
            Key of the object to retrieve, in base58-encoded multihash format

        Returns
        -------
            str : Raw object data
        """
        args = (multihash,)
        return self._client.request('/object/data', args, **kwargs)

    def object_new(self, template=None, **kwargs):
        """Creates a new object from an IPFS template.

        By default this creates and returns a new empty merkledag node, but you
        may pass an optional template argument to create a preformatted node.

        .. code-block:: python

            >>> c.object_new()
            {'Hash': 'QmdfTbBqBPQ7VNxZEYEj14VmRuZBkqFbiwReogJgS1zR1n'}

        Parameters
        ----------
        template : str
            Blueprints from which to construct the new object. Possible values:

             * ``"unixfs-dir"``
             * ``None``

        Returns
        -------
            dict : Object hash
        """
        args = (template,) if template is not None else ()
        return self._client.request('/object/new', args,
                                    decoder='json', **kwargs)

    def object_links(self, multihash, **kwargs):
        """Returns the links pointed to by the specified object.

        .. code-block:: python

            >>> c.object_links('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDx … ca7D')
            {'Hash': 'QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D',
             'Links': [
                {'Hash': 'Qmd2xkBfEwEs9oMTk77A6jrsgurpF3ugXSg7dtPNFkcNMV',
                 'Name': 'Makefile',          'Size': 174},
                {'Hash': 'QmeKozNssnkJ4NcyRidYgDY2jfRZqVEoRGfipkgath71bX',
                 'Name': 'example',           'Size': 1474},
                {'Hash': 'QmZAL3oHMQYqsV61tGvoAVtQLs1WzRe1zkkamv9qxqnDuK',
                 'Name': 'home',              'Size': 3947},
                {'Hash': 'QmZNPyKVriMsZwJSNXeQtVQSNU4v4KEKGUQaMT61LPahso',
                 'Name': 'lib',               'Size': 268261},
                {'Hash': 'QmSY8RfVntt3VdxWppv9w5hWgNrE31uctgTiYwKir8eXJY',
                 'Name': 'published-version', 'Size': 55}]}

        Parameters
        ----------
        multihash : str
            Key of the object to retrieve, in base58-encoded multihash format

        Returns
        -------
            dict : Object hash and merkedag links
        """
        args = (multihash,)
        return self._client.request('/object/links', args,
                                    decoder='json', **kwargs)

    def object_get(self, multihash, **kwargs):
        """Get and serialize the DAG node named by multihash.

        .. code-block:: python

            >>> c.object_get('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D')
            {'Data': '\x08\x01',
             'Links': [
                {'Hash': 'Qmd2xkBfEwEs9oMTk77A6jrsgurpF3ugXSg7dtPNFkcNMV',
                 'Name': 'Makefile',          'Size': 174},
                {'Hash': 'QmeKozNssnkJ4NcyRidYgDY2jfRZqVEoRGfipkgath71bX',
                 'Name': 'example',           'Size': 1474},
                {'Hash': 'QmZAL3oHMQYqsV61tGvoAVtQLs1WzRe1zkkamv9qxqnDuK',
                 'Name': 'home',              'Size': 3947},
                {'Hash': 'QmZNPyKVriMsZwJSNXeQtVQSNU4v4KEKGUQaMT61LPahso',
                 'Name': 'lib',               'Size': 268261},
                {'Hash': 'QmSY8RfVntt3VdxWppv9w5hWgNrE31uctgTiYwKir8eXJY',
                 'Name': 'published-version', 'Size': 55}]}

        Parameters
        ----------
        multihash : str
            Key of the object to retrieve, in base58-encoded multihash format

        Returns
        -------
            dict : Object data and links
        """
        args = (multihash,)
        return self._client.request('/object/get', args,
                                    decoder='json', **kwargs)

    def object_put(self, file, **kwargs):
        """Stores input as a DAG object and returns its key.

        .. code-block:: python

            >>> c.object_put(io.BytesIO(b'''
            ...       {
            ...           "Data": "another",
            ...           "Links": [ {
            ...               "Name": "some link",
            ...               "Hash": "QmXg9Pp2ytZ14xgmQjYEiHjVjMFXzCV … R39V",
            ...               "Size": 8
            ...           } ]
            ...       }'''))
            {'Hash': 'QmZZmY4KCu9r3e7M2Pcn46Fc5qbn6NpzaAGaYb22kbfTqm',
             'Links': [
                {'Hash': 'QmXg9Pp2ytZ14xgmQjYEiHjVjMFXzCVVEcRTWJBmLgR39V',
                 'Size': 8, 'Name': 'some link'}
             ]
            }

        Parameters
        ----------
        file : io.RawIOBase
            (JSON) object from which the DAG object will be created

        Returns
        -------
            dict : Hash and links of the created DAG object

                   See :meth:`~ipfsapi.Object.object_links`
        """
        body, headers = multipart.stream_files(file, self.chunk_size)
        return self._client.request('/object/put', decoder='json',
                                    data=body, headers=headers, **kwargs)

    def object_stat(self, multihash, **kwargs):
        """Get stats for the DAG node named by multihash.

        .. code-block:: python

            >>> c.object_stat('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D')
            {'LinksSize': 256, 'NumLinks': 5,
             'Hash': 'QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D',
             'BlockSize': 258, 'CumulativeSize': 274169, 'DataSize': 2}

        Parameters
        ----------
        multihash : str
            Key of the object to retrieve, in base58-encoded multihash format

        Returns
        -------
            dict
        """
        args = (multihash,)
        return self._client.request('/object/stat', args,
                                    decoder='json', **kwargs)

    def object_patch_append_data(self, multihash, new_data, **kwargs):
        """Creates a new merkledag object based on an existing one.

        The new object will have the provided data appended to it,
        and will thus have a new Hash.

        .. code-block:: python

            >>> c.object_patch_append_data("QmZZmY … fTqm", io.BytesIO(b"bla"))
            {'Hash': 'QmR79zQQj2aDfnrNgczUhvf2qWapEfQ82YQRt3QjrbhSb2'}

        Parameters
        ----------
        multihash : str
            The hash of an ipfs object to modify
        new_data : io.RawIOBase
            The data to append to the object's data section

        Returns
        -------
            dict : Hash of new object
        """
        args = (multihash,)
        body, headers = multipart.stream_files(new_data, self.chunk_size)
        return self._client.request('/object/patch/append-data', args,
                                    decoder='json',
                                    data=body, headers=headers, **kwargs)

    def object_patch_add_link(self, root, name, ref, create=False, **kwargs):
        """Creates a new merkledag object based on an existing one.

        The new object will have a link to the provided object.

        .. code-block:: python

            >>> c.object_patch_add_link(
            ...     'QmR79zQQj2aDfnrNgczUhvf2qWapEfQ82YQRt3QjrbhSb2',
            ...     'Johnny',
            ...     'QmR79zQQj2aDfnrNgczUhvf2qWapEfQ82YQRt3QjrbhSb2'
            ... )
            {'Hash': 'QmNtXbF3AjAk59gQKRgEdVabHcSsiPUnJwHnZKyj2x8Z3k'}

        Parameters
        ----------
        root : str
            IPFS hash for the object being modified
        name : str
            name for the new link
        ref : str
            IPFS hash for the object being linked to
        create : bool
            Create intermediary nodes

        Returns
        -------
            dict : Hash of new object
        """
        kwargs.setdefault("opts", {"create": create})

        args = ((root, name, ref),)
        return self._client.request('/object/patch/add-link', args,
                                    decoder='json', **kwargs)

    def object_patch_rm_link(self, root, link, **kwargs):
        """Creates a new merkledag object based on an existing one.

        The new object will lack a link to the specified object.

        .. code-block:: python

            >>> c.object_patch_rm_link(
            ...     'QmNtXbF3AjAk59gQKRgEdVabHcSsiPUnJwHnZKyj2x8Z3k',
            ...     'Johnny'
            ... )
            {'Hash': 'QmR79zQQj2aDfnrNgczUhvf2qWapEfQ82YQRt3QjrbhSb2'}

        Parameters
        ----------
        root : str
            IPFS hash of the object to modify
        link : str
            name of the link to remove

        Returns
        -------
            dict : Hash of new object
        """
        args = ((root, link),)
        return self._client.request('/object/patch/rm-link', args,
                                    decoder='json', **kwargs)

    def object_patch_set_data(self, root, data, **kwargs):
        """Creates a new merkledag object based on an existing one.

        The new object will have the same links as the old object but
        with the provided data instead of the old object's data contents.

        .. code-block:: python

            >>> c.object_patch_set_data(
            ...     'QmNtXbF3AjAk59gQKRgEdVabHcSsiPUnJwHnZKyj2x8Z3k',
            ...     io.BytesIO(b'bla')
            ... )
            {'Hash': 'QmSw3k2qkv4ZPsbu9DVEJaTMszAQWNgM1FTFYpfZeNQWrd'}

        Parameters
        ----------
        root : str
            IPFS hash of the object to modify
        data : io.RawIOBase
            The new data to store in root

        Returns
        -------
            dict : Hash of new object
        """
        args = (root,)
        body, headers = multipart.stream_files(data, self.chunk_size)
        return self._client.request('/object/patch/set-data', args,
                                    decoder='json',
                                    data=body, headers=headers, **kwargs)

    def file_ls(self, multihash, **kwargs):
        """Lists directory contents for Unix filesystem objects.

        The result contains size information. For files, the child size is the
        total size of the file contents. For directories, the child size is the
        IPFS link size.

        The path can be a prefixless reference; in this case, it is assumed
        that it is an ``/ipfs/`` reference and not ``/ipns/``.

        .. code-block:: python

            >>> c.file_ls('QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D')
            {'Arguments': {'QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D':
                           'QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D'},
             'Objects': {
               'QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D': {
                 'Hash': 'QmTkzDwWqPbnAh5YiV5VwcTLnGdwSNsNTn2aDxdXBFca7D',
                 'Size': 0, 'Type': 'Directory',
                 'Links': [
                   {'Hash': 'Qmd2xkBfEwEs9oMTk77A6jrsgurpF3ugXSg7dtPNFkcNMV',
                    'Name': 'Makefile', 'Size': 163,    'Type': 'File'},
                   {'Hash': 'QmeKozNssnkJ4NcyRidYgDY2jfRZqVEoRGfipkgath71bX',
                    'Name': 'example',  'Size': 1463,   'Type': 'File'},
                   {'Hash': 'QmZAL3oHMQYqsV61tGvoAVtQLs1WzRe1zkkamv9qxqnDuK',
                    'Name': 'home',     'Size': 3947,   'Type': 'Directory'},
                   {'Hash': 'QmZNPyKVriMsZwJSNXeQtVQSNU4v4KEKGUQaMT61LPahso',
                    'Name': 'lib',      'Size': 268261, 'Type': 'Directory'},
                   {'Hash': 'QmSY8RfVntt3VdxWppv9w5hWgNrE31uctgTiYwKir8eXJY',
                    'Name': 'published-version',
                    'Size': 47, 'Type': 'File'}
                   ]
               }
            }}

        Parameters
        ----------
        multihash : str
            The path to the object(s) to list links from

        Returns
        -------
            dict
        """
        args = (multihash,)
        return self._client.request('/file/ls', args, decoder='json', **kwargs)

    def resolve(self, name, recursive=False, **kwargs):
        """Accepts an identifier and resolves it to the referenced item.

        There are a number of mutable name protocols that can link among
        themselves and into IPNS. For example IPNS references can (currently)
        point at an IPFS object, and DNS links can point at other DNS links,
        IPNS entries, or IPFS objects. This command accepts any of these
        identifiers.

        .. code-block:: python

            >>> c.resolve("/ipfs/QmTkzDwWqPbnAh5YiV5VwcTLnGdw … ca7D/Makefile")
            {'Path': '/ipfs/Qmd2xkBfEwEs9oMTk77A6jrsgurpF3ugXSg7dtPNFkcNMV'}
            >>> c.resolve("/ipns/ipfs.io")
            {'Path': '/ipfs/QmTzQ1JRkWErjk39mryYw2WVaphAZNAREyMchXzYQ7c15n'}

        Parameters
        ----------
        name : str
            The name to resolve
        recursive : bool
            Resolve until the result is an IPFS name

        Returns
        -------
            dict : IPFS path of resource
        """
        kwargs.setdefault("opts", {"recursive": recursive})

        args = (name,)
        return self._client.request('/resolve', args, decoder='json', **kwargs)

    def pubsub_ls(self, **kwargs):
        return self._client.request('/pubsub/ls', decoder='json', **kwargs)

    def pubsub_peers(self, topic=None, **kwargs):
        if topic is None:
            args = []
        else:
            args = [topic]
        return self._client.request(
            '/pubsub/peers', args, decoder='json', **kwargs)

    def pubsub_pub(self, topic, payload, **kwargs):
        args = (topic, payload)
        return self._client.request(
            '/pubsub/pub', args, decoder='json', **kwargs)

    def pubsub_sub(self, topic, discover=False, **kwargs):
        args = (topic, discover)
        return self._client.request(
            '/pubsub/sub', args, decoder='json', **kwargs)

    def key_list(self, **kwargs):
        """Returns a list of generated public keys that can be used with name_publish

        .. code-block:: python

            >>> c.key_list()
            [{'Name': 'self',
              'Id': 'QmQf22bZar3WKmojipms22PkXH1MZGmvsqzQtuSvQE3uhm'},
             {'Name': 'example_key_name',
              'Id': 'QmQLaT5ZrCfSkXTH6rUKtVidcxj8jrW3X2h75Lug1AV7g8'}
            ]

        Returns
        -------
            list : List of dictionaries with Names and Ids of public keys.
        """
        return self._client.request('/key/list', decoder='json', **kwargs)

    def key_gen(self, key_name, type, size=2048, **kwargs):
        """Adds a new public key that can be used for name_publish.

        .. code-block:: python

            >>> c.key_gen('example_key_name')
            {'Name': 'example_key_name',
             'Id': 'QmQLaT5ZrCfSkXTH6rUKtVidcxj8jrW3X2h75Lug1AV7g8'}

        Parameters
        ----------
        key_name : str
            Name of the new Key to be generated. Used to reference the Keys.
        type : str
            Type of key to generate. The current possible keys types are:

             * ``"rsa"``
             * ``"ed25519"``
        size : int
            Bitsize of key to generate

        Returns
        -------
            dict : Key name and Key Id
        """

        opts = {"type": type, "size": size}
        kwargs.setdefault("opts", opts)
        args = (key_name,)

        return self._client.request('/key/gen', args,
                                    decoder='json', **kwargs)

    def key_rm(self, key_name, *key_names, **kwargs):
        """Remove a keypair

        .. code-block:: python

            >>> c.key_rm("bla")
            {"Keys": [
                {"Name": "bla",
                 "Id": "QmfJpR6paB6h891y7SYXGe6gapyNgepBeAYMbyejWA4FWA"}
            ]}

        Parameters
        ----------
        key_name : str
            Name of the key(s) to remove.

        Returns
        -------
            dict : List of key names and IDs that have been removed
        """
        args = (key_name,) + key_names
        return self._client.request('/key/rm', args, decoder='json', **kwargs)

    def key_rename(self, key_name, new_key_name, **kwargs):
        """Rename a keypair

        .. code-block:: python

            >>> c.key_rename("bla", "personal")
            {"Was": "bla",
             "Now": "personal",
             "Id": "QmeyrRNxXaasZaoDXcCZgryoBCga9shaHQ4suHAYXbNZF3",
             "Overwrite": False}

        Parameters
        ----------
        key_name : str
            Current name of the key to rename
        new_key_name : str
            New name of the key

        Returns
        -------
            dict : List of key names and IDs that have been removed
        """
        args = (key_name, new_key_name)
        return self._client.request('/key/rename', args, decoder='json',
                                    **kwargs)

    def name_publish(self, ipfs_path, resolve=True, lifetime="24h", ttl=None,
                     key=None, **kwargs):
        """Publishes an object to IPNS.

        IPNS is a PKI namespace, where names are the hashes of public keys, and
        the private key enables publishing new (signed) values. In publish, the
        default value of *name* is your own identity public key.

        .. code-block:: python

            >>> c.name_publish('/ipfs/QmfZY61ukoQuCX8e5Pt7v8pRfhkyxwZK … GZ5d')
            {'Value': '/ipfs/QmfZY61ukoQuCX8e5Pt7v8pRfhkyxwZKZMTodAtmvyGZ5d',
             'Name': 'QmVgNoP89mzpgEAAqK8owYoDEyB97MkcGvoWZir8otE9Uc'}

        Parameters
        ----------
        ipfs_path : str
            IPFS path of the object to be published
        resolve : bool
            Resolve given path before publishing
        lifetime : str
            Time duration that the record will be valid for

            Accepts durations such as ``"300s"``, ``"1.5h"`` or ``"2h45m"``.
            Valid units are:

             * ``"ns"``
             * ``"us"`` (or ``"µs"``)
             * ``"ms"``
             * ``"s"``
             * ``"m"``
             * ``"h"``
        ttl : int
            Time duration this record should be cached for
        key : string
             Name of the key to be used, as listed by 'ipfs key list'.

        Returns
        -------
            dict : IPNS hash and the IPFS path it points at
        """
        opts = {"lifetime": lifetime, "resolve": resolve}
        if ttl:
            opts["ttl"] = ttl
        if key:
            opts["key"] = key
        kwargs.setdefault("opts", opts)

        args = (ipfs_path,)
        return self._client.request('/name/publish', args,
                                    decoder='json', **kwargs)

    def name_resolve(self, name=None, recursive=False,
                     nocache=False, **kwargs):
        """Gets the value currently published at an IPNS name.

        IPNS is a PKI namespace, where names are the hashes of public keys, and
        the private key enables publishing new (signed) values. In resolve, the
        default value of ``name`` is your own identity public key.

        .. code-block:: python

            >>> c.name_resolve()
            {'Path': '/ipfs/QmfZY61ukoQuCX8e5Pt7v8pRfhkyxwZKZMTodAtmvyGZ5d'}

        Parameters
        ----------
        name : str
            The IPNS name to resolve (defaults to the connected node)
        recursive : bool
            Resolve until the result is not an IPFS name (default: false)
        nocache : bool
            Do not use cached entries (default: false)

        Returns
        -------
            dict : The IPFS path the IPNS hash points at
        """
        kwargs.setdefault("opts", {"recursive": recursive,
                                   "nocache": nocache})
        args = (name,) if name is not None else ()
        return self._client.request('/name/resolve', args,
                                    decoder='json', **kwargs)

    def dns(self, domain_name, recursive=False, **kwargs):
        """Resolves DNS links to the referenced object.

        Multihashes are hard to remember, but domain names are usually easy to
        remember. To create memorable aliases for multihashes, DNS TXT records
        can point to other DNS links, IPFS objects, IPNS keys, etc.
        This command resolves those links to the referenced object.

        For example, with this DNS TXT record::

            >>> import dns.resolver
            >>> a = dns.resolver.query("ipfs.io", "TXT")
            >>> a.response.answer[0].items[0].to_text()
            '"dnslink=/ipfs/QmTzQ1JRkWErjk39mryYw2WVaphAZNAREyMchXzYQ7c15n"'

        The resolver will give::

            >>> c.dns("ipfs.io")
            {'Path': '/ipfs/QmTzQ1JRkWErjk39mryYw2WVaphAZNAREyMchXzYQ7c15n'}

        Parameters
        ----------
        domain_name : str
           The domain-name name to resolve
        recursive : bool
            Resolve until the name is not a DNS link

        Returns
        -------
            dict : Resource were a DNS entry points to
        """
        kwargs.setdefault("opts", {"recursive": recursive})

        args = (domain_name,)
        return self._client.request('/dns', args, decoder='json', **kwargs)

    def pin_add(self, path, *paths, **kwargs):
        """Pins objects to local storage.

        Stores an IPFS object(s) from a given path locally to disk.

        .. code-block:: python

            >>> c.pin_add("QmfZY61ukoQuCX8e5Pt7v8pRfhkyxwZKZMTodAtmvyGZ5d")
            {'Pins': ['QmfZY61ukoQuCX8e5Pt7v8pRfhkyxwZKZMTodAtmvyGZ5d']}

        Parameters
        ----------
        path : str
            Path to object(s) to be pinned
        recursive : bool
            Recursively unpin the object linked to by the specified object(s)

        Returns
        -------
            dict : List of IPFS objects that have been pinned
        """
        #PY2: No support for kw-only parameters after glob parameters
        if "recursive" in kwargs:
            kwargs.setdefault("opts", {"recursive": kwargs.pop("recursive")})

        args = (path,) + paths
        return self._client.request('/pin/add', args, decoder='json', **kwargs)

    def pin_rm(self, path, *paths, **kwargs):
        """Removes a pinned object from local storage.

        Removes the pin from the given object allowing it to be garbage
        collected if needed.

        .. code-block:: python

            >>> c.pin_rm('QmfZY61ukoQuCX8e5Pt7v8pRfhkyxwZKZMTodAtmvyGZ5d')
            {'Pins': ['QmfZY61ukoQuCX8e5Pt7v8pRfhkyxwZKZMTodAtmvyGZ5d']}

        Parameters
        ----------
        path : str
            Path to object(s) to be unpinned
        recursive : bool
            Recursively unpin the object linked to by the specified object(s)

        Returns
        -------
            dict : List of IPFS objects that have been unpinned
        """
        #PY2: No support for kw-only parameters after glob parameters
        if "recursive" in kwargs:
            kwargs.setdefault("opts", {"recursive": kwargs["recursive"]})
            del kwargs["recursive"]

        args = (path,) + paths
        return self._client.request('/pin/rm', args, decoder='json', **kwargs)

    def pin_ls(self, type="all", **kwargs):
        """Lists objects pinned to local storage.

        By default, all pinned objects are returned, but the ``type`` flag or
        arguments can restrict that to a specific pin type or to some specific
        objects respectively.

        .. code-block:: python

            >>> c.pin_ls()
            {'Keys': {
                'QmNNPMA1eGUbKxeph6yqV8ZmRkdVat … YMuz': {'Type': 'recursive'},
                'QmNPZUCeSN5458Uwny8mXSWubjjr6J … kP5e': {'Type': 'recursive'},
                'QmNg5zWpRMxzRAVg7FTQ3tUxVbKj8E … gHPz': {'Type': 'indirect'},
                …
                'QmNiuVapnYCrLjxyweHeuk6Xdqfvts … wCCe': {'Type': 'indirect'}}}

        Parameters
        ----------
        type : "str"
            The type of pinned keys to list. Can be:

             * ``"direct"``
             * ``"indirect"``
             * ``"recursive"``
             * ``"all"``

        Returns
        -------
            dict : Hashes of pinned IPFS objects and why they are pinned
        """
        kwargs.setdefault("opts", {"type": type})

        return self._client.request('/pin/ls', decoder='json', **kwargs)

    def pin_update(self, from_path, to_path, **kwargs):
        """Replaces one pin with another.

        Updates one pin to another, making sure that all objects in the new pin
        are local. Then removes the old pin. This is an optimized version of
        using first using :meth:`~ipfsapi.Client.pin_add` to add a new pin
        for an object and then using :meth:`~ipfsapi.Client.pin_rm` to remove
        the pin for the old object.

        .. code-block:: python

            >>> c.pin_update("QmXMqez83NU77ifmcPs5CkNRTMQksBLkyfBf4H5g1NZ52P",
            ...              "QmUykHAi1aSjMzHw3KmBoJjqRUQYNkFXm8K1y7ZsJxpfPH")
            {"Pins": ["/ipfs/QmXMqez83NU77ifmcPs5CkNRTMQksBLkyfBf4H5g1NZ52P",
                      "/ipfs/QmUykHAi1aSjMzHw3KmBoJjqRUQYNkFXm8K1y7ZsJxpfPH"]}

        Parameters
        ----------
        from_path : str
            Path to the old object
        to_path : str
            Path to the new object to be pinned
        unpin : bool
            Should the pin of the old object be removed? (Default: ``True``)

        Returns
        -------
            dict : List of IPFS objects affected by the pinning operation
        """
        #PY2: No support for kw-only parameters after glob parameters
        if "unpin" in kwargs:
            kwargs.setdefault("opts", {"unpin": kwargs["unpin"]})
            del kwargs["unpin"]

        args = (from_path, to_path)
        return self._client.request('/pin/update', args, decoder='json',
                                    **kwargs)

    def pin_verify(self, path, *paths, **kwargs):
        """Verify that recursive pins are complete.

        Scan the repo for pinned object graphs and check their integrity.
        Issues will be reported back with a helpful human-readable error
        message to aid in error recovery. This is useful to help recover
        from datastore corruptions (such as when accidentally deleting
        files added using the filestore backend).

        .. code-block:: python

            >>> for item in c.pin_verify("QmNuvmuFeeWWpx…wTTZ", verbose=True):
            ...     print(item)
            ...
            {"Cid":"QmVkNdzCBukBRdpyFiKPyL2R15qPExMr9rV9RFV2kf9eeV","Ok":True}
            {"Cid":"QmbPzQruAEFjUU3gQfupns6b8USr8VrD9H71GrqGDXQSxm","Ok":True}
            {"Cid":"Qmcns1nUvbeWiecdGDPw8JxWeUfxCV8JKhTfgzs3F8JM4P","Ok":True}
            …

        Parameters
        ----------
        path : str
            Path to object(s) to be checked
        verbose : bool
            Also report status of items that were OK? (Default: ``False``)

        Returns
        -------
            iterable
        """
        #PY2: No support for kw-only parameters after glob parameters
        if "verbose" in kwargs:
            kwargs.setdefault("opts", {"verbose": kwargs["verbose"]})
            del kwargs["verbose"]

        args = (path,) + paths
        return self._client.request('/pin/verify', args, decoder='json',
                                    stream=True, **kwargs)

    def repo_gc(self, **kwargs):
        """Removes stored objects that are not pinned from the repo.

        .. code-block:: python

            >>> c.repo_gc()
            [{'Key': 'QmNPXDC6wTXVmZ9Uoc8X1oqxRRJr4f1sDuyQuwaHG2mpW2'},
             {'Key': 'QmNtXbF3AjAk59gQKRgEdVabHcSsiPUnJwHnZKyj2x8Z3k'},
             {'Key': 'QmRVBnxUCsD57ic5FksKYadtyUbMsyo9KYQKKELajqAp4q'},
             …
             {'Key': 'QmYp4TeCurXrhsxnzt5wqLqqUz8ZRg5zsc7GuUrUSDtwzP'}]

        Performs a garbage collection sweep of the local set of
        stored objects and remove ones that are not pinned in order
        to reclaim hard disk space. Returns the hashes of all collected
        objects.

        Returns
        -------
            dict : List of IPFS objects that have been removed
        """
        return self._client.request('/repo/gc', decoder='json', **kwargs)

    def repo_stat(self, **kwargs):
        """Displays the repo's status.

        Returns the number of objects in the repo and the repo's size,
        version, and path.

        .. code-block:: python

            >>> c.repo_stat()
            {'NumObjects': 354,
             'RepoPath': '…/.local/share/ipfs',
             'Version': 'fs-repo@4',
             'RepoSize': 13789310}

        Returns
        -------
            dict : General information about the IPFS file repository

        +------------+-------------------------------------------------+
        | NumObjects | Number of objects in the local repo.            |
        +------------+-------------------------------------------------+
        | RepoPath   | The path to the repo being currently used.      |
        +------------+-------------------------------------------------+
        | RepoSize   | Size in bytes that the repo is currently using. |
        +------------+-------------------------------------------------+
        | Version    | The repo version.                               |
        +------------+-------------------------------------------------+
        """
        return self._client.request('/repo/stat', decoder='json', **kwargs)

    def id(self, peer=None, **kwargs):
        """Shows IPFS Node ID info.

        Returns the PublicKey, ProtocolVersion, ID, AgentVersion and
        Addresses of the connected daemon or some other node.

        .. code-block:: python

            >>> c.id()
            {'ID': 'QmVgNoP89mzpgEAAqK8owYoDEyB97MkcGvoWZir8otE9Uc',
            'PublicKey': 'CAASpgIwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggE … BAAE=',
            'AgentVersion': 'go-libp2p/3.3.4',
            'ProtocolVersion': 'ipfs/0.1.0',
            'Addresses': [
                '/ip4/127.0.0.1/tcp/4001/ipfs/QmVgNoP89mzpgEAAqK8owYo … E9Uc',
                '/ip4/10.1.0.172/tcp/4001/ipfs/QmVgNoP89mzpgEAAqK8owY … E9Uc',
                '/ip4/172.18.0.1/tcp/4001/ipfs/QmVgNoP89mzpgEAAqK8owY … E9Uc',
                '/ip6/::1/tcp/4001/ipfs/QmVgNoP89mzpgEAAqK8owYoDEyB97 … E9Uc',
                '/ip6/fccc:7904:b05b:a579:957b:deef:f066:cad9/tcp/400 … E9Uc',
                '/ip6/fd56:1966:efd8::212/tcp/4001/ipfs/QmVgNoP89mzpg … E9Uc',
                '/ip6/fd56:1966:efd8:0:def1:34d0:773:48f/tcp/4001/ipf … E9Uc',
                '/ip6/2001:db8:1::1/tcp/4001/ipfs/QmVgNoP89mzpgEAAqK8 … E9Uc',
                '/ip4/77.116.233.54/tcp/4001/ipfs/QmVgNoP89mzpgEAAqK8 … E9Uc',
                '/ip4/77.116.233.54/tcp/10842/ipfs/QmVgNoP89mzpgEAAqK … E9Uc']}

        Parameters
        ----------
        peer : str
            Peer.ID of the node to look up (local node if ``None``)

        Returns
        -------
            dict : Information about the IPFS node
        """
        args = (peer,) if peer is not None else ()
        return self._client.request('/id', args, decoder='json', **kwargs)

    def bootstrap(self, **kwargs):
        """Compatiblity alias for :meth:`~ipfsapi.Client.bootstrap_list`."""
        self.bootstrap_list(**kwargs)

    def bootstrap_list(self, **kwargs):
        """Returns the addresses of peers used during initial discovery of the
        IPFS network.

        Peers are output in the format ``<multiaddr>/<peerID>``.

        .. code-block:: python

            >>> c.bootstrap_list()
            {'Peers': [
                '/ip4/104.131.131.82/tcp/4001/ipfs/QmaCpDMGvV2BGHeYER … uvuJ',
                '/ip4/104.236.176.52/tcp/4001/ipfs/QmSoLnSGccFuZQJzRa … ca9z',
                '/ip4/104.236.179.241/tcp/4001/ipfs/QmSoLPppuBtQSGwKD … KrGM',
                …
                '/ip4/178.62.61.185/tcp/4001/ipfs/QmSoLMeWqB7YGVLJN3p … QBU3']}

        Returns
        -------
            dict : List of known bootstrap peers
        """
        return self._client.request('/bootstrap', decoder='json', **kwargs)

    def bootstrap_add(self, peer, *peers, **kwargs):
        """Adds peers to the bootstrap list.

        Parameters
        ----------
        peer : str
            IPFS MultiAddr of a peer to add to the list

        Returns
        -------
            dict
        """
        args = (peer,) + peers
        return self._client.request('/bootstrap/add', args,
                                    decoder='json', **kwargs)

    def bootstrap_rm(self, peer, *peers, **kwargs):
        """Removes peers from the bootstrap list.

        Parameters
        ----------
        peer : str
            IPFS MultiAddr of a peer to remove from the list

        Returns
        -------
            dict
        """
        args = (peer,) + peers
        return self._client.request('/bootstrap/rm', args,
                                    decoder='json', **kwargs)

    def swarm_peers(self, **kwargs):
        """Returns the addresses & IDs of currently connected peers.

        .. code-block:: python

            >>> c.swarm_peers()
            {'Strings': [
                '/ip4/101.201.40.124/tcp/40001/ipfs/QmZDYAhmMDtnoC6XZ … kPZc',
                '/ip4/104.131.131.82/tcp/4001/ipfs/QmaCpDMGvV2BGHeYER … uvuJ',
                '/ip4/104.223.59.174/tcp/4001/ipfs/QmeWdgoZezpdHz1PX8 … 1jB6',
                …
                '/ip6/fce3: … :f140/tcp/43901/ipfs/QmSoLnSGccFuZQJzRa … ca9z']}

        Returns
        -------
            dict : List of multiaddrs of currently connected peers
        """
        return self._client.request('/swarm/peers', decoder='json', **kwargs)

    def swarm_addrs(self, **kwargs):
        """Returns the addresses of currently connected peers by peer id.

        .. code-block:: python

            >>> pprint(c.swarm_addrs())
            {'Addrs': {
                'QmNMVHJTSZHTWMWBbmBrQgkA1hZPWYuVJx2DpSGESWW6Kn': [
                    '/ip4/10.1.0.1/tcp/4001',
                    '/ip4/127.0.0.1/tcp/4001',
                    '/ip4/51.254.25.16/tcp/4001',
                    '/ip6/2001:41d0:b:587:3cae:6eff:fe40:94d8/tcp/4001',
                    '/ip6/2001:470:7812:1045::1/tcp/4001',
                    '/ip6/::1/tcp/4001',
                    '/ip6/fc02:2735:e595:bb70:8ffc:5293:8af8:c4b7/tcp/4001',
                    '/ip6/fd00:7374:6172:100::1/tcp/4001',
                    '/ip6/fd20:f8be:a41:0:c495:aff:fe7e:44ee/tcp/4001',
                    '/ip6/fd20:f8be:a41::953/tcp/4001'],
                'QmNQsK1Tnhe2Uh2t9s49MJjrz7wgPHj4VyrZzjRe8dj7KQ': [
                    '/ip4/10.16.0.5/tcp/4001',
                    '/ip4/127.0.0.1/tcp/4001',
                    '/ip4/172.17.0.1/tcp/4001',
                    '/ip4/178.62.107.36/tcp/4001',
                    '/ip6/::1/tcp/4001'],
                …
            }}

        Returns
        -------
            dict : Multiaddrs of peers by peer id
        """
        return self._client.request('/swarm/addrs', decoder='json', **kwargs)

    def swarm_connect(self, address, *addresses, **kwargs):
        """Opens a connection to a given address.

        This will open a new direct connection to a peer address. The address
        format is an IPFS multiaddr::

            /ip4/104.131.131.82/tcp/4001/ipfs/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ

        .. code-block:: python

            >>> c.swarm_connect("/ip4/104.131.131.82/tcp/4001/ipfs/Qma … uvuJ")
            {'Strings': ['connect QmaCpDMGvV2BGHeYERUEnRQAwe3 … uvuJ success']}

        Parameters
        ----------
        address : str
            Address of peer to connect to

        Returns
        -------
            dict : Textual connection status report
        """
        args = (address,) + addresses
        return self._client.request('/swarm/connect', args,
                                    decoder='json', **kwargs)

    def swarm_disconnect(self, address, *addresses, **kwargs):
        """Closes the connection to a given address.

        This will close a connection to a peer address. The address format is
        an IPFS multiaddr::

            /ip4/104.131.131.82/tcp/4001/ipfs/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ

        The disconnect is not permanent; if IPFS needs to talk to that address
        later, it will reconnect.

        .. code-block:: python

            >>> c.swarm_disconnect("/ip4/104.131.131.82/tcp/4001/ipfs/Qm … uJ")
            {'Strings': ['disconnect QmaCpDMGvV2BGHeYERUEnRQA … uvuJ success']}

        Parameters
        ----------
        address : str
            Address of peer to disconnect from

        Returns
        -------
            dict : Textual connection status report
        """
        args = (address,) + addresses
        return self._client.request('/swarm/disconnect', args,
                                    decoder='json', **kwargs)

    def swarm_filters_add(self, address, *addresses, **kwargs):
        """Adds a given multiaddr filter to the filter list.

        This will add an address filter to the daemons swarm. Filters applied
        this way will not persist daemon reboots, to achieve that, add your
        filters to the configuration file.

        .. code-block:: python

            >>> c.swarm_filters_add("/ip4/192.168.0.0/ipcidr/16")
            {'Strings': ['/ip4/192.168.0.0/ipcidr/16']}

        Parameters
        ----------
        address : str
            Multiaddr to filter

        Returns
        -------
            dict : List of swarm filters added
        """
        args = (address,) + addresses
        return self._client.request('/swarm/filters/add', args,
                                    decoder='json', **kwargs)

    def swarm_filters_rm(self, address, *addresses, **kwargs):
        """Removes a given multiaddr filter from the filter list.

        This will remove an address filter from the daemons swarm. Filters
        removed this way will not persist daemon reboots, to achieve that,
        remove your filters from the configuration file.

        .. code-block:: python

            >>> c.swarm_filters_rm("/ip4/192.168.0.0/ipcidr/16")
            {'Strings': ['/ip4/192.168.0.0/ipcidr/16']}

        Parameters
        ----------
        address : str
            Multiaddr filter to remove

        Returns
        -------
            dict : List of swarm filters removed
        """
        args = (address,) + addresses
        return self._client.request('/swarm/filters/rm', args,
                                    decoder='json', **kwargs)

    def dht_query(self, peer_id, *peer_ids, **kwargs):
        """Finds the closest Peer IDs to a given Peer ID by querying the DHT.

        .. code-block:: python

            >>> c.dht_query("/ip4/104.131.131.82/tcp/4001/ipfs/QmaCpDM … uvuJ")
            [{'ID': 'QmPkFbxAQ7DeKD5VGSh9HQrdS574pyNzDmxJeGrRJxoucF',
              'Extra': '', 'Type': 2, 'Responses': None},
             {'ID': 'QmR1MhHVLJSLt9ZthsNNhudb1ny1WdhY4FPW21ZYFWec4f',
              'Extra': '', 'Type': 2, 'Responses': None},
             {'ID': 'Qmcwx1K5aVme45ab6NYWb52K2TFBeABgCLccC7ntUeDsAs',
              'Extra': '', 'Type': 2, 'Responses': None},
             …
             {'ID': 'QmYYy8L3YD1nsF4xtt4xmsc14yqvAAnKksjo3F3iZs5jPv',
              'Extra': '', 'Type': 1, 'Responses': []}]

        Parameters
        ----------
        peer_id : str
            The peerID to run the query against

        Returns
        -------
            dict : List of peers IDs
        """
        args = (peer_id,) + peer_ids
        return self._client.request('/dht/query', args,
                                    decoder='json', **kwargs)

    def dht_findprovs(self, multihash, *multihashes, **kwargs):
        """Finds peers in the DHT that can provide a specific value.

        .. code-block:: python

            >>> c.dht_findprovs("QmNPXDC6wTXVmZ9Uoc8X1oqxRRJr4f1sDuyQu … mpW2")
            [{'ID': 'QmaxqKpiYNr62uSFBhxJAMmEMkT6dvc3oHkrZNpH2VMTLZ',
              'Extra': '', 'Type': 6, 'Responses': None},
             {'ID': 'QmaK6Aj5WXkfnWGoWq7V8pGUYzcHPZp4jKQ5JtmRvSzQGk',
              'Extra': '', 'Type': 6, 'Responses': None},
             {'ID': 'QmdUdLu8dNvr4MVW1iWXxKoQrbG6y1vAVWPdkeGK4xppds',
              'Extra': '', 'Type': 6, 'Responses': None},
             …
             {'ID': '', 'Extra': '', 'Type': 4, 'Responses': [
                {'ID': 'QmVgNoP89mzpgEAAqK8owYoDEyB97Mk … E9Uc', 'Addrs': None}
              ]},
             {'ID': 'QmaxqKpiYNr62uSFBhxJAMmEMkT6dvc3oHkrZNpH2VMTLZ',
              'Extra': '', 'Type': 1, 'Responses': [
                {'ID': 'QmSHXfsmN3ZduwFDjeqBn1C8b1tcLkxK6yd … waXw', 'Addrs': [
                    '/ip4/127.0.0.1/tcp/4001',
                    '/ip4/172.17.0.8/tcp/4001',
                    '/ip6/::1/tcp/4001',
                    '/ip4/52.32.109.74/tcp/1028'
                  ]}
              ]}]

        Parameters
        ----------
        multihash : str
            The DHT key to find providers for

        Returns
        -------
            dict : List of provider Peer IDs
        """
        args = (multihash,) + multihashes
        return self._client.request('/dht/findprovs', args,
                                    decoder='json', **kwargs)

    def dht_findpeer(self, peer_id, *peer_ids, **kwargs):
        """Queries the DHT for all of the associated multiaddresses.

        .. code-block:: python

            >>> c.dht_findpeer("QmaxqKpiYNr62uSFBhxJAMmEMkT6dvc3oHkrZN … MTLZ")
            [{'ID': 'QmfVGMFrwW6AV6fTWmD6eocaTybffqAvkVLXQEFrYdk6yc',
              'Extra': '', 'Type': 6, 'Responses': None},
             {'ID': 'QmTKiUdjbRjeN9yPhNhG1X38YNuBdjeiV9JXYWzCAJ4mj5',
              'Extra': '', 'Type': 6, 'Responses': None},
             {'ID': 'QmTGkgHSsULk8p3AKTAqKixxidZQXFyF7mCURcutPqrwjQ',
              'Extra': '', 'Type': 6, 'Responses': None},
             …
             {'ID': '', 'Extra': '', 'Type': 2,
              'Responses': [
                {'ID': 'QmaxqKpiYNr62uSFBhxJAMmEMkT6dvc3oHkrZNpH2VMTLZ',
                 'Addrs': [
                    '/ip4/10.9.8.1/tcp/4001',
                    '/ip6/::1/tcp/4001',
                    '/ip4/164.132.197.107/tcp/4001',
                    '/ip4/127.0.0.1/tcp/4001']}
              ]}]

        Parameters
        ----------
        peer_id : str
            The ID of the peer to search for

        Returns
        -------
            dict : List of multiaddrs
        """
        args = (peer_id,) + peer_ids
        return self._client.request('/dht/findpeer', args,
                                    decoder='json', **kwargs)

    def dht_get(self, key, *keys, **kwargs):
        """Queries the DHT for its best value related to given key.

        There may be several different values for a given key stored in the
        DHT; in this context *best* means the record that is most desirable.
        There is no one metric for *best*: it depends entirely on the key type.
        For IPNS, *best* is the record that is both valid and has the highest
        sequence number (freshest). Different key types may specify other rules
        for they consider to be the *best*.

        Parameters
        ----------
        key : str
            One or more keys whose values should be looked up

        Returns
        -------
            str
        """
        args = (key,) + keys
        res = self._client.request('/dht/get', args, decoder='json', **kwargs)

        if isinstance(res, dict) and "Extra" in res:
            return res["Extra"]
        else:
            for r in res:
                if "Extra" in r and len(r["Extra"]) > 0:
                    return r["Extra"]
        raise exceptions.Error("empty response from DHT")

    def dht_put(self, key, value, **kwargs):
        """Writes a key/value pair to the DHT.

        Given a key of the form ``/foo/bar`` and a value of any form, this will
        write that value to the DHT with that key.

        Keys have two parts: a keytype (foo) and the key name (bar). IPNS uses
        the ``/ipns/`` keytype, and expects the key name to be a Peer ID. IPNS
        entries are formatted with a special strucutre.

        You may only use keytypes that are supported in your ``ipfs`` binary:
        ``go-ipfs`` currently only supports the ``/ipns/`` keytype. Unless you
        have a relatively deep understanding of the key's internal structure,
        you likely want to be using the :meth:`~ipfsapi.Client.name_publish`
        instead.

        Value is arbitrary text.

        .. code-block:: python

            >>> c.dht_put("QmVgNoP89mzpgEAAqK8owYoDEyB97Mkc … E9Uc", "test123")
            [{'ID': 'QmfLy2aqbhU1RqZnGQyqHSovV8tDufLUaPfN1LNtg5CvDZ',
              'Extra': '', 'Type': 5, 'Responses': None},
             {'ID': 'QmZ5qTkNvvZ5eFq9T4dcCEK7kX8L7iysYEpvQmij9vokGE',
              'Extra': '', 'Type': 5, 'Responses': None},
             {'ID': 'QmYqa6QHCbe6eKiiW6YoThU5yBy8c3eQzpiuW22SgVWSB8',
              'Extra': '', 'Type': 6, 'Responses': None},
             …
             {'ID': 'QmP6TAKVDCziLmx9NV8QGekwtf7ZMuJnmbeHMjcfoZbRMd',
              'Extra': '', 'Type': 1, 'Responses': []}]

        Parameters
        ----------
        key : str
            A unique identifier
        value : str
            Abitrary text to associate with the input (2048 bytes or less)

        Returns
        -------
            list
        """
        args = (key, value)
        return self._client.request('/dht/put', args, decoder='json', **kwargs)

    def ping(self, peer, *peers, **kwargs):
        """Provides round-trip latency information for the routing system.

        Finds nodes via the routing system, sends pings, waits for pongs,
        and prints out round-trip latency information.

        .. code-block:: python

            >>> c.ping("QmTzQ1JRkWErjk39mryYw2WVaphAZNAREyMchXzYQ7c15n")
            [{'Success': True,  'Time': 0,
              'Text': 'Looking up peer QmTzQ1JRkWErjk39mryYw2WVaphAZN … c15n'},
             {'Success': False, 'Time': 0,
              'Text': 'Peer lookup error: routing: not found'}]

        Parameters
        ----------
        peer : str
            ID of peer to be pinged
        count : int
            Number of ping messages to send (Default: ``10``)

        Returns
        -------
            list : Progress reports from the ping
        """
        #PY2: No support for kw-only parameters after glob parameters
        if "count" in kwargs:
            kwargs.setdefault("opts", {"count": kwargs["count"]})
            del kwargs["count"]

        args = (peer,) + peers
        return self._client.request('/ping', args, decoder='json', **kwargs)

    def config(self, key, value=None, **kwargs):
        """Controls configuration variables.

        .. code-block:: python

            >>> c.config("Addresses.Gateway")
            {'Key': 'Addresses.Gateway', 'Value': '/ip4/127.0.0.1/tcp/8080'}
            >>> c.config("Addresses.Gateway", "/ip4/127.0.0.1/tcp/8081")
            {'Key': 'Addresses.Gateway', 'Value': '/ip4/127.0.0.1/tcp/8081'}

        Parameters
        ----------
        key : str
            The key of the configuration entry (e.g. "Addresses.API")
        value : dict
            The value to set the configuration entry to

        Returns
        -------
            dict : Requested/updated key and its (new) value
        """
        args = (key, value)
        return self._client.request('/config', args, decoder='json', **kwargs)

    def config_show(self, **kwargs):
        """Returns a dict containing the server's configuration.

        .. warning::

            The configuration file contains private key data that must be
            handled with care.

        .. code-block:: python

            >>> config = c.config_show()
            >>> config['Addresses']
            {'API': '/ip4/127.0.0.1/tcp/5001',
             'Gateway': '/ip4/127.0.0.1/tcp/8080',
             'Swarm': ['/ip4/0.0.0.0/tcp/4001', '/ip6/::/tcp/4001']},
            >>> config['Discovery']
            {'MDNS': {'Enabled': True, 'Interval': 10}}

        Returns
        -------
            dict : The entire IPFS daemon configuration
        """
        return self._client.request('/config/show', decoder='json', **kwargs)

    def config_replace(self, *args, **kwargs):
        """Replaces the existing config with a user-defined config.

        Make sure to back up the config file first if neccessary, as this
        operation can't be undone.
        """
        return self._client.request('/config/replace', args,
                                    decoder='json', **kwargs)

    def log_level(self, subsystem, level, **kwargs):
        r"""Changes the logging output of a running daemon.

        .. code-block:: python

            >>> c.log_level("path", "info")
            {'Message': "Changed log level of 'path' to 'info'\n"}

        Parameters
        ----------
        subsystem : str
            The subsystem logging identifier (Use ``"all"`` for all subsystems)
        level : str
            The desired logging level. Must be one of:

             * ``"debug"``
             * ``"info"``
             * ``"warning"``
             * ``"error"``
             * ``"fatal"``
             * ``"panic"``

        Returns
        -------
            dict : Status message
        """
        args = (subsystem, level)
        return self._client.request('/log/level', args,
                                    decoder='json', **kwargs)

    def log_ls(self, **kwargs):
        """Lists the logging subsystems of a running daemon.

        .. code-block:: python

            >>> c.log_ls()
            {'Strings': [
                'github.com/ipfs/go-libp2p/p2p/host', 'net/identify',
                'merkledag', 'providers', 'routing/record', 'chunk', 'mfs',
                'ipns-repub', 'flatfs', 'ping', 'mockrouter', 'dagio',
                'cmds/files', 'blockset', 'engine', 'mocknet', 'config',
                'commands/http', 'cmd/ipfs', 'command', 'conn', 'gc',
                'peerstore', 'core', 'coreunix', 'fsrepo', 'core/server',
                'boguskey', 'github.com/ipfs/go-libp2p/p2p/host/routed',
                'diagnostics', 'namesys', 'fuse/ipfs', 'node', 'secio',
                'core/commands', 'supernode', 'mdns', 'path', 'table',
                'swarm2', 'peerqueue', 'mount', 'fuse/ipns', 'blockstore',
                'github.com/ipfs/go-libp2p/p2p/host/basic', 'lock', 'nat',
                'importer', 'corerepo', 'dht.pb', 'pin', 'bitswap_network',
                'github.com/ipfs/go-libp2p/p2p/protocol/relay', 'peer',
                'transport', 'dht', 'offlinerouting', 'tarfmt', 'eventlog',
                'ipfsaddr', 'github.com/ipfs/go-libp2p/p2p/net/swarm/addr',
                'bitswap', 'reprovider', 'supernode/proxy', 'crypto', 'tour',
                'commands/cli', 'blockservice']}

        Returns
        -------
            dict : List of daemon logging subsystems
        """
        return self._client.request('/log/ls', decoder='json', **kwargs)

    def log_tail(self, **kwargs):
        r"""Reads log outputs as they are written.

        This function returns a reponse object that can be iterated over
        by the user. The user should make sure to close the response object
        when they are done reading from it.

        .. code-block:: python

            >>> for item in c.log_tail():
            ...     print(item)
            ...
            {"event":"updatePeer","system":"dht",
             "peerID":"QmepsDPxWtLDuKvEoafkpJxGij4kMax11uTH7WnKqD25Dq",
             "session":"7770b5e0-25ec-47cd-aa64-f42e65a10023",
             "time":"2016-08-22T13:25:27.43353297Z"}
            {"event":"handleAddProviderBegin","system":"dht",
             "peer":"QmepsDPxWtLDuKvEoafkpJxGij4kMax11uTH7WnKqD25Dq",
             "session":"7770b5e0-25ec-47cd-aa64-f42e65a10023",
             "time":"2016-08-22T13:25:27.433642581Z"}
            {"event":"handleAddProvider","system":"dht","duration":91704,
             "key":"QmNT9Tejg6t57Vs8XM2TVJXCwevWiGsZh3kB4HQXUZRK1o",
             "peer":"QmepsDPxWtLDuKvEoafkpJxGij4kMax11uTH7WnKqD25Dq",
             "session":"7770b5e0-25ec-47cd-aa64-f42e65a10023",
             "time":"2016-08-22T13:25:27.433747513Z"}
            {"event":"updatePeer","system":"dht",
             "peerID":"QmepsDPxWtLDuKvEoafkpJxGij4kMax11uTH7WnKqD25Dq",
             "session":"7770b5e0-25ec-47cd-aa64-f42e65a10023",
             "time":"2016-08-22T13:25:27.435843012Z"}
            …

        Returns
        -------
            iterable
        """
        return self._client.request('/log/tail', decoder='json',
                                    stream=True, **kwargs)

    def version(self, **kwargs):
        """Returns the software version of the currently connected node.

        .. code-block:: python

            >>> c.version()
            {'Version': '0.4.3-rc2', 'Repo': '4', 'Commit': '',
             'System': 'amd64/linux', 'Golang': 'go1.6.2'}

        Returns
        -------
            dict : Daemon and system version information
        """
        return self._client.request('/version', decoder='json', **kwargs)

    def files_cp(self, source, dest, **kwargs):
        """Copies files within the MFS.

        Due to the nature of IPFS this will not actually involve any of the
        file's content being copied.

        .. code-block:: python

            >>> c.files_ls("/")
            {'Entries': [
                {'Size': 0, 'Hash': '', 'Name': 'Software', 'Type': 0},
                {'Size': 0, 'Hash': '', 'Name': 'test', 'Type': 0}
            ]}
            >>> c.files_cp("/test", "/bla")
            ''
            >>> c.files_ls("/")
            {'Entries': [
                {'Size': 0, 'Hash': '', 'Name': 'Software', 'Type': 0},
                {'Size': 0, 'Hash': '', 'Name': 'bla', 'Type': 0},
                {'Size': 0, 'Hash': '', 'Name': 'test', 'Type': 0}
            ]}

        Parameters
        ----------
        source : str
            Filepath within the MFS to copy from
        dest : str
            Destination filepath with the MFS to which the file will be
            copied to
        """
        args = (source, dest)
        return self._client.request('/files/cp', args, **kwargs)

    def files_ls(self, path, **kwargs):
        """Lists contents of a directory in the MFS.

        .. code-block:: python

            >>> c.files_ls("/")
            {'Entries': [
                {'Size': 0, 'Hash': '', 'Name': 'Software', 'Type': 0}
            ]}

        Parameters
        ----------
        path : str
            Filepath within the MFS

        Returns
        -------
            dict : Directory entries
        """
        args = (path,)
        return self._client.request('/files/ls', args,
                                    decoder='json', **kwargs)

    def files_mkdir(self, path, parents=False, **kwargs):
        """Creates a directory within the MFS.

        .. code-block:: python

            >>> c.files_mkdir("/test")
            b''

        Parameters
        ----------
        path : str
            Filepath within the MFS
        parents : bool
            Create parent directories as needed and do not raise an exception
            if the requested directory already exists
        """
        kwargs.setdefault("opts", {"parents": parents})

        args = (path,)
        return self._client.request('/files/mkdir', args, **kwargs)

    def files_stat(self, path, **kwargs):
        """Returns basic ``stat`` information for an MFS file
        (including its hash).

        .. code-block:: python

            >>> c.files_stat("/test")
            {'Hash': 'QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn',
             'Size': 0, 'CumulativeSize': 4, 'Type': 'directory', 'Blocks': 0}

        Parameters
        ----------
        path : str
            Filepath within the MFS

        Returns
        -------
            dict : MFS file information
        """
        args = (path,)
        return self._client.request('/files/stat', args,
                                    decoder='json', **kwargs)

    def files_rm(self, path, recursive=False, **kwargs):
        """Removes a file from the MFS.

        .. code-block:: python

            >>> c.files_rm("/bla/file")
            b''

        Parameters
        ----------
        path : str
            Filepath within the MFS
        recursive : bool
            Recursively remove directories?
        """
        kwargs.setdefault("opts", {"recursive": recursive})

        args = (path,)
        return self._client.request('/files/rm', args, **kwargs)

    def files_read(self, path, offset=0, count=None, **kwargs):
        """Reads a file stored in the MFS.

        .. code-block:: python

            >>> c.files_read("/bla/file")
            b'hi'

        Parameters
        ----------
        path : str
            Filepath within the MFS
        offset : int
            Byte offset at which to begin reading at
        count : int
            Maximum number of bytes to read

        Returns
        -------
            str : MFS file contents
        """
        opts = {"offset": offset}
        if count is not None:
            opts["count"] = count
        kwargs.setdefault("opts", opts)

        args = (path,)
        return self._client.request('/files/read', args, **kwargs)

    def files_write(self, path, file, offset=0, create=False, truncate=False,
                    count=None, **kwargs):
        """Writes to a mutable file in the MFS.

        .. code-block:: python

            >>> c.files_write("/test/file", io.BytesIO(b"hi"), create=True)
            b''

        Parameters
        ----------
        path : str
            Filepath within the MFS
        file : io.RawIOBase
            IO stream object with data that should be written
        offset : int
            Byte offset at which to begin writing at
        create : bool
            Create the file if it does not exist
        truncate : bool
            Truncate the file to size zero before writing
        count : int
            Maximum number of bytes to read from the source ``file``
        """
        opts = {"offset": offset, "create": create, truncate: truncate}
        if count is not None:
            opts["count"] = count
        kwargs.setdefault("opts", opts)

        args = (path,)
        body, headers = multipart.stream_files(file, self.chunk_size)
        return self._client.request('/files/write', args,
                                    data=body, headers=headers, **kwargs)

    def files_mv(self, source, dest, **kwargs):
        """Moves files and directories within the MFS.

        .. code-block:: python

            >>> c.files_mv("/test/file", "/bla/file")
            b''

        Parameters
        ----------
        source : str
            Existing filepath within the MFS
        dest : str
            Destination to which the file will be moved in the MFS
        """
        args = (source, dest)
        return self._client.request('/files/mv', args, **kwargs)

    def shutdown(self):
        """Stop the connected IPFS daemon instance.

        Sending any further requests after this will fail with
        ``ipfsapi.exceptions.ConnectionError``, until you start another IPFS
        daemon instance.
        """
        return self._client.request('/shutdown')

    ###########
    # HELPERS #
    ###########

    @utils.return_field('Hash')
    def add_bytes(self, data, **kwargs):
        """Adds a set of bytes as a file to IPFS.

        .. code-block:: python

            >>> c.add_bytes(b"Mary had a little lamb")
            'QmZfF6C9j4VtoCsTp4KSrhYH47QMd3DNXVZBKaxJdhaPab'

        Also accepts and will stream generator objects.

        Parameters
        ----------
        data : bytes
            Content to be added as a file

        Returns
        -------
            str : Hash of the added IPFS object
        """
        body, headers = multipart.stream_bytes(data, self.chunk_size)
        return self._client.request('/add', decoder='json',
                                    data=body, headers=headers, **kwargs)

    @utils.return_field('Hash')
    def add_str(self, string, **kwargs):
        """Adds a Python string as a file to IPFS.

        .. code-block:: python

            >>> c.add_str(u"Mary had a little lamb")
            'QmZfF6C9j4VtoCsTp4KSrhYH47QMd3DNXVZBKaxJdhaPab'

        Also accepts and will stream generator objects.

        Parameters
        ----------
        string : str
            Content to be added as a file

        Returns
        -------
            str : Hash of the added IPFS object
        """
        body, headers = multipart.stream_text(string, self.chunk_size)
        return self._client.request('/add', decoder='json',
                                    data=body, headers=headers, **kwargs)

    def add_json(self, json_obj, **kwargs):
        """Adds a json-serializable Python dict as a json file to IPFS.

        .. code-block:: python

            >>> c.add_json({'one': 1, 'two': 2, 'three': 3})
            'QmVz9g7m5u3oHiNKHj2CJX1dbG1gtismRS3g9NaPBBLbob'

        Parameters
        ----------
        json_obj : dict
            A json-serializable Python dictionary

        Returns
        -------
            str : Hash of the added IPFS object
        """
        return self.add_bytes(encoding.Json().encode(json_obj), **kwargs)

    def get_json(self, multihash, **kwargs):
        """Loads a json object from IPFS.

        .. code-block:: python

            >>> c.get_json('QmVz9g7m5u3oHiNKHj2CJX1dbG1gtismRS3g9NaPBBLbob')
            {'one': 1, 'two': 2, 'three': 3}

        Parameters
        ----------
        multihash : str
           Multihash of the IPFS object to load

        Returns
        -------
            object : Deserialized IPFS JSON object value
        """
        return self.cat(multihash, decoder='json', **kwargs)

    def add_pyobj(self, py_obj, **kwargs):
        """Adds a picklable Python object as a file to IPFS.

        .. deprecated:: 0.4.2
           The ``*_pyobj`` APIs allow for arbitrary code execution if abused.
           Either switch to :meth:`~ipfsapi.Client.add_json` or use
           ``client.add_bytes(pickle.dumps(py_obj))`` instead.

        Please see :meth:`~ipfsapi.Client.get_pyobj` for the
        **security risks** of using these methods!

        .. code-block:: python

            >>> c.add_pyobj([0, 1.0, 2j, '3', 4e5])
            'QmWgXZSUTNNDD8LdkdJ8UXSn55KfFnNvTP1r7SyaQd74Ji'

        Parameters
        ----------
        py_obj : object
            A picklable Python object

        Returns
        -------
            str : Hash of the added IPFS object
        """
        warnings.warn("Using `*_pyobj` on untrusted data is a security risk",
                      DeprecationWarning)
        return self.add_bytes(encoding.Pickle().encode(py_obj), **kwargs)

    def get_pyobj(self, multihash, **kwargs):
        """Loads a pickled Python object from IPFS.

        .. deprecated:: 0.4.2
           The ``*_pyobj`` APIs allow for arbitrary code execution if abused.
           Either switch to :meth:`~ipfsapi.Client.get_json` or use
           ``pickle.loads(client.cat(multihash))`` instead.

        .. caution::

            The pickle module is not intended to be secure against erroneous or
            maliciously constructed data. Never unpickle data received from an
            untrusted or unauthenticated source.

            Please **read**
            `this article <https://www.cs.uic.edu/%7Es/musings/pickle/>`_ to
            understand the security risks of using this method!

        .. code-block:: python

            >>> c.get_pyobj('QmWgXZSUTNNDD8LdkdJ8UXSn55KfFnNvTP1r7SyaQd74Ji')
            [0, 1.0, 2j, '3', 400000.0]

        Parameters
        ----------
        multihash : str
            Multihash of the IPFS object to load

        Returns
        -------
            object : Deserialized IPFS Python object
        """
        warnings.warn("Using `*_pyobj` on untrusted data is a security risk",
                      DeprecationWarning)
        return self.cat(multihash, decoder='pickle', **kwargs)
