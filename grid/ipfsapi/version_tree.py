""" This module contains an implementation of an IPFS
version-control system, which is structured as a directed in-tree with nodes
represented by the bytes representation of the VersionTreeNode class. """
from typing import Optional, Iterator

from grid import ipfsapi

# TODO: Unit tests.
# TODO: Do we want to store the hash on the node after it's been committed?
class VersionTreeNode:
    """ Thin wrapper around a piece of IPFS-versioned data and the
    IPFS multihash of its parent. """
    # Delimiter for serializing packed object. Should not be alphanumeric.
    DELIMITER = b"|"

    def __init__(self,
                 contents: bytes,
                 parent_hash: Optional[str] = None,
                 ipfs_client: ipfsapi.Client = None):
        """ parent_hash is a UTF-8 IPFS multihash identifying
        this node's parent in the version tree. If parent_hash is None,
        this node is the root of a version tree. """
        self.contents = contents
        # Convert empty string to None to minimize typing bugs.
        self.parent_hash = parent_hash or None
        self.ipfs_client = ipfs_client

    def commit(self, ipfs_client: ipfsapi.Client = None) -> str:
        """ Commits the node to the version tree, and returns the
        UTF-8 multihash representing its IPFS ID"""
        return (ipfs_client or self.ipfs_client).add_bytes(self.to_bytes())

    @classmethod
    def get_node_by_hash(cls,
                         multihash: str,
                         ipfs_client: ipfsapi.Client) -> "VersionTreeNode":
        """ Retrieve and deserialize a VersionTreeNode addressed
        by it's UTF-8 multihash IPFS ID. """
        return cls.from_bytes(ipfs_client.cat(multihash))

    def get_with_ancestors(
            self,
            ipfs_client: ipfsapi.Client = None) -> Iterator["VersionTreeNode"]:
        """ Return an iterator containing this node and all its
        direct ancestors in the version tree, in that order. """
        yield self
        parent_hash = self.parent_hash
        while parent_hash is not None:
            parent_node = self.get_node_by_hash(
                parent_hash,
                (ipfs_client or self.ipfs_client))
            parent_hash = parent_node.parent_hash
            yield parent_node

    @classmethod
    def get_node_with_ancestors_by_hash(
            cls,
            multihash: str,
            ipfs_client: ipfsapi.Client) -> Iterator["VersionTreeNode"]:
        """ Convenience method to get an iterator of the node identified by the
        provided UTF-8 IPFS multihash, along with all of its ancestors, in
        that order."""
        return cls.get_node_by_hash(
            multihash, ipfs_client).get_with_ancestors(ipfs_client)

    def to_bytes(self) -> bytes:
        """ For contents b"foo", parent_hash "bar", and DELIMITER b"|",
        returns b"foo|bar" """
        parent_hash_bytes = self.parent_hash.encode("utf-8") if \
            self.parent_hash else \
            b""
        return self.DELIMITER.join((self.contents, parent_hash_bytes))

    @classmethod
    def from_bytes(cls, b: bytes) -> "VersionTreeNode":
        """ In case the contents section happens to contain the DELIMITER
        string, only splits on the final occurrence of DELIMITER. The
        multihash is hexadecimal, so it won't contain the non-hex DELIMITER."""
        contents, parent_hash_bytes = b.rsplit(cls.DELIMITER, maxsplit=1)
        return cls(contents, parent_hash_bytes.decode("utf-8"))

    def __repr__(self):
        return "VersionTreeNode with contents: {}\nparent_hash: {}".format(
            str(self.contents), self.parent_hash)

    def __eq__(self, other):
        return self.contents == other.contents and \
            self.parent_hash == other.parent_hash
