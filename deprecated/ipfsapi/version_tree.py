""" This module contains an implementation of an IPFS
version-control system, which is structured as a directed in-tree with nodes
represented by the bytes representation of the VersionTreeNode class. """
from typing import Optional, Iterator
from datetime import datetime
import json
import numpy as np
import pickle
import base64
from bitcoin import base58
import threading 

from grid import ipfsapi
from grid.lib import utils

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
        self.id_hash =  None
        self.parent_hash = parent_hash or None
        self.ipfs_client = ipfs_client

    def commit(self, ipfs_client: ipfsapi.Client = None,
                broadcast = True, 
                broadcast_period = 1) -> str:
        """ Commits the node to the version tree,if broadcast set to true, 
        broadcast child periodically on pubsub and 
        returns the UTF-8 multihash representing its IPFS ID"""
            
        self.id_hash = (ipfs_client or self.ipfs_client).add_bytes(self.to_bytes())
        # If there parent_hash is set to None it means it's the root of the tree
        # therefore don't need to broadcast
        if self.parent_hash is not None and broadcast:
            self.broadcast_child_periodically(ipfs_client, broadcast_period)
            
        return self.id_hash

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

    def to_json(self) -> str:
        """Jsonify attributes of the child that will be 
        broadcasted on the channel openmined:children_of_<parent_hash>"""
        child = {'id_hash': self.id_hash, 'parent_hash': self.parent_hash}
        return json.dumps(child)
    
    def broadcast_child_periodically(self, ipfs_client: ipfsapi.Client, 
                                     broadcast_period = 1):
        """Broadcast child periodically on openmined:children_of_<parent_hash>
        to indicate to the parent the existence of the child"""
        channel = 'openmined:children_of_' + str(self.parent_hash)
        ipfs_client.pubsub_pub(topic = channel, 
                               payload=self.to_json(),
                               stream = True)
        
        _args = (ipfs_client,)
        threading.Timer(broadcast_period, 
                        self.broadcast_child_periodically, 
                        args=_args).start()
    
    def get_children(
            self, 
            parent_hash,
            ipfs_client: ipfsapi.Client,
            timeout = 5) -> Iterator["VersionTreeNode"]:
        """Listen to openmined:children_of_<parent_hash> and 
        return list of children (VersionTreeNode)"""
        channel = 'openmined:children_of_' + str(parent_hash)
        child_messages =  self.listen_to_channel_impl(ipfs_client, 
                                           channel, 
                                           self.receive_child, 
                                           timeout)
        
        children_list = [self.get_node_by_hash(child['id_hash'],ipfs_client) 
                        for child in child_messages]
    
        return children_list
    
    def receive_child(self, message: str) -> dict:
        """Extract child attributes from messages
        published on the channel openmined:children_of_<parent_hash> """
        msg = utils.unpack(message)
        child = {}
        child['id_hash'] = msg['id_hash']
        child['parent_hash'] = msg['parent_hash']
        return child

    def listen_to_channel_impl(self,
                               ipfs_client: ipfsapi.Client,
                               channel: str,
                               handle_message: dict,
                               timeout: int = 5) -> list:
        """Listen to channel for a certain period of time 
        and return a list of messages (info extracted based
        on handle message)"""
        start_time = datetime.now()
        time_delta = 0
        out_list = list()

        new_messages = ipfs_client.pubsub_sub(topic=channel, stream = True)

        # new_messages is a generator which will keep yield new messages until
        # you return from the loop. If you do return from the loop, we will no
        # longer be subscribed.
        for m in new_messages:
            message = self.decode_message(m)
            if message is not None:
                out = handle_message(message)
                if out is not None and out not in out_list:
                    out_list.append(out)
                    
            time_delta = datetime.now() - start_time   
            
            if time_delta.seconds >= timeout:
                break
                
        return out_list

                    
    def decode_message(self, encoded: dict) -> dict:
        """Decode message published on pubsub"""
        if ('from' in encoded):
            decoded = {}
            decoded['from'] = base64.standard_b64decode(encoded['from'])
            decoded['data'] = base64.standard_b64decode(
                encoded['data']).decode('ascii')
            decoded['seqno'] = base64.standard_b64decode(encoded['seqno'])
            decoded['topicIDs'] = encoded['topicIDs']
            decoded['encoded'] = encoded
            return decoded
        else:
            return None
