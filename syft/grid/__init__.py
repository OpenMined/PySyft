from .webrtc.network import Network

DEFAULT_NETWORK_URL = "ws://34.89.48.186:80"


def register(node_id: str, **kwargs):
    """ Add this process as a new peer registering it in the grid network.
        
        Args:
            node_id: Id used to identify this node.
        Returns:
            peer: Peer Network instance.
    """
    if not kwargs:
        args = args = {"max_size": None, "timeout": 444, "url": DEFAULT_NETWORK_URL}
    else:
        args = kwargs

    peer = Network(node_id, **args)
    peer.start()
    return peer
