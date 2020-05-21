from .network import Network
import uuid

DEFAULT_NETWORK_URL = "ws://ec2-13-59-45-128.us-east-2.compute.amazonaws.com"

_registered_peer = None


def register(**kwargs):
    """ Add this process as a new peer registering it in the grid network.
        
        Returns:
            peer: Peer Network instance.
    """
    global _registered_peer

    if _registered_peer is None:
        if not kwargs:
            args = args = {"max_size": None, "timeout": 444, "url": DEFAULT_NETWORK_URL}
        else:
            args = kwargs

        peer_id = str(uuid.uuid4())
        _registered_peer = Network(peer_id, **args)
        _registered_peer.start()

    return _registered_peer
