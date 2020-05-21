from .network import Network
import sys

DEFAULT_NETWORK_URL = "ws://ec2-13-59-45-128.us-east-2.compute.amazonaws.com"


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

    sys.stdout.write(
        "Connecting to OpenGrid (" + "\033[94m" + DEFAULT_NETWORK_URL + "\033[0m" + ") ... \r"
    )
    peer = Network(node_id, **args)
    sys.stdout.flush()
    sys.stdout.write(
        "Connecting to OpenGrid ("
        + "\033[94m"
        + DEFAULT_NETWORK_URL
        + "\033[0m"
        + ") ... "
        + "\033[92m"
        + "OK"
        + "\033[0m"
        + "\nPeer ID: "
        + node_id
        + "\n"
    )

    sys.stdout.flush()
    sys.stdout.write(
        "\033[93m" + "DISCLAIMER" + "\033[0m"
        ":"
        + "\033[1m"
        + " OpenGrid is an experimental feature currently in alpha. Do not use this to protect real-world data.\n"
        + "\033[0m"
    )
    sys.stdout.write(
        "Where to get help: \n - Join our slack  (slack.openmined.org) and ask for help in the #lib_syft channel.\n - File a Github Issue: https://github.com/OpenMined/PySyft and add the string '#opengrid' in the issue title.\n"
    )
    peer.start()
    return peer
