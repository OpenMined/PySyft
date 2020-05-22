from .network import Network
import sys
import uuid

DEFAULT_NETWORK_URL = "ws://ec2-13-59-45-128.us-east-2.compute.amazonaws.com"


def register(**kwargs):
    """ Add this process as a new peer registering it in the grid network.
        
        Returns:
            peer: Peer Network instance.
    """
    if not kwargs:
        args = args = {"max_size": None, "timeout": 444, "url": DEFAULT_NETWORK_URL}
    else:
        args = kwargs

    peer_id = str(uuid.uuid4())
    sys.stdout.write(
        "Connecting to OpenGrid (" + "\033[94m" + DEFAULT_NETWORK_URL + "\033[0m" + ") ... "
    )
    peer = Network(peer_id, **args)

    sys.stdout.write("\033[92m" + "OK" + "\033[0m" + "\n")
    sys.stdout.write("Peer ID: " + peer_id + "\n")

    sys.stdout.write(
        "\033[93m" + "DISCLAIMER" + "\033[0m"
        ":"
        + "\033[1m"
        + " OpenGrid is an experimental feature currently in alpha. Do not use this to protect real-world data.\n"
        + "\033[0m"
    )

    sys.stdout.write("Where to get help: \n")
    sys.stdout.write(
        " - Join our slack  (https://slack.openmined.org) and ask for help in the #lib_syft channel.\n"
    )
    sys.stdout.write(
        " - File a Github Issue: https://github.com/OpenMined/PySyft and add the string '#opengrid' in the issue title.\n"
    )
    sys.stdout.write(
        " - Want to join in our development team? Apply here: https://forms.gle/wcH1vxzvPyDSbSVW6\n"
    )
    peer.start()

    return peer
