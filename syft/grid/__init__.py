import sys
import uuid

from syft.grid.rtc.network import Network

DEFAULT_NETWORK_URL = "ws://ec2-13-59-45-128.us-east-2.compute.amazonaws.com"

_registered_peer = None


def register(**kwargs):
    """Add this process as a new peer registering it in the grid network.
    Returns:
        peer: Peer Network instance.
    """
    global _registered_peer

    if isinstance(_registered_peer, Network):
        sys.stdout.write(
            "\033[93m" + "WARNING" + "\033[0m"
            ":" + f" You are already a registered peer!\n{_registered_peer}\n"
        )

        return _registered_peer

    try:
        if not kwargs:
            args = {"max_size": None, "timeout": 444, "url": DEFAULT_NETWORK_URL}
        else:
            args = kwargs

        peer_id = str(uuid.uuid4())
        sys.stdout.write(
            "Connecting to OpenGrid (" + "\033[94m" + args["url"] + "\033[0m" + ") ... "
        )

        _registered_peer = Network(peer_id, **args)

        sys.stdout.write("\033[92m" + "OK" + "\033[0m" + "\n")
        sys.stdout.write("Peer ID: " + peer_id + "\n")

        sys.stdout.write(
            "\033[93m" + "DISCLAIMER" + "\033[0m"
            ":"
            + "\033[1m"
            + " OpenGrid is an experimental feature currently in alpha."
            + " Do not use this to protect real-world data.\n"
            + "\033[0m"
        )

        sys.stdout.write("Where to get help: \n")
        sys.stdout.write(
            " - Join our slack  (https://slack.openmined.org) and ask for"
            + "help in the #lib_syft channel.\n"
        )
        sys.stdout.write(
            " - File a Github Issue: https://github.com/OpenMined/PySyft and add the"
            + " string '#opengrid' in the issue title.\n"
        )
        sys.stdout.write(
            " - Want to join in our development team? Apply here:"
            + " https://forms.gle/wcH1vxzvPyDSbSVW6\n"
        )

        _registered_peer.start()

        return _registered_peer

    except Exception as e:
        sys.stdout.write("\033[91m" + "FAIL" + "\033[0m" + "\n")
        sys.stdout.write("You were not able to register your node.\n")
