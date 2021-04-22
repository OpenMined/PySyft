# stdlib
import argparse
import socket
import socketserver
from time import time

# syft absolute
from syft.grid.example_nodes.network import signaling_server


def free_port() -> int:
    with socketserver.TCPServer(("localhost", 0), None) as s:  # type: ignore
        return s.server_address[1]


def check_connectivity() -> None:
    start = time()
    while time() - start < 15:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", args.port)) == 0:
                break
    else:
        raise TimeoutError("Can't connect to the signaling server")


# Create the parser
my_parser = argparse.ArgumentParser(description="Start a signaling server for Syft.")

# Add the arguments
my_parser.add_argument(
    "--port",
    type=int,
    default=free_port(),
    help="the port on which to bind the signaling server",
)
my_parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="the ip address on which to bind the signaling server",
)
my_parser.add_argument(
    "--assert-connectivity", type=bool, default=True, help="Check if the binding worker"
)
my_parser.add_argument(
    "--timeout", type=int, default=15, help="Connectivity check timeout"
)

args = my_parser.parse_args()


if __name__ == "__main__":
    print(args.port)
    signaling_server(port=args.port, host=args.host)
