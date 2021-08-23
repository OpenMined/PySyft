# stdlib
import argparse
from multiprocessing import Process
import socket
import socketserver
from time import time

# syft absolute
from syft.grid.example_nodes.network import signaling_server


def free_port() -> int:
    with socketserver.TCPServer(("localhost", 0), None) as s:  # type: ignore
        return s.server_address[1]


def check_connectivity() -> bool:
    start = time()
    while time() - start < 15:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", args.port)) == 0:
                return True
    else:
        return False


arg_parser = argparse.ArgumentParser(description="Start a signaling server for Syft.")
arg_parser.add_argument(
    "--port",
    type=int,
    default=free_port(),
    help="the port on which to bind the signaling server",
)
arg_parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="the ip address on which to bind the signaling server",
)
arg_parser.add_argument(
    "--dry_run", type=bool, default=False, help="Check if the binding works"
)
arg_parser.add_argument(
    "--timeout", type=int, default=15, help="Connectivity check timeout"
)

args = arg_parser.parse_args()


if __name__ == "__main__":
    proc = Process(target=signaling_server, args=(args.port, args.host))
    if args.dry_run:
        proc.start()
        connected = check_connectivity()
        proc.terminate()
        exit(0) if connected else exit(1)
    else:
        proc.daemon = True
        proc.start()
        exit(0)
