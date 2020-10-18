import subprocess
import sys
from pathlib import Path
import argparse

if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help=(
            "if set, websocket server worker will load the test dataset "
            "instead of the training dataset",
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""if set, websocket server worker will be started in verbose mode""",
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default="normal",
        help="""can run websocket server for websockets examples of mnist/mnist-parallel or
        pen_testing/steal_data_over_sockets. Type 'mnist' for starting server
        for websockets-example-MNIST, `mnist-parallel` for websockets-example-MNIST-parallel
        and 'steal_data' for pen_tesing stealing data over sockets""",
    )
    parser.add_argument("--pytest_testing", action="store_true", help="""Used for pytest testing""")
    args = parser.parse_args()

    python = Path(sys.executable).name
    FILE_PATH = Path(__file__).resolve().parents[1].joinpath("run_websocket_server.py")
    call_alice = [
        python,
        FILE_PATH,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--id",
        args.id,
        "--pytest_testing",
    ]

    if args.verbose:
        call_alice.append("--verbose")

    if args.testing:
        call_alice.append("--testing")

    subprocess.Popen(call_alice)
