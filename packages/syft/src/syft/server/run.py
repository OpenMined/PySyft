# stdlib
import argparse

# relative
from ..orchestra import Orchestra
from ..orchestra import ServerHandle


def str_to_bool(bool_str: str | None) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


def run() -> ServerHandle | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="command: launch", type=str, default="none")
    parser.add_argument(
        "--name", help="server name", type=str, default="syft-server", dest="name"
    )
    parser.add_argument(
        "--server-type",
        help="server type",
        type=str,
        default="datasite",
        dest="server_type",
    )
    parser.add_argument(
        "--host",
        help="host for binding",
        type=str,
        default="0.0.0.0",  # nosec
        dest="host",
    )

    parser.add_argument(
        "--port", help="port for binding", type=int, default=8080, dest="port"
    )
    parser.add_argument(
        "--dev-mode",
        help="developer mode",
        type=str,
        default="True",
        dest="dev_mode",
    )
    parser.add_argument(
        "--reset",
        help="reset",
        type=str,
        default="True",
        dest="reset",
    )
    parser.add_argument(
        "--processes",
        help="processing mode",
        type=int,
        default=0,
        dest="processes",
    )
    parser.add_argument(
        "--tail",
        help="tail mode",
        type=str,
        default="True",
        dest="tail",
    )

    args = parser.parse_args()
    if args.command != "launch":
        print("syft launch is the only command currently supported")

    args.dev_mode = str_to_bool(args.dev_mode)
    args.reset = str_to_bool(args.reset)
    args.tail = str_to_bool(args.tail)

    server = Orchestra.launch(
        name=args.name,
        server_type=args.server_type,
        host=args.host,
        port=args.port,
        dev_mode=args.dev_mode,
        reset=args.reset,
        processes=args.processes,
        tail=args.tail,
    )
    if not args.tail:
        return server
    return None
