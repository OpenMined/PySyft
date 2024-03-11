# stdlib
import argparse

# third party
from hagrid.orchestra import NodeHandle

# relative
from ..client.deploy import Orchestra


def str_to_bool(bool_str: str | None) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


def run() -> NodeHandle | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="command: launch", type=str, default="none")
    parser.add_argument(
        "--name", help="node name", type=str, default="syft-node", dest="name"
    )
    parser.add_argument(
        "--node-type", help="node type", type=str, default="python", dest="node_type"
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
        "--local-db",
        help="reset",
        type=str,
        default="False",
        dest="local_db",
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
    parser.add_argument(
        "--cmd",
        help="cmd mode",
        type=str,
        default="False",
        dest="cmd",
    )

    args = parser.parse_args()

    if args.command != "launch":
        print("syft launch is the only command currently supported")

    args.dev_mode = str_to_bool(args.dev_mode)
    args.reset = str_to_bool(args.reset)
    args.local_db = str_to_bool(args.local_db)
    args.tail = str_to_bool(args.tail)
    args.cmd = str_to_bool(args.cmd)

    node = Orchestra.launch(
        name=args.name,
        node_type=args.node_type,
        host=args.host,
        port=args.port,
        dev_mode=args.dev_mode,
        reset=args.reset,
        local_db=args.local_db,
        processes=args.processes,
        tail=args.tail,
        cmd=args.cmd,
    )
    if not args.tail:
        return node
    return None
