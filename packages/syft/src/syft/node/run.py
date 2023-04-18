# stdlib
import argparse

# relative
from ..client.deploy import Orchestra


def run():
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
        type=bool,
        default=True,
        dest="dev_mode",
    )
    parser.add_argument(
        "--reset",
        help="reset",
        type=bool,
        default=True,
        dest="reset",
    )
    parser.add_argument(
        "--local-db",
        help="reset",
        type=bool,
        default=False,
        dest="local_db",
    )
    parser.add_argument(
        "--processes",
        help="processing mode",
        type=int,
        default=False,
        dest="processes",
    )
    parser.add_argument(
        "--tail",
        help="tail mode",
        type=bool,
        default=True,
        dest="tail",
    )
    parser.add_argument(
        "--cmd",
        help="cmd mode",
        type=bool,
        default=False,
        dest="cmd",
    )

    args = parser.parse_args()

    if args.command != "launch":
        print("syft launch is the only command currently supported")

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
