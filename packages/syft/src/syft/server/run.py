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
        "--payment-required",
        help="payment required",
        type=str,
        default="False",
        dest="payment_required",
    )
    parser.add_argument(
        "--server-payment-handle",
        help="server payment handle",
        type=str,
        default="",
        dest="server_payment_handle",
    )
    parser.add_argument(
        "--payment-api",
        help="payment api",
        type=str,
        default="",
        dest="payment_api",
    )
    parser.add_argument(
        "--compute-price-module-path",
        help="compute price module path",
        type=str,
        default="",
        dest="compute_price_module_path",
    )
    parser.add_argument(
        "--compute-price-func-name",
        help="compute price function name",
        type=str,
        default="",
        dest="compute_price_func_name",
    )

    args = parser.parse_args()
    if args.command != "launch":
        print("syft launch is the only command currently supported")

    args.dev_mode = str_to_bool(args.dev_mode)
    args.reset = str_to_bool(args.reset)
    args.local_db = str_to_bool(args.local_db)
    args.tail = str_to_bool(args.tail)
    args.payment_required = str_to_bool(args.payment_required)
    args.server_payment_handle = str(args.server_payment_handle)
    args.payment_api = str(args.payment_api)
    args.compute_price_module_path = str(args.compute_price_module_path)
    args.compute_price_func_name = str(args.compute_price_func_name)

    server = Orchestra.launch(
        name=args.name,
        host=args.host,
        port=args.port,
        dev_mode=args.dev_mode,
        reset=args.reset,
        local_db=args.local_db,
        processes=args.processes,
        tail=args.tail,
        payment_required=args.payment_required,
        server_payment_handle=args.server_payment_handle,
        payment_api=args.payment_api,
        compute_price_module_path=args.compute_price_module_path,
        compute_price_func_name=args.compute_price_func_name,
    )
    if not args.tail:
        return server
    return None
