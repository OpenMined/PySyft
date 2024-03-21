# stdlib
import argparse
import random
import socket


def find_available_port(
    host: str, port: int | None = None, search: bool = False
) -> int:
    if port is None:
        port = random.randint(1500, 65000)  # nosec
    port_available = False
    while not port_available:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result_of_check = sock.connect_ex((host, port))

            if result_of_check != 0:
                port_available = True
                break
            else:
                if search:
                    port += 1
                else:
                    break
            sock.close()

        except Exception as e:
            print(f"Failed to check port {port}. {e}")
    sock.close()

    if search is False and port_available is False:
        error = (
            f"{port} is in use, either free the port or "
            + f"try: {port}+ to auto search for a port"
        )
        raise Exception(error)
    return port


def handle_command_line() -> None:
    parser = argparse.ArgumentParser(description="Find available port")

    parser.add_argument("--host", type=str, default="localhost")

    parser.add_argument("--port", type=int, default=None)

    parser.add_argument("--search", type=bool, default=True)

    args = parser.parse_args()

    print(find_available_port(args.host, args.port, args.search))


if __name__ == "__main__":
    handle_command_line()
