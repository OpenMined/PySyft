# stdlib
import socket
import socketserver
import subprocess
import time


def generate_random_port() -> int:
    with socketserver.TCPServer(("localhost", 0), None) as s:  # type: ignore
        return int(s.server_address[1])


if __name__ == "__main__":
    port = generate_random_port()
    proc = subprocess.Popen(["syft-network", str(port)])
    start = time.time()

    while time.time() - start < 15:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", port)) == 0:
                break
    else:
        exit(1)

    proc.terminate()
