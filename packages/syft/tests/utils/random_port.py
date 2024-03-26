# stdlib
import socket


def get_random_port():
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.bind(("", 0))
    return soc.getsockname()[1]
