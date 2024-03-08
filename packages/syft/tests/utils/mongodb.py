"""
NOTE:

At the moment testing using container is the easiest way to test MongoDB.

>> `mockmongo` does not support CodecOptions+TypeRegistry. It also doesn't sort on custom types.
>> Mongo binaries are no longer compiled for generic linux.
There's no guarantee that interpolated download URL will work with latest version of the OS, especially on Github CI.
"""

# stdlib
import socket

# third party
import docker

MONGO_CONTAINER_PREFIX = "pytest_mongo"
MONGO_VERSION = "7.0"


def get_random_port():
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.bind(("", 0))
    return soc.getsockname()[1]


def start_mongo_server(name, dbname="syft"):
    port = get_random_port()
    __start_mongo_container(name, port)
    return f"mongodb://127.0.0.1:{port}/{dbname}"


def stop_mongo_server(name):
    __destroy_mongo_container(name)


def __start_mongo_container(name, port=27017):
    client = docker.from_env()
    container_name = f"{MONGO_CONTAINER_PREFIX}_{name}"

    try:
        return client.containers.get(container_name)
    except docker.errors.NotFound:
        return client.containers.run(
            name=container_name,
            image=f"mongo:{MONGO_VERSION}",
            ports={"27017/tcp": port},
            detach=True,
            remove=True,
            auto_remove=True,
            labels={"name": "pytest-syft"},
        )


def __destroy_mongo_container(name):
    client = docker.from_env()
    container_name = f"{MONGO_CONTAINER_PREFIX}_{name}"

    try:
        container = client.containers.get(container_name)
        container.stop()
    except docker.errors.NotFound:
        pass
