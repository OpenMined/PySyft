"""
NOTE:

At the moment testing using container is the easiest way to test MongoDB.

>> `mockmongo` does not support CodecOptions+TypeRegistry. It also doesn't sort on custom types.
>> Mongo binaries are no longer compiled for generic linux.
There's no guarantee that interpolated download URL will work with latest version of the OS, especially on Github CI.
"""

# stdlib
from pathlib import Path
import platform
from shutil import copyfileobj
import subprocess
from tarfile import TarFile
from tempfile import gettempdir
from time import sleep
import zipfile

# third party
import distro
import docker
import psutil
import requests

# relative
from .random_port import get_random_port

MONGO_CONTAINER_PREFIX = "pytest_mongo"
MONGO_VERSION = "7.0"
MONGO_FULL_VERSION = f"{MONGO_VERSION}.6"
PLATFORM_ARCH = platform.machine()
PLATFORM_SYS = platform.system()
DISTRO_MONIKER = distro.id() + distro.major_version() + distro.minor_version()

MONGOD_PIDFILE = "mongod.pid"

MONGO_BINARIES = {
    "Darwin": f"https://fastdl.mongodb.org/osx/mongodb-macos-{PLATFORM_ARCH}-{MONGO_FULL_VERSION}.tgz",
    "Linux": f"https://fastdl.mongodb.org/linux/mongodb-linux-{PLATFORM_ARCH}-{DISTRO_MONIKER}-{MONGO_FULL_VERSION}.tgz",
    "Windows": f"https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-{MONGO_FULL_VERSION}.zip",
}


def start_mongo_server(name, dbname="syft"):
    port = get_random_port()

    try:
        __start_mongo_proc(name, port)
    except Exception:
        __start_mongo_container(name, port)

    return f"mongodb://127.0.0.1:{port}/{dbname}"


def stop_mongo_server(name):
    if PLATFORM_SYS in MONGO_BINARIES.keys():
        __kill_mongo_proc(name)
    else:
        __kill_mongo_container(name)


def __start_mongo_proc(name, port):
    download_dir = Path(gettempdir(), "mongodb")
    exec_path = __download_mongo(download_dir)
    if not exec_path:
        raise Exception("Failed to download MongoDB binaries")

    root_dir = Path(gettempdir(), name)

    db_path = Path(root_dir, "db")
    db_path.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            str(exec_path),
            "--port",
            str(port),
            "--dbpath",
            str(db_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    pid_path = root_dir / MONGOD_PIDFILE
    pid_path.write_text(str(proc.pid))

    return proc.pid


def __kill_mongo_proc(name):
    root_dir = Path(gettempdir(), name)
    pid_path = root_dir / MONGOD_PIDFILE
    pid = int(pid_path.read_text())

    mongod_proc = psutil.Process(pid)
    mongod_proc.terminate()
    sleep(1)


def __download_mongo(download_dir):
    url = MONGO_BINARIES.get(PLATFORM_SYS)
    if url is None:
        raise NotImplementedError(f"Unsupported platform: {PLATFORM_SYS}")

    download_path = Path(download_dir, f"mongodb_{MONGO_FULL_VERSION}.archive")
    download_path.parent.mkdir(parents=True, exist_ok=True)

    if not download_path.exists():
        # download the archive
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(download_path, "wb") as f:
                copyfileobj(r.raw, f)

        # extract it
        if url.endswith(".zip"):
            archive = zipfile.ZipFile(download_path, "r")
        else:
            archive = TarFile.open(download_path, "r")

        archive.extractall(download_dir)
        archive.close()

    for path in download_dir.glob(f"**/*{MONGO_FULL_VERSION}*/bin/mongod*"):
        if path.suffix not in (".exe", ""):
            continue
        return path


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


def __kill_mongo_container(name):
    client = docker.from_env()
    container_name = f"{MONGO_CONTAINER_PREFIX}_{name}"

    try:
        container = client.containers.get(container_name)
        container.stop()
    except docker.errors.NotFound:
        pass
