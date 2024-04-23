# stdlib
from pathlib import Path

# third party
from filelock import FileLock
import nginx
from nginx import Conf


class NginxConfigBuilder:
    def __init__(self, filename: str | Path) -> None:
        self.filename = Path(filename)
        self.lock = FileLock(f"{filename}.lock")
        self.lock_timeout = 30

    def read(self) -> Conf:
        with self.lock.acquire(timeout=self.lock_timeout):
            conf = nginx.loadf(self.filename)

        return conf

    def write(self, conf: Conf) -> None:
        with self.lock.acquire(timeout=self.lock_timeout):
            nginx.dumpf(conf, self.filename)

    def add_server(self, listen_port: int, location: str, proxy_pass: str) -> None:
        conf = self.read()
        server = conf.servers.add()
        server.listen = listen_port
        location = server.locations.add()
        location.path = location
        location.proxy_pass = proxy_pass
        self.write(conf)

    def remove_server(self, listen_port: int) -> None:
        conf = self.read()
        for server in conf.servers:
            if server.listen == listen_port:
                conf.servers.remove(server)
                break
        self.write(conf)

    def modify_location_for_port(
        self, listen_port: int, location: str, proxy_pass: str
    ) -> None:
        conf = self.read()
        for server in conf.servers:
            if server.listen == listen_port:
                for loc in server.locations:
                    if loc.path == location:
                        loc.proxy_pass = proxy_pass
                        break
        self.write(conf)
