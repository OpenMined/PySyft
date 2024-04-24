# stdlib
from pathlib import Path

# third party
from filelock import FileLock
import nginx
from nginx import Conf


class RatholeNginxConfigBuilder:
    def __init__(self, filename: str | Path) -> None:
        self.filename = Path(filename).absolute()

        if not self.filename.exists():
            self.filename.touch()

        self.lock = FileLock(f"{filename}.lock")
        self.lock_timeout = 30

    def read(self) -> Conf:
        with self.lock.acquire(timeout=self.lock_timeout):
            conf = nginx.loadf(self.filename)

        return conf

    def write(self, conf: Conf) -> None:
        with self.lock.acquire(timeout=self.lock_timeout):
            nginx.dumpf(conf, self.filename)

    def add_server(
        self,
        listen_port: int,
        location: str,
        proxy_pass: str,
        server_name: str | None = None,
    ) -> None:
        n_config = self.read()
        server_to_modify = self.find_server_with_listen_port(listen_port)

        if server_to_modify is not None:
            server_to_modify.add(
                nginx.Location(location, nginx.Key("proxy_pass", proxy_pass))
            )
            if server_name is not None:
                server_to_modify.add(nginx.Key("server_name", server_name))
        else:
            server = nginx.Server(
                nginx.Key("listen", listen_port),
                nginx.Location(location, nginx.Key("proxy_pass", proxy_pass)),
            )
            if server_name is not None:
                server.add(nginx.Key("server_name", server_name))

            n_config.add(server)

        self.write(n_config)

    def remove_server(self, listen_port: int) -> None:
        conf = self.read()
        for server in conf.servers:
            for child in server.children:
                if child.name == "listen" and int(child.value) == listen_port:
                    conf.remove(server)
                    break
        self.write(conf)

    def find_server_with_listen_port(self, listen_port: int) -> nginx.Server | None:
        conf = self.read()
        for server in conf.servers:
            for child in server.children:
                if child.name == "listen" and int(child.value) == listen_port:
                    return server
        return None

    def modify_proxy_for_port(
        self, listen_port: int, location: str, proxy_pass: str
    ) -> None:
        conf = self.read()
        server_to_modify = self.find_server_with_listen_port(listen_port)

        if server_to_modify is None:
            raise ValueError(f"Server with listen port {listen_port} not found")

        for location in server_to_modify.locations:
            if location.value != location:
                continue
            for key in location.keys:
                if key.name == "proxy_pass":
                    key.value = proxy_pass
                    break

        self.write(conf)
