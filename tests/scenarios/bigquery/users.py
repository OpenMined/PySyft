# stdlib
from dataclasses import dataclass
from typing import Any

# syft absolute
from syft.service.user.user_roles import ServiceRole


@dataclass
class TestUser:
    name: str
    email: str
    password: str
    role: ServiceRole
    server_cache: Any | None = None

    def client(self, server=None):
        if server is None:
            server = self.server_cache
        else:
            self.server_cache = server

        return server.login(email=self.email, password=self.password)
