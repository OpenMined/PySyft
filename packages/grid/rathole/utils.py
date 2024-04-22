# stdlib

# third party
from filelock import FileLock

# relative
from .models import RatholeConfig
from .toml_writer import TomlReaderWriter

lock = FileLock("rathole.toml.lock")


class RatholeClientToml:
    filename: str = "client.toml"

    def __init__(self) -> None:
        self.client_toml = TomlReaderWriter(lock=lock, filename=self.filename)

    def set_remote_addr(self, remote_host: str) -> None:
        """Add a new remote address to the client toml file."""

        toml = self.client_toml.read()

        # Add the new remote address
        if "client" not in toml:
            toml["client"] = {}

            toml["client"]["remote_addr"] = remote_host

        if remote_host not in toml["client"]["remote"]:
            toml["client"]["remote"].append(remote_host)

        self.client_toml.write(toml_dict=toml)

    def add_config(self, config: RatholeConfig) -> None:
        """Add a new config to the toml file."""

        toml = self.client_toml.read()

        # Add the new config
        if "services" not in toml["client"]:
            toml["client"]["services"] = {}

        if config.uuid not in toml["client"]["services"]:
            toml["client"]["services"][config.uuid] = {}

        toml["client"]["services"][config.uuid] = {
            "token": config.secret_token,
            "local_addr": config.local_address,
        }

        self.client_toml.write(toml)

    def remove_config(self, uuid: str) -> None:
        """Remove a config from the toml file."""

        toml = self.client_toml.read()

        # Remove the config
        if "services" not in toml["client"]:
            return

        if uuid not in toml["client"]["services"]:
            return

        del toml["client"]["services"][uuid]

        self.client_toml.write(toml)

    def update_config(self, config: RatholeConfig) -> None:
        """Update a config in the toml file."""

        toml = self.client_toml.read()

        # Update the config
        if "services" not in toml["client"]:
            return

        if config.uuid not in toml["client"]["services"]:
            return

        toml["client"]["services"][config.uuid] = {
            "token": config.secret_token,
            "local_addr": config.local_address,
        }

        self.client_toml.write(toml)

    def get_config(self, uuid: str) -> RatholeConfig | None:
        """Get a config from the toml file."""

        toml = self.client_toml.read()

        # Get the config
        if "services" not in toml["client"]:
            return None

        if uuid not in toml["client"]["services"]:
            return None

        service = toml["client"]["services"][uuid]

        return RatholeConfig(
            uuid=uuid,
            secret_token=service["token"],
            local_addr_host=service["local_addr"].split(":")[0],
            local_addr_port=service["local_addr"].split(":")[1],
        )

    def _validate(self) -> bool:
        if not self.client_toml.filename.exists():
            return False

        toml = self.client_toml.read()

        if not toml["client"]["remote_addr"]:
            return False

        for uuid, config in toml["client"]["services"].items():
            if not uuid:
                return False

            if not config["token"] or not config["local_addr"]:
                return False

        return True

    @property
    def is_valid(self) -> bool:
        return self._validate()


class RatholeServerToml:
    filename: str = "server.toml"

    def __init__(self) -> None:
        self.server_toml = TomlReaderWriter(lock=lock, filename=self.filename)

    def set_bind_address(self, bind_address: str) -> None:
        """Set the bind address in the server toml file."""

        toml = self.server_toml.read()

        # Set the bind address
        toml["server"]["bind_addr"] = bind_address

        self.server_toml.write(toml)

    def add_config(self, config: RatholeConfig) -> None:
        """Add a new config to the toml file."""

        toml = self.server_toml.read()

        # Add the new config
        if "services" not in toml["server"]:
            toml["server"]["services"] = {}

        if config.uuid not in toml["server"]["services"]:
            toml["server"]["services"][config.uuid] = {}

        toml["server"]["services"][config.uuid] = {
            "token": config.secret_token,
            "bind_addr": config.local_address,
        }

        self.server_toml.write(toml)

    def remove_config(self, uuid: str) -> None:
        """Remove a config from the toml file."""

        toml = self.server_toml.read()

        # Remove the config
        if "services" not in toml["server"]:
            return

        if uuid not in toml["server"]["services"]:
            return

        del toml["server"]["services"][uuid]

        self.server_toml.write(toml)

    def update_config(self, config: RatholeConfig) -> None:
        """Update a config in the toml file."""

        toml = self.server_toml.read()

        # Update the config
        if "services" not in toml["server"]:
            return

        if config.uuid not in toml["server"]["services"]:
            return

        toml["server"]["services"][config.uuid] = {
            "token": config.secret_token,
            "bind_addr": config.local_address,
        }

        self.server_toml.write(toml)

    def _validate(self) -> bool:
        if not self.server_toml.filename.exists():
            return False

        toml = self.server_toml.read()

        if not toml["server"]["bind_addr"]:
            return False

        for uuid, config in toml["server"]["services"].items():
            if not uuid:
                return False

            if not config["token"] or not config["bind_addr"]:
                return False

        return True

    def is_valid(self) -> bool:
        return self._validate()
