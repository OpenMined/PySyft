# third party
import tomli
import tomli_w

# relative
from .rathole import RatholeConfig


class TomlReaderWriter:
    @staticmethod
    def load(toml_str: str) -> dict:
        return tomli.loads(toml_str)

    @staticmethod
    def dump(toml_dict: str) -> str:
        return tomli_w.dumps(toml_dict)


class RatholeBaseToml:
    filename: str

    def __init__(self, toml_str: str) -> None:
        self.toml_writer = TomlReaderWriter
        self.toml_str = toml_str

    def read(self) -> dict:
        return self.toml_writer.load(self.toml_str)

    def save(self, toml_dict: dict) -> None:
        self.toml_str = self.toml_writer.dump(toml_dict)

    def _validate(self) -> bool:
        raise NotImplementedError

    @property
    def is_valid(self) -> bool:
        return self._validate()


class RatholeClientToml(RatholeBaseToml):
    filename: str = "client.toml"

    def set_remote_addr(self, remote_host: str) -> None:
        """Add a new remote address to the client toml file."""

        toml = self.read()

        # Add the new remote address
        if "client" not in toml:
            toml["client"] = {}

        toml["client"]["remote_addr"] = remote_host

        self.save(toml)

    def clear_remote_addr(self) -> None:
        """Clear the remote address from the client toml file."""

        toml = self.read()

        # Clear the remote address
        if "client" not in toml:
            return

        toml["client"]["remote_addr"] = ""

        self.save(toml)

    def add_config(self, config: RatholeConfig) -> None:
        """Add a new config to the toml file."""

        toml = self.read()

        # Add the new config
        if "services" not in toml["client"]:
            toml["client"]["services"] = {}

        if config.uuid not in toml["client"]["services"]:
            toml["client"]["services"][config.uuid] = {}

        toml["client"]["services"][config.uuid] = {
            "token": config.secret_token,
            "local_addr": config.local_address,
        }

        self.save(toml)

    def remove_config(self, uuid: str) -> None:
        """Remove a config from the toml file."""

        toml = self.read()

        # Remove the config
        if "services" not in toml["client"]:
            return

        if uuid not in toml["client"]["services"]:
            return

        del toml["client"]["services"][uuid]

        self.save(toml)

    def update_config(self, config: RatholeConfig) -> None:
        """Update a config in the toml file."""

        toml = self.read()

        # Update the config
        if "services" not in toml["client"]:
            return

        if config.uuid not in toml["client"]["services"]:
            return

        toml["client"]["services"][config.uuid] = {
            "token": config.secret_token,
            "local_addr": config.local_address,
        }

        self.save(toml)

    def get_config(self, uuid: str) -> RatholeConfig | None:
        """Get a config from the toml file."""

        toml = self.read()

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
        toml = self.read()

        if not toml["client"]["remote_addr"]:
            return False

        for uuid, config in toml["client"]["services"].items():
            if not uuid:
                return False

            if not config["token"] or not config["local_addr"]:
                return False

        return True


class RatholeServerToml(RatholeBaseToml):
    filename: str = "server.toml"

    def set_rathole_listener_addr(self, bind_addr: str) -> None:
        """Set the bind address in the server toml file."""

        toml = self.read()

        # Set the bind address
        toml["server"]["bind_addr"] = bind_addr

        self.save(toml)

    def get_rathole_listener_addr(self) -> str:
        """Get the bind address from the server toml file."""

        toml = self.read()

        return toml["server"]["bind_addr"]

    def add_config(self, config: RatholeConfig) -> None:
        """Add a new config to the toml file."""

        toml = self.read()

        # Add the new config
        if "services" not in toml["server"]:
            toml["server"]["services"] = {}

        if config.uuid not in toml["server"]["services"]:
            toml["server"]["services"][config.uuid] = {}

        toml["server"]["services"][config.uuid] = {
            "token": config.secret_token,
            "bind_addr": config.local_address,
        }

        self.save(toml)

    def remove_config(self, uuid: str) -> None:
        """Remove a config from the toml file."""

        toml = self.read()

        # Remove the config
        if "services" not in toml["server"]:
            return

        if uuid not in toml["server"]["services"]:
            return

        del toml["server"]["services"][uuid]

        self.save(toml)

    def update_config(self, config: RatholeConfig) -> None:
        """Update a config in the toml file."""

        toml = self.read()

        # Update the config
        if "services" not in toml["server"]:
            return

        if config.uuid not in toml["server"]["services"]:
            return

        toml["server"]["services"][config.uuid] = {
            "token": config.secret_token,
            "bind_addr": config.local_address,
        }

        self.save(toml)

    def _validate(self) -> bool:
        toml = self.read()

        if not toml["server"]["bind_addr"]:
            return False

        for uuid, config in toml["server"]["services"].items():
            if not uuid:
                return False

            if not config["token"] or not config["bind_addr"]:
                return False

        return True
