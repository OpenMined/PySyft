# stdlib
import argparse
import os
import platform
import re
import sys
from functools import cached_property
from pathlib import Path


class Platform:
    system = platform.system()
    uname = platform.uname()

    @staticmethod
    def get() -> str:
        return Platform.system

    @staticmethod
    def windows() -> bool:
        return Platform.system == "Windows"

    @staticmethod
    def macos() -> bool:
        return Platform.system == "Darwin"

    @staticmethod
    def linux() -> bool:
        return Platform.system == "Linux"

    @staticmethod
    def wsl2() -> bool:
        return "wsl2" in Platform.uname.release.lower()


class Hosts:
    def __init__(self, path: str | None = None) -> None:
        self.__path = path
        self.content = self.read()

    @cached_property
    def path(self) -> Path:
        if self.__path:
            p = self.__path
        elif Platform.linux():
            p = "/etc/hosts"
        elif Platform.macos():
            new_hosts = "/etc/hosts"
            old_hosts = "/private/etc/hosts"
            p = new_hosts if os.path.exists(new_hosts) else old_hosts
        elif Platform.windows():
            p = r"C:\Windows\System32\drivers\etc\hosts"
        else:
            msg = f"Unsupported OS: {Platform.system}"
            raise Exception(msg)

        p = Path(p).absolute()
        assert p.exists(), "Host file does not exist"
        return p

    def read(self) -> str:
        return self.path.read_text()

    def get(self, datasite: str) -> list[str]:
        return re.findall(rf"(.+)\s+{datasite}", self.content)

    def add(self, ip: str, datasite: str) -> None:
        if self.get(datasite):
            return

        self.content = self.content.rstrip() + f"\n{ip}\t{datasite}"
        self.__write()

    def remove(self, datasite: str) -> None:
        if not self.get(datasite):
            return

        self.content = re.sub(rf"(.+)\s+{datasite}\n", "", self.content)
        self.__write()

    def update(self, ip: str, datasite: str) -> None:
        if not self.get(datasite):
            self.add(ip, datasite)

        # inplace
        self.content = re.sub(
            rf"(.+)\s+{datasite}\n", f"{ip}\t{datasite}\n", self.content,
        )
        self.__write()

    def __write(self) -> None:
        cleaned = re.sub("\n{2}\n+", "\n", self.content)
        self.path.write_text(cleaned.rstrip() + "\n")


def running_as_root() -> bool:
    if not Platform.windows():
        return os.geteuid() == 0
    else:
        # stdlib
        import ctypes

        return ctypes.windll.shell32.IsUserAnAdmin() == 1


def wsl2_disable_auto_hosts() -> None:
    if not Platform.wsl2():
        return

    # stdlib
    import configparser

    conf_path = Path("/etc/wsl.conf")
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(conf_path)

    if "network" not in conf:
        conf["network"] = {}

    if conf["network"]["generateHosts"] != "false":
        conf["network"]["generateHosts"] = "false"
        with conf_path.open("w") as fp:
            conf.write(fp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add",
        nargs=2,
        action="append",
        default=[],
        metavar=("IP", "DATASITE"),
        help="Add entry to hosts file",
    )
    parser.add_argument(
        "--add-k3d-registry",
        action="store_true",
        default=False,
        help="Add entry for k3d-registry.localhost",
    )
    parser.add_argument(
        "--fix-docker-hosts",
        action="store_true",
        default=False,
        help="Windows - Fix *.docker.internal. Linux/macOS - remove them",
    )
    parser.add_argument(
        "--disable-wsl2-auto-hosts",
        action="store_true",
        default=False,
        dest="wsl2_disable_auto_hosts",
        help="[Optional] Disable automatic /etc/hosts generation from Windows in WSL2",
    )
    parser.add_argument(
        "--hosts",
        type=Path,
        default=None,
        help="[Optional] Path to a hosts-like file",
    )

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if not args.hosts and not running_as_root():
        sys.exit(1)

    hosts = Hosts(args.hosts)


    if len(args.add):
        for ip, datasite in args.add:
            hosts.update(ip, datasite)

    if args.add_k3d_registry:
        hosts.update("127.0.0.1", "k3d-registry.localhost")

    if args.fix_docker_hosts:
        if Platform.windows() or Platform.wsl2():
            hosts.update("0.0.0.0", "host.docker.internal")
            hosts.update("0.0.0.0", "gateway.docker.internal")
            hosts.update("127.0.0.1", "kubernetes.docker.internal")
        else:
            hosts.remove("host.docker.internal")
            hosts.remove("gateway.docker.internal")
            hosts.remove("kubernetes.docker.internal")

    if args.wsl2_disable_auto_hosts and Platform.wsl2():
        wsl2_disable_auto_hosts()



if __name__ == "__main__":
    main()
