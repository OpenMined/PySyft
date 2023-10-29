# stdlib
import argparse
from functools import cached_property
import os
from pathlib import Path
import platform
import re
import sys


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
    def __init__(self, path: str = None) -> None:
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
            p = "C:\Windows\System32\drivers\etc\hosts"
        else:
            raise Exception(f"Unsupported OS: {Platform.system}")

        p = Path(p).absolute()
        assert p.exists(), "Host file does not exist"
        return p

    def read(self) -> str:
        return self.path.read_text()

    def get(self, domain: str) -> list[str]:
        return re.findall(f"(.+)\s+{domain}", self.content)

    def add(self, ip: str, domain: str) -> None:
        if self.get(domain):
            return

        self.content = self.content.rstrip() + f"\n{ip}\t{domain}"
        self.__write()

    def remove(self, domain: str) -> None:
        if not self.get(domain):
            return

        self.content = re.sub(f"(.+)\s+{domain}\n", "", self.content)
        self.__write()

    def update(self, ip: str, domain: str) -> None:
        if not self.get(domain):
            self.add(ip, domain)

        # inplace
        self.content = re.sub(f"(.+)\s+{domain}\n", f"{ip}\t{domain}\n", self.content)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add",
        nargs=2,
        action="append",
        default=[],
        metavar=("IP", "DOMAIN"),
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
        print(
            "ERROR: This script must be run as root since it will modify system hosts file"
        )
        sys.exit(1)

    hosts = Hosts(args.hosts)

    print(">> Args", args.__dict__)
    print(">> OS:", Platform.system)
    print(">> Release:", Platform.uname.release)
    print(">> Version:", Platform.uname.version)
    print(">> Hosts file:", hosts.path)

    if len(args.add):
        for ip, domain in args.add:
            print(f">> Adding {ip} {domain}")
            hosts.update(ip, domain)

    if args.add_k3d_registry:
        print(">> Adding k3d registry host entry")
        hosts.update("127.0.0.1", "k3d-registry.localhost")

    if args.fix_docker_hosts:
        if Platform.windows() or Platform.wsl2():
            print(">> Fixing docker host entries for Windows/WSL2")
            hosts.update("0.0.0.0", "host.docker.internal")
            hosts.update("0.0.0.0", "gateway.docker.internal")
            hosts.update("127.0.0.1", "kubernetes.docker.internal")
        else:
            print(">> Removing docker host entries")
            hosts.remove("host.docker.internal")
            hosts.remove("gateway.docker.internal")
            hosts.remove("kubernetes.docker.internal")

    if args.wsl2_disable_auto_hosts and Platform.wsl2():
        print(">> Disabling auto hosts generation")
        wsl2_disable_auto_hosts()

    print(">> Done")
    print("-" * 50)
    print(hosts.read())
    print("-" * 50)


if __name__ == "__main__":
    main()
