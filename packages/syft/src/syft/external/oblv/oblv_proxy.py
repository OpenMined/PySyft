# stdlib
import base64
import os
import platform
import subprocess  # nosec
import sys
import tarfile
import zipfile

# third party
import requests

# relative
from ...util.util import bcolors


def check_oblv_proxy_installation_status():
    try:
        result = subprocess.run(["oblv", "-V"], capture_output=True, text=True)  # nosec
        if result.stderr:
            raise subprocess.CalledProcessError(  # nosec
                returncode=result.returncode, cmd=result.args, stderr=result.stderr
            )
        result = result.stdout.strip()
        return result
    except Exception as e:
        if e.__class__ == FileNotFoundError:
            system_name = platform.system()
            result = "Oblv Proxy Not Installed. Call the method install_oblv_proxy "
            if system_name == "Windows":
                result += (
                    "to install the proxy for this session. If you already have the proxy installed,"
                    " add it to your PATH."
                )
            elif system_name == "Linux":
                result += (
                    "to install the proxy globally. If you already have the proxy installed,"
                    " create a link to the installation as /usr/local/bin/oblv"
                )

            print(
                bcolors.RED
                + bcolors.BOLD
                + "Exception"
                + bcolors.BLACK
                + bcolors.ENDC
                + ": "
                + result,
                file=sys.stderr,
            )
        else:
            raise Exception(e)


def install_oblv_proxy(with_package: bool = False):
    """Oblivious Proxy Installation

    Args:
        with_package (bool, optional): Only available for .msi, .deb, .rpm. Defaults to False.
    """
    system_name = platform.system()
    if system_name == "Windows":
        windows_proxy_installation(with_package)
    elif system_name == "Linux":
        linux_proxy_installation(with_package)
    elif system_name == "Darwin":
        darwin_proxy_installation()


def windows_proxy_installation(with_package: bool = False):
    try:
        if with_package:
            url = "https://api.oblivious.ai/oblv-ccli/0.4.0/packages/oblv-0.4.0-x86_64.msi"
            res = requests.get(url)  # nosec
            path = os.path.join(os.path.expanduser("~"), "oblv-0.4.0-x86_64.msi")
            with open(path, "wb") as f:
                f.write(res.content)
            os.system(f"msiexec /I {path} /quiet /QB-!")  # nosec
        else:
            url = "https://api.oblivious.ai/oblv-ccli/0.4.0/oblv-ccli-0.4.0-x86_64-pc-windows-msvc.zip"
            res = requests.get(url)  # nosec
            path = (
                os.getcwd().replace("\\", "/")
                + "/oblv-ccli-0.4.0-x86_64-pc-windows-msvc.zip"
            )
            with open(path, "wb") as f:
                f.write(res.content)
            with zipfile.ZipFile(path, "r") as zipObj:  # nosec
                zipObj.extractall()  # nosec
            os.environ["PATH"] += (
                ";" + os.getcwd() + "\\oblv-ccli-0.4.0-x86_64-pc-windows-msvc;"
            )
    except Exception as e:
        print(
            bcolors.RED
            + bcolors.BOLD
            + "Exception"
            + bcolors.BLACK
            + bcolors.ENDC
            + ": "
            + e.__cause__,
            file=sys.stderr,
        )


def linux_proxy_installation(with_package: bool = False):
    try:
        if with_package:
            try:
                os.system("dpkg")  # nosec
            except Exception:
                url = "https://api.oblivious.ai/oblv-ccli/0.4.0/packages/oblv-0.4.0-1.x86_64.rpm"
                res = requests.get(url)  # nosec
                path = os.path.join(os.path.expanduser("~"), "oblv-0.4.0-1.x86_64.rpm")
                with open(path, "wb") as f:
                    f.write(res.content)
                os.system(f"rpm -i {path}")  # nosec
            else:
                url = "https://api.oblivious.ai/oblv-ccli/0.4.0/packages/oblv_0.4.0_amd64.deb"
                res = requests.get(url)  # nosec
                path = os.path.join(os.path.expanduser("~"), "oblv_0.4.0_amd64.deb")
                with open(path, "wb") as f:
                    f.write(res.content)
                os.system(f"dpkg -i {path}")  # nosec
        else:
            url = "https://api.oblivious.ai/oblv-ccli/0.4.0/oblv-ccli-0.4.0-x86_64-unknown-linux-musl.tar.gz"
            file_name = "oblv-ccli-0.4.0-x86_64-unknown-linux-musl.tar.gz"
            res = requests.get(url, stream=True)  # nosec
            if res.status_code == 200:
                with open(file_name, "wb") as f:
                    f.write(res.raw.read())
            path = os.getcwd() + "/oblv-ccli-0.4.0-x86_64-unknown-linux-musl"
            file = tarfile.open(file_name)  # nosec
            file.extractall(path)  # nosec

            os.symlink(
                "/usr/local/bin/oblv",
                os.getcwd() + "/oblv-ccli-0.4.0-x86_64-unknown-linux-musl/oblv",
            )
            print(
                bcolors.green(bcolors.bold("Successfully")) + " installed Oblivous CLI"
            )
    except Exception as e:
        print(
            bcolors.RED
            + bcolors.BOLD
            + "Exception"
            + bcolors.BLACK
            + bcolors.ENDC
            + ": "
            + e.__cause__,
            file=sys.stderr,
        )


def darwin_proxy_installation():
    url = "https://api.oblivious.ai/oblv-ccli/0.4.0/oblv-ccli-0.4.0-x86_64-apple-darwin.tar.gz"
    file_name = "oblv-ccli-0.4.0-x86_64-apple-darwin.tar.gz"
    res = requests.get(url, stream=True)  # nosec
    if res.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(res.raw.read())
    path = os.getcwd() + "/oblv-ccli-0.4.0-x86_64-apple-darwin"
    file = tarfile.open(file_name)
    file.extractall(path)  # nosec

    os.symlink(
        "/usr/local/bin/oblv", os.getcwd() + "/oblv-ccli-0.4.0-x86_64-apple-darwin/oblv"
    )
    print(bcolors.green(bcolors.bold("Successfully")) + " installed Oblivous CLI")


def create_oblv_key_pair(key_name):
    if check_oblv_proxy_installation_status() is None:
        return
    try:
        file_path = os.path.join(os.path.expanduser("~"), ".ssh", key_name)
        result = subprocess.run(  # nosec
            ["oblv", "keygen", "--key-name", key_name, "--output", file_path],
            capture_output=True,
        )
        if result.stderr:
            raise subprocess.CalledProcessError(  # nosec
                returncode=result.returncode, cmd=result.args, stderr=result.stderr
            )
        result = result.stdout.strip()
        return get_oblv_public_key(key_name)
    except Exception as e:
        raise Exception(e)


def get_oblv_public_key(key_name):
    try:
        filepath = os.path.join(
            os.path.expanduser("~"), ".ssh", key_name, key_name + "_public.der"
        )
        with open(filepath, "rb") as f:
            public_key = f.read()
        public_key = base64.encodebytes(public_key).decode("UTF-8").replace("\n", "")
        return public_key
    except FileNotFoundError:
        print(
            bcolors.RED
            + bcolors.BOLD
            + "Exception"
            + bcolors.BLACK
            + bcolors.ENDC
            + ": "
            + "No key found with given name",
            file=sys.stderr,
        )
        raise FileNotFoundError
    except Exception as e:
        raise Exception(e)
