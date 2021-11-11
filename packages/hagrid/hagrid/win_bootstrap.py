# stdlib
import subprocess

install_choco_pwsh = """
Start-Process PowerShell -Verb RunAs -ArgumentList "Set-ExecutionPolicy Bypass -Scope Process -Force;
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'));
Read-Host -Prompt string"
"""


def make_admin_cmd(admin_cmd: str) -> str:
    return (
        f"Start-Process PowerShell -Verb RunAs -ArgumentList "
        '"'
        "Set-ExecutionPolicy Bypass -Scope Process -Force; "
        f"{admin_cmd}; "
        "Read-Host -Prompt string"
        '"'
    )


def has_binary(binary: str) -> bool:
    try:
        subprocess.check_call(["where.exe", binary])
        return True
    except Exception:
        pass
    return False


def install_choco() -> None:
    if not has_binary("choco.exe"):
        try:
            powershell_cmds = ["-command", install_choco_pwsh]
            val = input(
                "Do you want to install choco.exe (Administrator Required) (Y/n): "
            )
            print(val)
            if "y" in val.lower():
                print("running pwsh", powershell_cmds)
                output = subprocess.check_call(["powershell.exe"] + powershell_cmds)
                print("output", output)
            else:
                print("ok no problem")
        except Exception as e:
            print("failed", e)
    else:
        print("choco.exe already installed")


def install_package(package: str, binary: str, admin: bool = False) -> None:
    if not has_binary(binary):
        try:
            choco_args = f"choco.exe install {package} -y"
            val = input(f"Do you want to install {package} (Y/n): ")
            print(val)
            if "y" in val.lower():
                powershell_cmds = ["-command", make_admin_cmd(choco_args)]
                print("running pwsh", powershell_cmds)
                output = subprocess.check_call(["powershell.exe"] + powershell_cmds)
                print("output", output)
            else:
                print("ok no problem")
        except Exception as e:
            print("failed", e)
    else:
        print(f"{binary} already installed")


def install_docker_desktop() -> None:
    return install_package(package="docker-desktop", binary="docker.exe", admin=True)


def install_wsl2() -> None:
    # there is no wsl2.exe so we need to check for Default Version: 2 in 'wsl --status'
    # and then set it if its not set
    return install_package(package="wsl2", binary="wsl2.exe", admin=True)


def install_ubuntu_wsl2() -> None:
    return install_package(
        package="wsl-ubuntu-2004", binary="ubuntu2004.exe", admin=True
    )


def install_git() -> None:
    return install_package(package="git", binary="git.exe", admin=True)


def install_anaconda3() -> None:
    return install_package(package="anaconda3", binary="conda.exe", admin=True)


def install_deps():
    install_choco()
    if has_binary("choco.exe"):
        install_anaconda3()
        install_git()
        install_wsl2()
        install_ubuntu_wsl2()
        install_docker_desktop()
    else:
        print("not found?")


install_deps()
