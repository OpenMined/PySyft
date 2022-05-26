# stdlib
from datetime import datetime
import json
import os
import re
import socket
import stat
import subprocess  # nosec
import sys
import tempfile
import time
from typing import Any
from typing import Callable
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from typing import Tuple
from typing import Tuple as TypeTuple
from typing import Union
from typing import cast

# third party
import click
import requests
import rich
from rich.live import Live

# relative
from . import __version__
from .art import hagrid
from .auth import AuthCredentials
from .cache import DEFAULT_BRANCH
from .cache import arg_cache
from .deps import DEPENDENCIES
from .deps import ENVIRONMENT
from .deps import MissingDependency
from .deps import allowed_hosts
from .deps import docker_info
from .deps import is_windows
from .deps import wsl_info
from .deps import wsl_linux_info
from .grammar import BadGrammar
from .grammar import GrammarVerb
from .grammar import parse_grammar
from .land import get_land_verb
from .launch import get_launch_verb
from .lib import GIT_REPO
from .lib import GRID_SRC_PATH
from .lib import GRID_SRC_VERSION
from .lib import check_api_metadata
from .lib import check_docker_version
from .lib import check_host
from .lib import check_jupyter_server
from .lib import check_login_page
from .lib import commit_hash
from .lib import docker_desktop_memory
from .lib import generate_process_status_table
from .lib import generate_user_table
from .lib import hagrid_root
from .lib import name_tag
from .lib import save_vm_details_as_json
from .lib import update_repo
from .lib import use_branch
from .mode import EDITABLE_MODE
from .rand_sec import generate_sec_random_password
from .style import RichGroup


def get_azure_image(short_name: str) -> str:
    prebuild_070 = (
        "madhavajay1632269232059:openmined_mj_grid_domain_ubuntu_1:domain_070:latest"
    )
    fresh_ubuntu = "Canonical:0001-com-ubuntu-server-focal:20_04-lts:latest"
    if short_name == "default":
        return fresh_ubuntu
    elif short_name == "domain_0.7.0":
        return prebuild_070
    raise Exception(f"Image name doesn't exist: {short_name}. Try: default or 0.7.0")


@click.group(cls=RichGroup)
def cli() -> None:
    pass


@click.command(
    help="Restore some part of the hagrid installation or deployment to its initial/starting state."
)
@click.argument("location", type=str, nargs=1)
def clean(location: str) -> None:

    if location == "library" or location == "volumes":
        print("Deleting all Docker volumes in 2 secs (Ctrl-C to stop)")
        time.sleep(2)
        subprocess.call("docker volume rm $(docker volume ls -q)", shell=True)  # nosec

    if location == "containers" or location == "pantry":
        print("Deleting all Docker containers in 2 secs (Ctrl-C to stop)")
        time.sleep(2)
        subprocess.call("docker rm -f $(docker ps -a -q)", shell=True)  # nosec

    if location == "images":
        print("Deleting all Docker images in 2 secs (Ctrl-C to stop)")
        time.sleep(2)
        subprocess.call("docker rmi $(docker images -q)", shell=True)  # nosec


@click.command(help="Start a new PyGrid domain/network node!")
@click.argument("args", type=str, nargs=-1)
@click.option(
    "--username",
    default=None,
    required=False,
    type=str,
    help="Optional: the username for provisioning the remote host",
)
@click.option(
    "--key_path",
    default=None,
    required=False,
    type=str,
    help="Optional: the path to the key file for provisioning the remote host",
)
@click.option(
    "--password",
    default=None,
    required=False,
    type=str,
    help="Optional: the password for provisioning the remote host",
)
@click.option(
    "--repo",
    default=None,
    required=False,
    type=str,
    help="Optional: repo to fetch source from",
)
@click.option(
    "--branch",
    default=None,
    required=False,
    type=str,
    help="Optional: branch to monitor for updates",
)
@click.option(
    "--tail",
    default="true",
    required=False,
    type=str,
    help="Optional: don't tail logs on launch",
)
@click.option(
    "--headless",
    default="false",
    required=False,
    type=str,
    help="Optional: don't start the frontend container",
)
@click.option(
    "--cmd",
    default="false",
    required=False,
    type=str,
    help="Optional: print the cmd without running it",
)
@click.option(
    "--jupyter",
    is_flag=True,
    help="Optional: enable Jupyter Notebooks",
)
@click.option(
    "--build",
    default=None,
    required=False,
    type=str,
    help="Optional: enable or disable forcing re-build",
)
@click.option(
    "--provision",
    default="true",
    required=False,
    type=str,
    help="Optional: enable or disable provisioning VMs",
)
@click.option(
    "--node_count",
    default=1,
    required=False,
    type=click.IntRange(1, 250),
    help="Optional: number of independent nodes/VMs to launch",
)
@click.option(
    "--auth_type",
    default=None,
    type=click.Choice(["key", "password"], case_sensitive=False),
)
@click.option(
    "--ansible_extras",
    default="",
    type=str,
)
@click.option("--tls", is_flag=True, help="Launch with TLS configuration")
@click.option("--test", is_flag=True, help="Launch with test configuration")
@click.option("--dev", is_flag=True, help="Shortcut for development release")
@click.option(
    "--release",
    default="production",
    required=False,
    type=click.Choice(["production", "development"], case_sensitive=False),
    help="Optional: choose between production and development release",
)
@click.option(
    "--cert_store_path",
    default="/home/om/certs",
    required=False,
    type=str,
    help="Optional: remote path to store and load TLS cert and key",
)
@click.option(
    "--upload_tls_cert",
    default="",
    required=False,
    type=str,
    help="Optional: local path to TLS cert to upload and store at --cert_store_path",
)
@click.option(
    "--upload_tls_key",
    default="",
    required=False,
    type=str,
    help="Optional: local path to TLS private key to upload and store at --cert_store_path",
)
@click.option(
    "--use_blob_storage",
    default=None,
    required=False,
    type=str,
    help="Optional: flag to use blob storage",
)
@click.option(
    "--image_name",
    default=None,
    required=False,
    type=str,
    help="Optional: image to use for the VM",
)
@click.option(
    "--tag",
    default=None,
    required=False,
    type=str,
    help="Optional: container image tag to use",
)
@click.option(
    "--build_src",
    default=DEFAULT_BRANCH,
    required=False,
    type=str,
    help="Optional: git branch to use for launch / build operations",
)
@click.option(
    "--platform",
    default=None,
    required=False,
    type=str,
    help="Optional: run docker with a different platform like linux/arm64",
)
def launch(args: TypeTuple[str], **kwargs: TypeDict[str, Any]) -> None:
    verb = get_launch_verb()
    try:
        grammar = parse_grammar(args=args, verb=verb)
        verb.load_grammar(grammar=grammar)
    except BadGrammar as e:
        print(e)
        return

    try:
        update_repo(repo=GIT_REPO, branch=str(kwargs["build_src"]))
    except Exception as e:
        print(f"Failed to update repo. {e}")
    try:
        cmds = create_launch_cmd(verb=verb, kwargs=kwargs)
        cmds = [cmds] if isinstance(cmds, str) else cmds
    except Exception as e:
        print(f"{e}")
        return

    dry_run = True
    if "cmd" not in kwargs or str_to_bool(cast(str, kwargs["cmd"])) is False:
        dry_run = False

    try:
        execute_commands(cmds, dry_run=dry_run)
    except Exception as e:
        print(f"{e}")
        return


def execute_commands(cmds: TypeList, dry_run: bool = False) -> None:
    """Execute the launch commands and display their status in realtime.

    Args:
        cmds (list): list of commands to be executed
        dry_run (bool, optional): If `True` only displays cmds to be executed. Defaults to False.
    """
    process_list: TypeList = []
    console = rich.get_console()

    username, password = (
        extract_username_and_pass(cmds[0]) if len(cmds) > 0 else ("-", "-")
    )
    # display VM credentials
    console.print(generate_user_table(username=username, password=password))

    for cmd in cmds:
        if dry_run:
            print("Running: \n", hide_password(cmd=cmd))
            continue

        # use powershell if environment is Windows
        cmd_to_exec = ["powershell.exe", "-Command", cmd] if is_windows() else cmd

        try:
            if len(cmds) > 1:
                process = subprocess.Popen(  # nosec
                    cmd_to_exec,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=GRID_SRC_PATH,
                    shell=True,
                )

                ip_address = extract_host_ip_from_cmd(cmd)
                jupyter_token = extract_jupyter_token(cmd)
                process_list.append((ip_address, process, jupyter_token))
            else:
                display_jupyter_token(cmd)
                subprocess.run(  # nosec
                    cmd_to_exec,
                    shell=True,
                    cwd=GRID_SRC_PATH,
                )
        except Exception as e:
            print(f"Failed to run cmd: {cmd}. {e}")

    if dry_run is False and len(process_list) > 0:
        # display VM launch status
        display_vm_status(process_list)

        # save vm details as json
        save_vm_details_as_json(username, password, process_list)


def display_vm_status(process_list: TypeList) -> None:
    """Display the status of the processes being executed on the VM.

    Args:
        process_list (list): list of processes executed.
    """

    # Generate the table showing the status of each process being executed
    status_table, process_completed = generate_process_status_table(process_list)

    # Render the live table
    with Live(status_table, refresh_per_second=1) as live:

        # Loop till all processes have not completed executing
        while not process_completed:
            status_table, process_completed = generate_process_status_table(
                process_list
            )
            live.update(status_table)  # Update the process status table


def display_jupyter_token(cmd: str) -> None:
    token = extract_jupyter_token(cmd=cmd)
    if token is not None:
        print(f"Jupyter Token: {token}")


def extract_username_and_pass(cmd: str) -> Tuple:
    # Extract username
    matcher = r"--user (.+?) "
    username = re.findall(matcher, cmd)
    username = username[0] if len(username) > 0 else None

    # Extract password
    matcher = r"ansible_ssh_pass='(.+?)'"
    password = re.findall(matcher, cmd)
    password = password[0] if len(password) > 0 else None

    return username, password


def extract_jupyter_token(cmd: str) -> Optional[str]:
    matcher = r"jupyter_token='(.+?)'"
    token = re.findall(matcher, cmd)
    if len(token) == 1:
        return token[0]
    return None


def hide_password(cmd: str) -> str:
    try:
        matcher = r"ansible_ssh_pass='(.+?)'"
        passwords = re.findall(matcher, cmd)
        if len(passwords) > 0:
            password = passwords[0]
            stars = "*" * 4
            cmd = cmd.replace(
                f"ansible_ssh_pass='{password}'", f"ansible_ssh_pass='{stars}'"
            )
        return cmd
    except Exception as e:
        print("Failed to hide password.")
        raise e


def hide_azure_vm_password(azure_cmd: str) -> str:
    try:
        matcher = r"admin-password '(.+?)'"
        passwords = re.findall(matcher, azure_cmd)
        if len(passwords) > 0:
            password = passwords[0]
            stars = "*" * 4
            azure_cmd = azure_cmd.replace(
                f"admin-password '{password}'", f"admin-password '{stars}'"
            )
        return azure_cmd
    except Exception as e:
        print("Failed to hide password.")
        raise e


class QuestionInputError(Exception):
    pass


class QuestionInputPathError(Exception):
    pass


class Question:
    def __init__(
        self,
        var_name: str,
        question: str,
        kind: str,
        default: Optional[str] = None,
        cache: bool = False,
        options: Optional[TypeList[str]] = None,
    ) -> None:
        self.var_name = var_name
        self.question = question
        self.default = default
        self.kind = kind
        self.cache = cache
        self.options = options if options is not None else []

    def validate(self, value: str) -> str:
        value = value.strip()
        if self.default is not None and value == "":
            return self.default

        if self.kind == "path":
            value = os.path.expanduser(value)
            if not os.path.exists(value):
                error = f"{value} is not a valid path."
                if self.default is not None:
                    error += f" Try {self.default}"
                raise QuestionInputPathError(f"{error}")

        if self.kind == "yesno":
            if value.lower().startswith("y"):
                return "y"
            elif value.lower().startswith("n"):
                return "n"
            else:
                raise QuestionInputError(f"{value} is not an yes or no answer")

        if self.kind == "options":
            if value in self.options:
                return value
            first_letter = value.lower()[0]
            for option in self.options:
                if option.startswith(first_letter):
                    return option

            raise QuestionInputError(
                f"{value} is not one of the options: {self.options}"
            )

        if self.kind == "password":
            try:
                return validate_password(password=value)
            except Exception as e:
                raise QuestionInputError(f"Invalid password. {e}")
        return value


def ask(question: Question, kwargs: TypeDict[str, str]) -> str:
    if question.var_name in kwargs and kwargs[question.var_name] is not None:
        value = kwargs[question.var_name]
    else:
        if question.default is not None:
            value = click.prompt(question.question, type=str, default=question.default)
        elif question.var_name == "password":
            value = click.prompt(
                question.question, type=str, hide_input=True, confirmation_prompt=True
            )
        else:
            value = click.prompt(question.question, type=str)

    try:
        value = question.validate(value=value)
    except QuestionInputError as e:
        print(e)
        return ask(question=question, kwargs=kwargs)
    if question.cache:
        setattr(arg_cache, question.var_name, value)

    return value


def fix_key_permission(private_key_path: str) -> None:
    key_permission = oct(stat.S_IMODE(os.stat(private_key_path).st_mode))
    chmod_permission = "400"
    octal_permission = f"0o{chmod_permission}"
    if key_permission != octal_permission:
        print(
            f"Fixing key permission: {private_key_path}, setting to {chmod_permission}"
        )
        try:
            os.chmod(private_key_path, int(octal_permission, 8))
        except Exception as e:
            print("Failed to fix key permission", e)
            raise e


def private_to_public_key(private_key_path: str, temp_path: str, username: str) -> str:
    # check key permission
    fix_key_permission(private_key_path=private_key_path)
    output_path = f"{temp_path}/hagrid_{username}_key.pub"
    cmd = f"ssh-keygen -f {private_key_path} -y > {output_path}"
    try:
        subprocess.check_call(cmd, shell=True)  # nosec
    except Exception as e:
        print("failed to make ssh key", e)
        raise e
    return output_path


def check_azure_authed() -> bool:
    cmd = "az account show"
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)  # nosec
        return True
    except Exception:  # nosec
        pass
    return False


def login_azure() -> bool:
    cmd = "az login"
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)  # nosec
        return True
    except Exception:  # nosec
        pass
    return False


def check_azure_cli_installed() -> bool:
    try:
        result = subprocess.run(  # nosec
            ["az", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        if result.returncode != 0:
            raise FileNotFoundError("az not installed")
    except Exception:  # nosec
        msg = "\nYou don't appear to have the Azure CLI installed!!! \n\n\
Please install it and then retry your command.\
\n\nInstallation Instructions: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli\n"
        raise FileNotFoundError(msg)

    return True


def check_gcloud_cli_installed() -> bool:
    try:
        subprocess.call(["gcloud", "version"])  # nosec
        print("Gcloud cli installed!")
    except FileNotFoundError:
        msg = "\nYou don't appear to have the gcloud CLI tool installed! \n\n\
Please install it and then retry again.\
\n\nInstallation Instructions: https://cloud.google.com/sdk/docs/install-sdk \n"
        raise FileNotFoundError(msg)

    return True


def check_gcloud_authed() -> bool:
    try:
        result = subprocess.run(  # nosec
            ["gcloud", "auth", "print-identity-token"], stdout=subprocess.PIPE
        )
        if result.returncode == 0:
            return True
    except Exception:  # nosec
        pass
    return False


def login_gcloud() -> bool:
    cmd = "gcloud auth login"
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)  # nosec
        return True
    except Exception:  # nosec
        pass
    return False


def str_to_bool(bool_str: Optional[str]) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


ART = str_to_bool(os.environ.get("HAGRID_ART", "True"))


def generate_gcloud_key_at_path(key_path: str) -> str:
    key_path = os.path.expanduser(key_path)
    if os.path.exists(key_path):
        raise Exception(f"Can't generate key since path already exists. {key_path}")
    else:
        # triggers a key check
        cmd = "gcloud compute ssh '' --dry-run"
        try:
            subprocess.check_call(cmd, shell=True)  # nosec
        except Exception:  # nosec
            pass
        if not os.path.exists(key_path):
            raise Exception(f"gcloud failed to generate ssh-key at: {key_path}")

    return key_path


def generate_key_at_path(key_path: str) -> str:
    key_path = os.path.expanduser(key_path)
    if os.path.exists(key_path):
        raise Exception(f"Can't generate key since path already exists. {key_path}")
    else:
        cmd = f"ssh-keygen -N '' -f {key_path}"
        try:
            subprocess.check_call(cmd, shell=True)  # nosec
            if not os.path.exists(key_path):
                raise Exception(f"Failed to generate ssh-key at: {key_path}")
        except Exception as e:
            raise e

    return key_path


def validate_password(password: str) -> str:
    """Validate if the password entered by the user is valid.

    Password length should be between 12 - 123 characters
    Passwords must also meet 3 out of the following 4 complexity requirements:
    - Have lower characters
    - Have upper characters
    - Have a digit
    - Have a special character

    Args:
        password (str): password for the vm

    Returns:
        str: password if it is valid
    """
    # Validate password length
    if len(password) < 12 or len(password) > 123:
        raise ValueError("Password length should be between 12 - 123 characters")

    # Valid character types
    character_types = {
        "upper_case": False,
        "lower_case": False,
        "digit": False,
        "special": False,
    }

    for ch in password:
        if ch.islower():
            character_types["lower_case"] = True
        elif ch.isupper():
            character_types["upper_case"] = True
        elif ch.isdigit():
            character_types["digit"] = True
        elif ch.isascii():
            character_types["special"] = True
        else:
            raise ValueError(f"{ch} is not a valid character for password")

    # Validate characters in the password
    required_character_type_count = sum(
        [int(value) for value in character_types.values()]
    )

    if required_character_type_count >= 3:
        return password

    absent_character_types = ", ".join(
        char_type for char_type, value in character_types.items() if value is False
    ).strip(", ")

    raise ValueError(
        f"At least one {absent_character_types} character types must be present"
    )


def create_launch_cmd(
    verb: GrammarVerb,
    kwargs: TypeDict[str, Any],
    ignore_docker_version_check: Optional[bool] = False,
) -> Union[str, TypeList[str]]:
    parsed_kwargs: TypeDict[str, Any] = {}
    host_term = verb.get_named_term_hostgrammar(name="host")
    host = host_term.host
    auth: Optional[AuthCredentials] = None

    tail = True
    if "tail" in kwargs and not str_to_bool(kwargs["tail"]):
        tail = False

    parsed_kwargs = {}

    if "build" in kwargs and kwargs["build"] is not None:
        parsed_kwargs["build"] = str_to_bool(cast(str, kwargs["build"]))
    else:
        parsed_kwargs["build"] = None

    parsed_kwargs["use_blob_storage"] = (
        kwargs["use_blob_storage"] if "use_blob_storage" in kwargs else None
    )

    parsed_kwargs["node_count"] = (
        int(kwargs["node_count"]) if "node_count" in kwargs else 1
    )

    if parsed_kwargs["node_count"] > 1 and host not in ["azure"]:
        print("\nArgument `node_count` is only supported with `azure`.\n")
    else:
        # Default to detached mode if running more than one nodes
        tail = False if parsed_kwargs["node_count"] > 1 else tail

    headless = False
    if "headless" in kwargs and str_to_bool(cast(str, kwargs["headless"])):
        headless = True
    parsed_kwargs["headless"] = headless

    parsed_kwargs["tls"] = bool(kwargs["tls"]) if "tls" in kwargs else False
    parsed_kwargs["test"] = bool(kwargs["test"]) if "test" in kwargs else False
    parsed_kwargs["dev"] = bool(kwargs["dev"]) if "dev" in kwargs else False

    parsed_kwargs["release"] = "production"
    if "release" in kwargs and kwargs["release"] != "production":
        parsed_kwargs["release"] = kwargs["release"]

    # if we use --dev override it
    if parsed_kwargs["dev"] is True:
        parsed_kwargs["release"] = "development"

    if "cert_store_path" in kwargs:
        parsed_kwargs["cert_store_path"] = kwargs["cert_store_path"]
    if "upload_tls_cert" in kwargs:
        parsed_kwargs["upload_tls_cert"] = kwargs["upload_tls_cert"]
    if "upload_tls_key" in kwargs:
        parsed_kwargs["upload_tls_key"] = kwargs["upload_tls_key"]
    if "provision" in kwargs:
        parsed_kwargs["provision"] = str_to_bool(cast(str, kwargs["provision"]))

    if "image_name" in kwargs and kwargs["image_name"] is not None:
        parsed_kwargs["image_name"] = kwargs["image_name"]
    else:
        parsed_kwargs["image_name"] = "default"

    if "tag" in kwargs and kwargs["tag"] is not None and kwargs["tag"] != "":
        parsed_kwargs["tag"] = kwargs["tag"]
    else:
        parsed_kwargs["tag"] = None

    if "jupyter" in kwargs and kwargs["jupyter"] is not None:
        parsed_kwargs["jupyter"] = str_to_bool(cast(str, kwargs["jupyter"]))
    else:
        parsed_kwargs["jupyter"] = False

    # allows changing docker platform to other cpu architectures like arm64
    parsed_kwargs["platform"] = kwargs["platform"] if "platform" in kwargs else None

    if host in ["docker"]:

        if not ignore_docker_version_check:
            version = check_docker_version()
        else:
            version = "n/a"

        if version:
            # If the user is using docker desktop (OSX/Windows), check to make sure there's enough RAM.
            # If the user is using Linux this isn't an issue because Docker scales to the avaialble RAM,
            # but on Docker Desktop it defaults to 2GB which isn't enough.
            dd_memory = docker_desktop_memory()
            if dd_memory < 8192 and dd_memory != -1:
                raise Exception(
                    "You appear to be using Docker Desktop but don't have "
                    "enough memory allocated. It appears you've configured "
                    f"Memory:{dd_memory} MB when 8192MB (8GB) is required. "
                    f"Please open Docker Desktop Preferences panel and set Memory"
                    f" to 8GB or higher. \n\n"
                    f"\tOSX Help: https://docs.docker.com/desktop/mac/\n"
                    f"\tWindows Help: https://docs.docker.com/desktop/windows/\n\n"
                    f"Then re-run your hagrid command.\n\n"
                    f"If you see this warning on Linux then something isn't right. "
                    f"Please file a Github Issue on PySyft's Github"
                )

            if is_windows() and not DEPENDENCIES["wsl"]:
                raise Exception(
                    "You must install wsl2 for Windows to use HAGrid.\n"
                    "In PowerShell or Command Prompt type:\n> wsl --install\n\n"
                    "Read more here: https://docs.microsoft.com/en-us/windows/wsl/install"
                )

            return create_launch_docker_cmd(
                verb=verb, docker_version=version, tail=tail, kwargs=parsed_kwargs
            )

    elif host in ["vm"]:
        if (
            DEPENDENCIES["vagrant"]
            and DEPENDENCIES["virtualbox"]
            and DEPENDENCIES["ansible-playbook"]
        ):
            return create_launch_vagrant_cmd(verb=verb)
        else:
            errors = []
            if not DEPENDENCIES["vagrant"]:
                errors.append("vagrant")
            if not DEPENDENCIES["virtualbox"]:
                errors.append("virtualbox")
            if not DEPENDENCIES["ansible-playbook"]:
                errors.append("ansible-playbook")
            raise MissingDependency(
                f"Launching a VM locally requires: {' '.join(errors)}"
            )
    elif host in ["azure"]:
        check_azure_cli_installed()

        while not check_azure_authed():
            print("You need to log into Azure")
            login_azure()

        if DEPENDENCIES["ansible-playbook"]:

            resource_group = ask(
                question=Question(
                    var_name="azure_resource_group",
                    question="What resource group name do you want to use (or create)?",
                    default=arg_cache.azure_resource_group,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            location = ask(
                question=Question(
                    var_name="azure_location",
                    question="If this is a new resource group what location?",
                    default=arg_cache.azure_location,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            size = ask(
                question=Question(
                    var_name="azure_size",
                    question="What size machine?",
                    default=arg_cache.azure_size,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            username = ask(
                question=Question(
                    var_name="azure_username",
                    question="What do you want the username for the VM to be?",
                    default=arg_cache.azure_username,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            parsed_kwargs["auth_type"] = ask(
                question=Question(
                    var_name="auth_type",
                    question="Do you want to login with a key or password",
                    default=arg_cache.auth_type,
                    kind="option",
                    options=["key", "password"],
                    cache=True,
                ),
                kwargs=kwargs,
            )

            key_path = None
            if parsed_kwargs["auth_type"] == "key":
                key_path_question = Question(
                    var_name="azure_key_path",
                    question=f"Absolute path of the private key to access {username}@{host}?",
                    default=arg_cache.azure_key_path,
                    kind="path",
                    cache=True,
                )
                try:
                    key_path = ask(
                        key_path_question,
                        kwargs=kwargs,
                    )
                except QuestionInputPathError as e:
                    print(e)
                    key_path = str(e).split("is not a valid path")[0].strip()

                    create_key_question = Question(
                        var_name="azure_key_path",
                        question=f"Key {key_path} does not exist. Do you want to create it? (y/n)",
                        default="y",
                        kind="yesno",
                    )
                    create_key = ask(
                        create_key_question,
                        kwargs=kwargs,
                    )
                    if create_key == "y":
                        key_path = generate_key_at_path(key_path=key_path)
                    else:
                        raise QuestionInputError(
                            "Unable to create VM without a private key"
                        )
            elif parsed_kwargs["auth_type"] == "password":
                auto_generate_password = ask(
                    question=Question(
                        var_name="auto_generate_password",
                        question="Do you want to auto-generate the password? (y/n)",
                        kind="yesno",
                    ),
                    kwargs=kwargs,
                )
                if auto_generate_password == "y":  # nosec
                    parsed_kwargs["password"] = generate_sec_random_password(length=16)
                elif auto_generate_password == "n":  # nosec
                    parsed_kwargs["password"] = ask(
                        question=Question(
                            var_name="password",
                            question=f"Password for {username}@{host}?",
                            kind="password",
                        ),
                        kwargs=kwargs,
                    )

            repo = ask(
                Question(
                    var_name="azure_repo",
                    question="Repo to fetch source from?",
                    default=arg_cache.azure_repo,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )
            branch = ask(
                Question(
                    var_name="azure_branch",
                    question="Branch to monitor for updates?",
                    default=arg_cache.azure_branch,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            use_branch(branch=branch)

            password = parsed_kwargs.get("password")

            auth = AuthCredentials(
                username=username, key_path=key_path, password=password
            )

            if not auth.valid:
                raise Exception(f"Login Credentials are not valid. {auth}")

            return create_launch_azure_cmd(
                verb=verb,
                resource_group=resource_group,
                location=location,
                size=size,
                username=username,
                password=password,
                key_path=key_path,
                repo=repo,
                branch=branch,
                auth=auth,
                ansible_extras=kwargs["ansible_extras"],
                kwargs=parsed_kwargs,
            )
        else:
            errors = []
            if not DEPENDENCIES["ansible-playbook"]:
                errors.append("ansible-playbook")
            msg = "\nERROR!!! MISSING DEPENDENCY!!!"
            msg += f"\n\nLaunching a Cloud VM requires: {' '.join(errors)}"
            msg += "\n\nPlease follow installation instructions: "
            msg += "https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html#"
            msg += "\n\nNote: we've found the 'conda' based installation instructions to work best"
            msg += " (e.g. something lke 'conda install -c conda-forge ansible'). "
            msg += "The pip based instructions seem to be a bit buggy if you're using a conda environment"
            msg += "\n"
            raise MissingDependency(msg)

    elif host in ["gcp"]:
        check_gcloud_cli_installed()

        while not check_gcloud_authed():
            print("You need to log into Google Cloud")
            login_gcloud()

        if DEPENDENCIES["ansible-playbook"]:
            project_id = ask(
                question=Question(
                    var_name="gcp_project_id",
                    question="What PROJECT ID do you want to use?",
                    default=arg_cache.gcp_project_id,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            zone = ask(
                question=Question(
                    var_name="gcp_zone",
                    question="What zone do you want your VM in?",
                    default=arg_cache.gcp_zone,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            machine_type = ask(
                question=Question(
                    var_name="gcp_machine_type",
                    question="What size machine?",
                    default=arg_cache.gcp_machine_type,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            username = ask(
                question=Question(
                    var_name="gcp_username",
                    question="What is your shell username?",
                    default=arg_cache.gcp_username,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            key_path_question = Question(
                var_name="gcp_key_path",
                question=f"Private key to access user@{host}?",
                default=arg_cache.gcp_key_path,
                kind="path",
                cache=True,
            )
            try:
                key_path = ask(
                    key_path_question,
                    kwargs=kwargs,
                )
            except QuestionInputPathError as e:
                print(e)
                key_path = str(e).split("is not a valid path")[0].strip()

                create_key_question = Question(
                    var_name="gcp_key_path",
                    question=f"Key {key_path} does not exist. Do you want gcloud to make it? (y/n)",
                    default="y",
                    kind="yesno",
                )
                create_key = ask(
                    create_key_question,
                    kwargs=kwargs,
                )
                if create_key == "y":
                    key_path = generate_gcloud_key_at_path(key_path=key_path)
                else:
                    raise QuestionInputError(
                        "Unable to create VM without a private key"
                    )

            repo = ask(
                Question(
                    var_name="gcp_repo",
                    question="Repo to fetch source from?",
                    default=arg_cache.gcp_repo,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )
            branch = ask(
                Question(
                    var_name="gcp_branch",
                    question="Branch to monitor for updates?",
                    default=arg_cache.gcp_branch,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            use_branch(branch=branch)

            auth = AuthCredentials(username=username, key_path=key_path)

            return create_launch_gcp_cmd(
                verb=verb,
                project_id=project_id,
                zone=zone,
                machine_type=machine_type,
                repo=repo,
                auth=auth,
                branch=branch,
                ansible_extras=kwargs["ansible_extras"],
                kwargs=parsed_kwargs,
            )
        else:
            errors = []
            if not DEPENDENCIES["ansible-playbook"]:
                errors.append("ansible-playbook")
            msg = "\nERROR!!! MISSING DEPENDENCY!!!"
            msg += f"\n\nLaunching a Cloud VM requires: {' '.join(errors)}"
            msg += "\n\nPlease follow installation instructions: "
            msg += "https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html#"
            msg += "\n\nNote: we've found the 'conda' based installation instructions to work best"
            msg += " (e.g. something lke 'conda install -c conda-forge ansible'). "
            msg += "The pip based instructions seem to be a bit buggy if you're using a conda environment"
            msg += "\n"
            raise MissingDependency(msg)

    elif host in ["aws"]:
        print("Coming soon.")
        return ""
    else:
        if DEPENDENCIES["ansible-playbook"]:
            if host != "localhost":
                parsed_kwargs["username"] = ask(
                    question=Question(
                        var_name="username",
                        question=f"Username for {host} with sudo privledges?",
                        default=arg_cache.username,
                        kind="string",
                        cache=True,
                    ),
                    kwargs=kwargs,
                )
                parsed_kwargs["auth_type"] = ask(
                    question=Question(
                        var_name="auth_type",
                        question="Do you want to login with a key or password",
                        default=arg_cache.auth_type,
                        kind="option",
                        options=["key", "password"],
                        cache=True,
                    ),
                    kwargs=kwargs,
                )
                if parsed_kwargs["auth_type"] == "key":
                    parsed_kwargs["key_path"] = ask(
                        question=Question(
                            var_name="key_path",
                            question=f"Private key to access {parsed_kwargs['username']}@{host}?",
                            default=arg_cache.key_path,
                            kind="path",
                            cache=True,
                        ),
                        kwargs=kwargs,
                    )
                elif parsed_kwargs["auth_type"] == "password":
                    parsed_kwargs["password"] = ask(
                        question=Question(
                            var_name="password",
                            question=f"Password for {parsed_kwargs['username']}@{host}?",
                            kind="password",
                        ),
                        kwargs=kwargs,
                    )

            parsed_kwargs["repo"] = ask(
                question=Question(
                    var_name="repo",
                    question="Repo to fetch source from?",
                    default=arg_cache.repo,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            parsed_kwargs["branch"] = ask(
                Question(
                    var_name="branch",
                    question="Branch to monitor for updates?",
                    default=arg_cache.branch,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

            auth = None
            if host != "localhost":
                if parsed_kwargs["auth_type"] == "key":
                    auth = AuthCredentials(
                        username=parsed_kwargs["username"],
                        key_path=parsed_kwargs["key_path"],
                    )
                else:
                    auth = AuthCredentials(
                        username=parsed_kwargs["username"],
                        key_path=parsed_kwargs["password"],
                    )
                if not auth.valid:
                    raise Exception(f"Login Credentials are not valid. {auth}")
            parsed_kwargs["ansible_extras"] = kwargs["ansible_extras"]
            return create_launch_custom_cmd(verb=verb, auth=auth, kwargs=parsed_kwargs)
        else:
            errors = []
            if not DEPENDENCIES["ansible-playbook"]:
                errors.append("ansible-playbook")
            raise MissingDependency(
                f"Launching a Custom VM requires: {' '.join(errors)}"
            )

    host_options = ", ".join(allowed_hosts)
    raise MissingDependency(
        f"Launch requires a correct host option, try: {host_options}"
    )


def create_launch_docker_cmd(
    verb: GrammarVerb,
    docker_version: str,
    kwargs: TypeDict[str, Any],
    tail: bool = True,
) -> str:
    host_term = verb.get_named_term_hostgrammar(name="host")
    node_name = verb.get_named_term_type(name="node_name")
    node_type = verb.get_named_term_type(name="node_type")

    snake_name = str(node_name.snake_input)
    tag = name_tag(name=str(node_name.input))

    if ART:
        hagrid()

    print(
        "Launching a "
        + str(node_type.input)
        + " PyGrid node on port "
        + str(host_term.free_port)
        + "!\n"
    )

    print("  - TYPE: " + str(node_type.input))
    print("  - NAME: " + str(snake_name))
    print("  - TAG: " + str(tag))
    print("  - PORT: " + str(host_term.free_port))
    print("  - DOCKER: " + docker_version)
    print("  - TAIL: " + str(tail))
    print("\n")

    version_string = kwargs["tag"]
    version_hash = "dockerhub"
    build = kwargs["build"]
    if "release" in kwargs and kwargs["release"] == "development":
        # force version to have -dev at the end in dev mode
        # during development we can use the latest beta version
        if version_string is None:
            version_string = GRID_SRC_VERSION[0]
        version_string += "-dev"
        version_hash = GRID_SRC_VERSION[1]
        if build is None:
            build = True
    else:
        if build is None:
            build = False

        # during production the default would be stable
        if version_string == "local":
            # this can be used in VMs in production to auto update from src
            version_string = GRID_SRC_VERSION[0]
            version_hash = GRID_SRC_VERSION[1]
            build = True
        elif version_string is None:
            version_string = "stable"

    use_blob_storage = "True"
    if str(node_type.input) == "network":
        use_blob_storage = "False"
    elif "use_blob_storage" in kwargs and kwargs["use_blob_storage"] is not None:
        use_blob_storage = str(str_to_bool(kwargs["use_blob_storage"]))

    envs = {
        "RELEASE": "production",
        "COMPOSE_DOCKER_CLI_BUILD": 1,
        "DOCKER_BUILDKIT": 1,
        "HTTP_PORT": int(host_term.free_port),
        "HTTPS_PORT": int(host_term.free_port_tls),
        "TRAEFIK_TAG": str(tag),
        "DOMAIN_NAME": str(snake_name),
        "NODE_TYPE": str(node_type.input),
        "TRAEFIK_PUBLIC_NETWORK_IS_EXTERNAL": "False",
        "VERSION": version_string,
        "VERSION_HASH": version_hash,
        "USE_BLOB_STORAGE": use_blob_storage,
        "STACK_API_KEY": str(
            generate_sec_random_password(length=48, special_chars=False)
        ),
    }

    if "platform" in kwargs and kwargs["platform"] is not None:
        envs["DOCKER_DEFAULT_PLATFORM"] = kwargs["platform"]

    if "tls" in kwargs and kwargs["tls"] is True and len(kwargs["cert_store_path"]) > 0:
        envs["TRAEFIK_TLS_CERTS"] = kwargs["cert_store_path"]

    if (
        "tls" in kwargs
        and kwargs["tls"] is True
        and "test" in kwargs
        and kwargs["test"] is True
    ):
        envs["IGNORE_TLS_ERRORS"] = "True"

    if "test" in kwargs and kwargs["test"] is True:
        envs["S3_VOLUME_SIZE_MB"] = "100"  # GitHub CI is small

    if "release" in kwargs:
        envs["RELEASE"] = kwargs["release"]

    cmd = ""
    args = []
    for k, v in envs.items():
        if is_windows():
            # powershell envs
            quoted = f"'{v}'" if not isinstance(v, int) else v
            args.append(f"$env:{k}={quoted}")
        else:
            args.append(f"{k}={v}")
    if is_windows():
        cmd += "; ".join(args)
        cmd += "; "
    else:
        cmd += " ".join(args)

    if not build:
        pull_cmd = str(cmd)
        pull_cmd += " docker compose pull"

    cmd += " docker compose -p " + snake_name
    if str(node_type.input) == "network":
        cmd += " --profile network"
    else:
        cmd += " --profile blob-storage"

    # network frontend disabled
    if str(node_type.input) != "network" and kwargs["headless"] is False:
        cmd += " --profile frontend"

    cmd += " --file docker-compose.yml"
    if build:
        cmd += " --file docker-compose.build.yml"
    if "release" in kwargs and kwargs["release"] == "development":
        cmd += " --file docker-compose.dev.yml"
    if "tls" in kwargs and kwargs["tls"] is True:
        cmd += " --file docker-compose.tls.yml"
    if "test" in kwargs and kwargs["test"] is True:
        cmd += " --file docker-compose.test.yml"
    cmd += " up"

    if not tail:
        cmd += " -d"

    if build:
        cmd += " --build"  # force rebuild
    else:
        if is_windows():
            cmd = pull_cmd + "; " + cmd
        else:
            cmd = pull_cmd + " && " + cmd

    return cmd


def create_launch_vagrant_cmd(verb: GrammarVerb) -> str:
    host_term = verb.get_named_term_hostgrammar(name="host")
    node_name = verb.get_named_term_type(name="node_name")
    node_type = verb.get_named_term_type(name="node_type")

    snake_name = str(node_name.snake_input)

    if ART:
        hagrid()

    print(
        "Launching a "
        + str(node_type.input)
        + " PyGrid node on port "
        + str(host_term.port)
        + "!\n"
    )

    print("  - TYPE: " + str(node_type.input))
    print("  - NAME: " + str(snake_name))
    print("  - PORT: " + str(host_term.port))
    # print("  - VAGRANT: " + "1")
    # print("  - VIRTUALBOX: " + "1")
    print("\n")

    cmd = ""
    cmd += 'ANSIBLE_ARGS="'
    cmd += f"-e 'node_name={snake_name}'"
    cmd += f"-e 'node_type={node_type.input}'"
    cmd += '" '
    cmd += "vagrant up --provision"
    cmd = "cd " + GRID_SRC_PATH + ";" + cmd
    return cmd


def get_or_make_resource_group(resource_group: str, location: str = "westus") -> None:
    cmd = f"az group show --resource-group {resource_group}"
    exists = True
    try:
        subprocess.check_call(cmd, shell=True)  # nosec
    except Exception:  # nosec
        # group doesn't exist so lets create it
        exists = False

    if not exists:
        cmd = f"az group create -l {location} -n {resource_group}"
        try:
            print(f"Creating resource group.\nRunning: {cmd}")
            subprocess.check_call(cmd, shell=True)  # nosec
        except Exception as e:
            raise Exception(
                f"Unable to create resource group {resource_group} @ {location}. {e}"
            )


def extract_host_ip(stdout: bytes) -> Optional[str]:
    output = stdout.decode("utf-8")

    try:
        j = json.loads(output)
        if "publicIpAddress" in j:
            return str(j["publicIpAddress"])
    except Exception:  # nosec
        matcher = r'publicIpAddress":\s+"(.+)"'
        ips = re.findall(matcher, output)
        if len(ips) > 0:
            return ips[0]

    return None


def get_vm_host_ips(node_name: str, resource_group: str) -> Optional[TypeList]:
    cmd = f"az vm list-ip-addresses -g {resource_group} --query "
    cmd += f""""[?starts_with(virtualMachine.name, '{node_name}')]"""
    cmd += '''.virtualMachine.network.publicIpAddresses[0].ipAddress"'''
    output = subprocess.check_output(cmd, shell=True)  # nosec
    try:
        host_ips = json.loads(output)
        return host_ips
    except Exception as e:
        print(f"Failed to extract ips: {e}")

    return None


def is_valid_ip(host_or_ip: str) -> bool:
    matcher = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
    ips = re.findall(matcher, host_or_ip.strip())
    if len(ips) == 1:
        return True
    return False


def extract_host_ip_gcp(stdout: bytes) -> Optional[str]:
    output = stdout.decode("utf-8")

    try:
        matcher = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
        ips = re.findall(matcher, output)
        if len(ips) == 2:
            return ips[1]
    except Exception:  # nosec
        pass

    return None


def extract_host_ip_from_cmd(cmd: str) -> Optional[str]:

    try:
        matcher = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
        ips = re.findall(matcher, cmd)
        if ips:
            return ips[0]
    except Exception:  # nosec
        pass

    return None


def check_ip_for_ssh(
    host_ip: str, timeout: int = 600, wait_time: int = 5, silent: bool = False
) -> bool:

    if not silent:
        print(f"Checking VM at {host_ip} is up")
    checks = int(timeout / wait_time)  # 10 minutes in 5 second chunks
    first_run = True
    while checks > 0:
        checks -= 1
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(wait_time)
            result = sock.connect_ex((host_ip, 22))
            sock.close()
            if result == 0:
                if not silent:
                    print(f"VM at {host_ip} is up!")
                return True
            else:
                if first_run:
                    if not silent:
                        print("Waiting for VM to start", end="", flush=True)
                    first_run = False
                else:
                    if not silent:
                        print(".", end="", flush=True)
        except Exception:  # nosec
            pass
    return False


def make_vm_azure(
    node_name: str,
    resource_group: str,
    username: str,
    password: Optional[str],
    key_path: Optional[str],
    size: str,
    image_name: str,
    node_count: int,
) -> TypeList:
    disk_size_gb = "200"
    try:
        temp_dir = tempfile.TemporaryDirectory()
        public_key_path = (
            private_to_public_key(
                private_key_path=key_path, temp_path=temp_dir.name, username=username
            )
            if key_path
            else None
        )
    except Exception:  # nosec
        temp_dir.cleanup()

    authentication_type = "ssh" if key_path else "password"
    cmd = f"az vm create -n {node_name} -g {resource_group} --size {size} "
    cmd += f"--image {image_name} --os-disk-size-gb {disk_size_gb} "
    cmd += f"--public-ip-sku Standard --authentication-type {authentication_type} --admin-username {username} "
    cmd += f"--ssh-key-values {public_key_path} " if public_key_path else ""
    cmd += f"--admin-password '{password}' " if password else ""
    cmd += f"--count {node_count} " if node_count > 1 else ""

    host_ips: Optional[TypeList] = []
    try:
        print(f"Creating vm.\nRunning: {hide_azure_vm_password(cmd)}")
        subprocess.check_output(cmd, shell=True)  # nosec
        host_ips = get_vm_host_ips(node_name=node_name, resource_group=resource_group)
    except Exception as e:
        print("failed", e)
    finally:
        temp_dir.cleanup()

    if not host_ips:
        raise Exception("Failed to create vm or get VM public ip")

    try:
        # clean up temp public key
        if public_key_path:
            os.unlink(public_key_path)
    except Exception:  # nosec
        pass

    return host_ips


def open_port_vm_azure(
    resource_group: str, node_name: str, port_name: str, port: int, priority: int
) -> None:
    cmd = f"az network nsg rule create --resource-group {resource_group} "
    cmd += f"--nsg-name {node_name}NSG --name {port_name} --destination-port-ranges {port} --priority {priority}"
    try:
        print(f"Creating {port_name} {port} ngs rule.\nRunning: {cmd}")
        output = subprocess.check_call(cmd, shell=True)  # nosec
        print("output", output)
        pass
    except Exception as e:
        print("failed", e)


def create_project(project_id: str) -> None:
    cmd = f"gcloud projects create {project_id} --set-as-default"
    try:
        print(f"Creating project.\nRunning: {cmd}")
        subprocess.check_call(cmd, shell=True)  # nosec
    except Exception as e:
        print("failed", e)

    print("create project complete")


def create_launch_gcp_cmd(
    verb: GrammarVerb,
    project_id: str,
    zone: str,
    machine_type: str,
    ansible_extras: str,
    kwargs: TypeDict[str, Any],
    repo: str,
    branch: str,
    auth: AuthCredentials,
) -> str:
    # create project if it doesn't exist
    create_project(project_id)
    # vm
    node_name = verb.get_named_term_type(name="node_name")
    kebab_name = str(node_name.kebab_input)
    disk_size_gb = "200"
    host_ip = make_gcp_vm(
        vm_name=kebab_name,
        project_id=project_id,
        zone=zone,
        machine_type=machine_type,
        disk_size_gb=disk_size_gb,
    )

    # get old host
    host_term = verb.get_named_term_hostgrammar(name="host")

    host_up = check_ip_for_ssh(host_ip=host_ip)
    if not host_up:
        raise Exception(f"Something went wrong launching the VM at IP: {host_ip}.")

    if "provision" in kwargs and not kwargs["provision"]:
        print("Skipping automatic provisioning.")
        print("VM created with:")
        print(f"IP: {host_ip}")
        print(f"User: {auth.username}")
        print(f"Key: {auth.key_path}")
        print("\nConnect with:")
        print(f"ssh -i {auth.key_path} {auth.username}@{host_ip}")
        sys.exit(0)

    # replace
    host_term.parse_input(host_ip)
    verb.set_named_term_type(name="host", new_term=host_term)

    extra_kwargs = {
        "repo": repo,
        "branch": branch,
        "auth_type": "key",
        "ansible_extras": ansible_extras,
    }
    kwargs.update(extra_kwargs)

    # provision
    return create_launch_custom_cmd(verb=verb, auth=auth, kwargs=kwargs)


def make_gcp_vm(
    vm_name: str, project_id: str, zone: str, machine_type: str, disk_size_gb: str
) -> str:
    create_cmd = "gcloud compute instances create"
    network_settings = "network=default,network-tier=PREMIUM"
    maintenance_policy = "MIGRATE"
    scopes = [
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring.write",
        "https://www.googleapis.com/auth/servicecontrol",
        "https://www.googleapis.com/auth/service.management.readonly",
        "https://www.googleapis.com/auth/trace.append",
    ]
    tags = "http-server,https-server"
    disk_image = "projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20220308"
    disk = (
        f"auto-delete=yes,boot=yes,device-name={vm_name},image={disk_image},"
        + f"mode=rw,size={disk_size_gb},type=pd-ssd"
    )
    security_flags = (
        "--no-shielded-secure-boot --shielded-vtpm "
        + "--shielded-integrity-monitoring --reservation-affinity=any"
    )

    cmd = (
        f"{create_cmd} {vm_name} "
        + f"--project={project_id} "
        + f"--zone={zone} "
        + f"--machine-type={machine_type} "
        + f"--create-disk={disk} "
        + f"--network-interface={network_settings} "
        + f"--maintenance-policy={maintenance_policy} "
        + f"--scopes={','.join(scopes)} --tags={tags} "
        + f"{security_flags}"
    )

    host_ip = None
    try:
        print(f"Creating vm.\nRunning: {cmd}")
        output = subprocess.check_output(cmd, shell=True)  # nosec
        host_ip = extract_host_ip_gcp(stdout=output)
    except Exception as e:
        print("failed", e)

    if host_ip is None:
        raise Exception("Failed to create vm or get VM public ip")

    return host_ip


def create_launch_azure_cmd(
    verb: GrammarVerb,
    resource_group: str,
    location: str,
    size: str,
    username: str,
    password: Optional[str],
    key_path: Optional[str],
    repo: str,
    branch: str,
    auth: AuthCredentials,
    ansible_extras: str,
    kwargs: TypeDict[str, Any],
) -> TypeList[str]:

    get_or_make_resource_group(resource_group=resource_group, location=location)

    node_count = kwargs.get("node_count", 1)
    print("Total VMs to create: ", node_count)

    # vm
    node_name = verb.get_named_term_type(name="node_name")
    snake_name = str(node_name.snake_input)
    image_name = get_azure_image(kwargs["image_name"])
    host_ips = make_vm_azure(
        snake_name,
        resource_group,
        username,
        password,
        key_path,
        size,
        image_name,
        node_count,
    )

    # open port 80
    open_port_vm_azure(
        resource_group=resource_group,
        node_name=snake_name,
        port_name="HTTP",
        port=80,
        priority=500,
    )

    # open port 443
    open_port_vm_azure(
        resource_group=resource_group,
        node_name=snake_name,
        port_name="HTTPS",
        port=443,
        priority=501,
    )

    if kwargs["jupyter"]:
        # open port 8888
        open_port_vm_azure(
            resource_group=resource_group,
            node_name=snake_name,
            port_name="Jupyter",
            port=8888,
            priority=502,
        )

    launch_cmds: TypeList[str] = []

    for host_ip in host_ips:
        # get old host
        host_term = verb.get_named_term_hostgrammar(name="host")

        # replace
        host_term.parse_input(host_ip)
        verb.set_named_term_type(name="host", new_term=host_term)

        if "provision" in kwargs and not kwargs["provision"]:
            print("Skipping automatic provisioning.")
            print("VM created with:")
            print(f"Name: {snake_name}")
            print(f"IP: {host_ip}")
            print(f"User: {username}")
            print(f"Password: {password}")
            print(f"Key: {key_path}")
            print("\nConnect with:")
            if kwargs["auth_type"] == "key":
                print(f"ssh -i {key_path} {username}@{host_ip}")
            else:
                print(f"ssh {username}@{host_ip}")
        else:
            extra_kwargs = {
                "repo": repo,
                "branch": branch,
                "ansible_extras": ansible_extras,
            }
            kwargs.update(extra_kwargs)

            # provision
            host_up = check_ip_for_ssh(host_ip=host_ip)
            if not host_up:
                print(f"Warning: {host_ip} ssh not available yet")
            launch_cmd = create_launch_custom_cmd(verb=verb, auth=auth, kwargs=kwargs)
            launch_cmds.append(launch_cmd)

    return launch_cmds


def create_ansible_land_cmd(
    verb: GrammarVerb, auth: Optional[AuthCredentials], kwargs: TypeDict[str, Any]
) -> str:
    try:
        host_term = verb.get_named_term_hostgrammar(name="host")
        print("Landing PyGrid node on port " + str(host_term.port) + "!\n")

        print("  - PORT: " + str(host_term.port))
        print("\n")

        playbook_path = GRID_SRC_PATH + "/ansible/site.yml"
        ansible_cfg_path = GRID_SRC_PATH + "/ansible.cfg"
        auth = cast(AuthCredentials, auth)

        if not os.path.exists(playbook_path):
            print(f"Can't find playbook site.yml at: {playbook_path}")
        cmd = f"ANSIBLE_CONFIG={ansible_cfg_path} ansible-playbook "
        if host_term.host == "localhost":
            cmd += "--connection=local "
        cmd += f"-i {host_term.host}, {playbook_path}"
        if host_term.host != "localhost" and kwargs["auth_type"] == "key":
            cmd += f" --private-key {auth.key_path} --user {auth.username}"
        elif host_term.host != "localhost" and kwargs["auth_type"] == "password":
            cmd += f" -c paramiko --user {auth.username}"

        ANSIBLE_ARGS = {"install": "false"}

        if host_term.host != "localhost" and kwargs["auth_type"] == "password":
            ANSIBLE_ARGS["ansible_ssh_pass"] = kwargs["password"]

        if host_term.host == "localhost":
            ANSIBLE_ARGS["local"] = "true"

        if "ansible_extras" in kwargs and kwargs["ansible_extras"] != "":
            options = kwargs["ansible_extras"].split(",")
            for option in options:
                parts = option.strip().split("=")
                if len(parts) == 2:
                    ANSIBLE_ARGS[parts[0]] = parts[1]

        for k, v in ANSIBLE_ARGS.items():
            cmd += f" -e \"{k}='{v}'\""

        cmd = "cd " + GRID_SRC_PATH + ";" + cmd
        return cmd
    except Exception as e:
        print(f"Failed to construct custom deployment cmd: {cmd}. {e}")
        raise e


def create_launch_custom_cmd(
    verb: GrammarVerb, auth: Optional[AuthCredentials], kwargs: TypeDict[str, Any]
) -> str:
    try:
        host_term = verb.get_named_term_hostgrammar(name="host")
        node_name = verb.get_named_term_type(name="node_name")
        node_type = verb.get_named_term_type(name="node_type")
        # source_term = verb.get_named_term_type(name="source")

        snake_name = str(node_name.snake_input)

        if ART:
            hagrid()

        print(
            "Launching a "
            + str(node_type.input)
            + " PyGrid node on port "
            + str(host_term.port)
            + "!\n"
        )

        print("  - TYPE: " + str(node_type.input))
        print("  - NAME: " + str(snake_name))
        print("  - PORT: " + str(host_term.port))
        print("\n")

        playbook_path = GRID_SRC_PATH + "/ansible/site.yml"
        ansible_cfg_path = GRID_SRC_PATH + "/ansible.cfg"
        auth = cast(AuthCredentials, auth)

        if not os.path.exists(playbook_path):
            print(f"Can't find playbook site.yml at: {playbook_path}")
        cmd = f"ANSIBLE_CONFIG={ansible_cfg_path} ansible-playbook "
        if host_term.host == "localhost":
            cmd += "--connection=local "
        cmd += f"-i {host_term.host}, {playbook_path}"
        if host_term.host != "localhost" and kwargs["auth_type"] == "key":
            cmd += f" --private-key {auth.key_path} --user {auth.username}"
        elif host_term.host != "localhost" and kwargs["auth_type"] == "password":
            cmd += f" -c paramiko --user {auth.username}"

        version_string = kwargs["tag"]
        if version_string is None:
            version_string = "local"

        ANSIBLE_ARGS = {
            "node_type": node_type.input,
            "node_name": snake_name,
            "github_repo": kwargs["repo"],
            "repo_branch": kwargs["branch"],
            "docker_tag": version_string,
        }

        if host_term.host != "localhost" and kwargs["auth_type"] == "password":
            ANSIBLE_ARGS["ansible_ssh_pass"] = kwargs["password"]

        if host_term.host == "localhost":
            ANSIBLE_ARGS["local"] = "true"

        if kwargs["tls"] is True:
            ANSIBLE_ARGS["tls"] = "true"

        if "release" in kwargs:
            ANSIBLE_ARGS["release"] = kwargs["release"]

        if (
            kwargs["tls"] is True
            and "cert_store_path" in kwargs
            and len(kwargs["cert_store_path"]) > 0
        ):
            ANSIBLE_ARGS["cert_store_path"] = kwargs["cert_store_path"]

        if (
            kwargs["tls"] is True
            and "upload_tls_key" in kwargs
            and len(kwargs["upload_tls_key"]) > 0
        ):
            ANSIBLE_ARGS["upload_tls_key"] = kwargs["upload_tls_key"]

        if (
            kwargs["tls"] is True
            and "upload_tls_cert" in kwargs
            and len(kwargs["upload_tls_cert"]) > 0
        ):
            ANSIBLE_ARGS["upload_tls_cert"] = kwargs["upload_tls_cert"]

        if kwargs["jupyter"] is True:
            ANSIBLE_ARGS["jupyter"] = "true"
            ANSIBLE_ARGS["jupyter_token"] = generate_sec_random_password(
                length=48, upper_case=False, special_chars=False
            )

        if "ansible_extras" in kwargs and kwargs["ansible_extras"] != "":
            options = kwargs["ansible_extras"].split(",")
            for option in options:
                parts = option.strip().split("=")
                if len(parts) == 2:
                    ANSIBLE_ARGS[parts[0]] = parts[1]

        # if mode == "deploy":
        #     ANSIBLE_ARGS["deploy"] = "true"

        for k, v in ANSIBLE_ARGS.items():
            cmd += f" -e \"{k}='{v}'\""

        cmd = "cd " + GRID_SRC_PATH + ";" + cmd
        return cmd
    except Exception as e:
        print(f"Failed to construct custom deployment cmd: {cmd}. {e}")
        raise e


def create_land_cmd(verb: GrammarVerb, kwargs: TypeDict[str, Any]) -> str:
    host_term = verb.get_named_term_hostgrammar(name="host")
    host = host_term.host if host_term.host is not None else ""

    if host in ["docker"]:
        if verb.get_named_term_grammar("node_name").input == "all":
            # subprocess.call("docker rm `docker ps -aq` --force", shell=True) # nosec
            return "docker rm `docker ps -aq` --force"

        version = check_docker_version()
        if version:
            return create_land_docker_cmd(verb=verb)
    elif host == "localhost" or is_valid_ip(host):
        parsed_kwargs = {}
        if DEPENDENCIES["ansible-playbook"]:
            if host != "localhost":
                parsed_kwargs["username"] = ask(
                    question=Question(
                        var_name="username",
                        question=f"Username for {host} with sudo privledges?",
                        default=arg_cache.username,
                        kind="string",
                        cache=True,
                    ),
                    kwargs=kwargs,
                )
                parsed_kwargs["auth_type"] = ask(
                    question=Question(
                        var_name="auth_type",
                        question="Do you want to login with a key or password",
                        default=arg_cache.auth_type,
                        kind="option",
                        options=["key", "password"],
                        cache=True,
                    ),
                    kwargs=kwargs,
                )
                if parsed_kwargs["auth_type"] == "key":
                    parsed_kwargs["key_path"] = ask(
                        question=Question(
                            var_name="key_path",
                            question=f"Private key to access {parsed_kwargs['username']}@{host}?",
                            default=arg_cache.key_path,
                            kind="path",
                            cache=True,
                        ),
                        kwargs=kwargs,
                    )
                elif parsed_kwargs["auth_type"] == "password":
                    parsed_kwargs["password"] = ask(
                        question=Question(
                            var_name="password",
                            question=f"Password for {parsed_kwargs['username']}@{host}?",
                            kind="password",
                        ),
                        kwargs=kwargs,
                    )

            auth = None
            if host != "localhost":
                if parsed_kwargs["auth_type"] == "key":
                    auth = AuthCredentials(
                        username=parsed_kwargs["username"],
                        key_path=parsed_kwargs["key_path"],
                    )
                else:
                    auth = AuthCredentials(
                        username=parsed_kwargs["username"],
                        key_path=parsed_kwargs["password"],
                    )
                if not auth.valid:
                    raise Exception(f"Login Credentials are not valid. {auth}")
            parsed_kwargs["ansible_extras"] = kwargs["ansible_extras"]
            return create_ansible_land_cmd(verb=verb, auth=auth, kwargs=parsed_kwargs)
        else:
            errors = []
            if not DEPENDENCIES["ansible-playbook"]:
                errors.append("ansible-playbook")
            raise MissingDependency(
                f"Launching a Custom VM requires: {' '.join(errors)}"
            )

    host_options = ", ".join(allowed_hosts)
    raise MissingDependency(
        f"Launch requires a correct host option, try: {host_options}"
    )


def create_land_docker_cmd(verb: GrammarVerb) -> str:
    node_name = verb.get_named_term_type(name="node_name")
    snake_name = str(node_name.snake_input)

    cmd = ""
    cmd += "docker compose"
    cmd += ' --file "docker-compose.yml"'
    cmd += ' --project-name "' + snake_name + '"'
    cmd += " down"

    cmd = "cd " + GRID_SRC_PATH + ";export $(cat .env | sed 's/#.*//g' | xargs);" + cmd
    return cmd


@click.command(help="Stop a running PyGrid domain/network node.")
@click.argument("args", type=str, nargs=-1)
@click.option(
    "--cmd",
    default="false",
    required=False,
    type=str,
    help="Optional: print the cmd without running it",
)
@click.option(
    "--ansible_extras",
    default="",
    type=str,
)
@click.option(
    "--build_src",
    default=DEFAULT_BRANCH,
    required=False,
    type=str,
    help="Optional: git branch to use for launch / build operations",
)
def land(args: TypeTuple[str], **kwargs: TypeDict[str, Any]) -> None:
    verb = get_land_verb()

    try:
        grammar = parse_grammar(args=args, verb=verb)
        verb.load_grammar(grammar=grammar)
    except BadGrammar as e:
        print(e)
        return

    try:
        update_repo(repo=GIT_REPO, branch=str(kwargs["build_src"]))
    except Exception as e:
        print(f"Failed to update repo. {e}")

    try:
        cmd = create_land_cmd(verb=verb, kwargs=kwargs)
    except Exception as e:
        print(f"{e}")
        return
    print("Running: \n", hide_password(cmd=cmd))

    if "cmd" not in kwargs or str_to_bool(cast(str, kwargs["cmd"])) is False:
        print("Running: \n", cmd)
        try:
            subprocess.call(cmd, shell=True, cwd=GRID_SRC_PATH)  # nosec
        except Exception as e:
            print(f"Failed to run cmd: {cmd}. {e}")


cli.add_command(launch)
cli.add_command(land)
cli.add_command(clean)


@click.command(help="Show HAGrid debug information")
@click.argument("args", type=str, nargs=-1)
def debug(args: TypeTuple[str], **kwargs: TypeDict[str, Any]) -> None:
    now = datetime.now().astimezone()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S %Z")
    debug_info: TypeDict[str, Any] = {}
    debug_info["datetime"] = dt_string
    debug_info["python_binary"] = sys.executable
    debug_info["dependencies"] = DEPENDENCIES
    debug_info["environment"] = ENVIRONMENT
    debug_info["hagrid"] = __version__
    debug_info["hagrid_dev"] = EDITABLE_MODE
    debug_info["hagrid_path"] = hagrid_root()
    debug_info["hagrid_repo_sha"] = commit_hash()
    debug_info["docker"] = docker_info()
    if is_windows():
        debug_info["wsl"] = wsl_info()
        debug_info["wsl_linux"] = wsl_linux_info()
    print("\n\nWhen reporting bugs, please copy everything between the lines.")
    print("==================================================================\n")
    print(json.dumps(debug_info))
    print("\n=================================================================\n\n")


cli.add_command(debug)

DEFAULT_HEALTH_CHECKS = ["host", "UI (βeta)", "api", "ssh", "jupyter"]
HEALTH_CHECK_FUNCTIONS = {
    "host": check_host,
    "UI (βeta)": check_login_page,
    "api": check_api_metadata,
    "ssh": check_ip_for_ssh,
    "jupyter": check_jupyter_server,
}

HEALTH_CHECK_ICONS = {
    "host": "🔌",
    "UI (βeta)": "🖱",
    "api": "⚙️",
    "ssh": "🔐",
    "jupyter": "📗",
}

HEALTH_CHECK_URLS = {
    "host": "{ip_address}",
    "UI (βeta)": "http://{ip_address}/login",
    "api": "http://{ip_address}/api/v1",
    "ssh": "hagrid ssh {ip_address}",
    "jupyter": "http://{ip_address}:8888",
}


def check_host_health(ip_address: str, keys: TypeList[str]) -> TypeDict[str, bool]:
    status = {}
    for key in keys:
        func: Callable = HEALTH_CHECK_FUNCTIONS[key]  # type: ignore
        status[key] = func(ip_address, silent=True)
    return status


def icon_status(status: bool) -> str:
    return "✅" if status else "❌"


def get_health_checks(ip_address: str) -> TypeTuple[bool, TypeList[TypeList[str]]]:
    keys = list(DEFAULT_HEALTH_CHECKS)
    if "localhost" in ip_address:
        new_keys = []
        for key in keys:
            if key not in ["host", "jupyter", "ssh"]:
                new_keys.append(key)
        keys = new_keys

    health_status = check_host_health(ip_address=ip_address, keys=keys)
    complete_status = all(health_status.values())

    # figure out how to add this back?
    # console.print("[bold magenta]Checking host:[/bold magenta]", ip_address, ":mage:")
    table_contents = []
    for key, value in health_status.items():
        table_contents.append(
            [
                HEALTH_CHECK_ICONS[key],
                key,
                HEALTH_CHECK_URLS[key].replace("{ip_address}", ip_address),
                icon_status(value),
            ]
        )

    return complete_status, table_contents


def create_check_table(
    table_contents: TypeList[TypeList[str]], time_left: int = 0
) -> rich.table.Table:
    table = rich.table.Table()
    table.add_column("PyGrid", style="magenta")
    table.add_column("Info", justify="left")
    time_left_str = "" if time_left == 0 else str(time_left)
    table.add_column(time_left_str, justify="left")
    for row in table_contents:
        table.add_row(row[1], row[2], row[3])
    return table


@click.command(help="Check health of an IP address/addresses or a resource group")
@click.argument("ip_addresses", type=str, nargs=-1)
@click.option(
    "--wait",
    is_flag=True,
    help="Optional: wait until checks pass",
)
@click.option(
    "--silent",
    is_flag=True,
    help="Optional: don't refresh output during wait",
)
def check(
    ip_addresses: TypeList[str], wait: bool = False, silent: bool = False
) -> None:
    console = rich.get_console()
    if len(ip_addresses) == 0:
        headers = {"User-Agent": "curl/7.79.1"}
        print("Detecting External IP...")
        ip_res = requests.get("https://ifconfig.co", headers=headers)
        ip_address = ip_res.text.strip()
        ip_addresses = [ip_address]

    if len(ip_addresses) == 1:
        ip_address = ip_addresses[0]
        status, table_contents = get_health_checks(ip_address=ip_address)
        table = create_check_table(table_contents=table_contents)
        max_timeout = 600
        if wait and not status:
            table = create_check_table(
                table_contents=table_contents, time_left=max_timeout
            )
            if silent:
                print("Checking...")
            while not status:
                if not silent:
                    with Live(table, refresh_per_second=4, screen=True) as live:
                        max_timeout -= 1
                        if max_timeout % 5 == 0:
                            status, table_contents = get_health_checks(ip_address)
                        table = create_check_table(
                            table_contents=table_contents, time_left=max_timeout
                        )
                        live.update(table)
                        if status:
                            break
                        time.sleep(1)
                else:
                    max_timeout -= 1
                    if max_timeout % 5 == 0:
                        status, table_contents = get_health_checks(ip_address)
                    table = create_check_table(
                        table_contents=table_contents, time_left=max_timeout
                    )
                    time.sleep(1)
        console.print(table)
    else:
        for ip_address in ip_addresses:
            _, table_contents = get_health_checks(ip_address)
            table = create_check_table(table_contents=table_contents)
            console.print(table)


cli.add_command(check)


# add Hagrid info to the cli
@click.command(help="Show Hagrid info")
def version() -> None:
    print(f"Hagrid version: {__version__}")


cli.add_command(version)


def ssh_into_remote_machine(
    host_ip: str,
    username: str,
    auth_type: str,
    private_key_path: Optional[str],
    cmd: str = "",
) -> None:
    """Access or execute command on the remote machine.

    Args:
        host_ip (str): ip address of the VM
        private_key_path (str): private key of the VM
        username (str): username on the VM
        cmd (str, optional): Command to execute on the remote machine. Defaults to "".
    """
    try:
        if auth_type == "key":
            subprocess.call(  # nosec
                ["ssh", "-i", f"{private_key_path}", f"{username}@{host_ip}", cmd]
            )
        elif auth_type == "password":
            subprocess.call(["ssh", f"{username}@{host_ip}", cmd])  # nosec
    except Exception as e:
        raise e


@click.command(help="SSH into the IP address or a resource group")
@click.argument("ip_address", type=str)
@click.option(
    "--cmd",
    type=str,
    required=False,
    default="",
    help="Optional: command to execute on the remote machine.",
)
def ssh(ip_address: str, cmd: str) -> None:
    kwargs: TypeDict = {}
    key_path: Optional[str] = None

    if check_ip_for_ssh(ip_address, timeout=10, silent=False):
        username = ask(
            question=Question(
                var_name="azure_username",
                question="What is the username for the VM?",
                default=arg_cache.azure_username,
                kind="string",
                cache=True,
            ),
            kwargs=kwargs,
        )
        auth_type = ask(
            question=Question(
                var_name="auth_type",
                question="Do you want to login with a key or password",
                default=arg_cache.auth_type,
                kind="option",
                options=["key", "password"],
                cache=True,
            ),
            kwargs=kwargs,
        )

        if auth_type == "key":
            key_path = ask(
                question=Question(
                    var_name="azure_key_path",
                    question="Absolute path to the private key of the VM?",
                    default=arg_cache.azure_key_path,
                    kind="string",
                    cache=True,
                ),
                kwargs=kwargs,
            )

        # SSH into the remote and execute the command
        ssh_into_remote_machine(
            host_ip=ip_address,
            username=username,
            auth_type=auth_type,
            private_key_path=key_path,
            cmd=cmd,
        )


cli.add_command(ssh)
