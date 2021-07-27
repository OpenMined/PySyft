# stdlib
import hashlib
import json
import os
import re
import subprocess
from typing import Any
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple

# third party
import click

# relative
from .art import hagrid
from .art import motorcycle
from .auth import AuthCredentials
from .cache import arg_cache
from .deps import DEPENDENCIES
from .deps import MissingDependency
from .grammar import BadGrammar
from .grammar import GrammarVerb
from .grammar import parse_grammar
from .launch import get_launch_verb
from .lib import GRID_SRC_PATH
from .lib import check_docker_version
from .lib import name_tag
from .style import RichGroup


@click.group(cls=RichGroup)
def cli():
    pass


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
# @click.option(
#     "--password",
#     default=None,
#     required=False,
#     type=str,
#     help="Optional: the password for provisioning the remote host",
# )
# @click.option(
#     "--mode",
#     default=None,
#     required=False,
#     type=str,
#     help="Optional: mode either provision or deploy, where deploy is a quick code update",
# )
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
def launch(args: TypeTuple[str], **kwargs: TypeDict[str, Any]):
    verb = get_launch_verb()
    try:
        grammar = parse_grammar(args=args, verb=verb)
        verb.load_grammar(grammar=grammar)
    except BadGrammar as e:
        print(e)
        return

    try:
        cmd = create_launch_cmd(verb=verb, kwargs=kwargs)
    except Exception as e:
        print(f"{e}")
        return
    print("Running: \n", cmd)
    subprocess.call(cmd, shell=True)


class QuestionInputError(Exception):
    pass


class Question:
    def __init__(
        self,
        var_name: str,
        question: str,
        kind: str,
        default: Optional[str] = None,
        cache: bool = False,
    ) -> None:
        self.var_name = var_name
        self.question = question
        self.default = default
        self.kind = kind
        self.cache = cache

    def validate(self, value: str) -> str:
        if self.default is not None and value == "":
            return self.default

        if self.kind == "path":
            value = os.path.expanduser(value)
            if not os.path.exists(value):
                error = f"{value} is not a valid path."
                if self.default is not None:
                    error += f" Try {self.default}"
                raise QuestionInputError(error)

        if self.kind == "yesno":
            if value.lower().startswith("y"):
                return "y"
            elif value.lower().startswith("n"):
                return "n"

        return value


def ask(question: Question, kwargs: TypeDict[str, str]) -> TypeDict[str, Any]:
    if question.var_name in kwargs and kwargs[question.var_name] is not None:
        value = kwargs[question.var_name]
    else:
        if question.default is not None:
            value = click.prompt(question.question, type=str, default=question.default)
        else:
            value = click.prompt(question.question, type=str)

    value = question.validate(value=value)
    if question.cache:
        setattr(arg_cache, question.var_name, value)

    return value


def requires_kwargs(
    required: TypeList[Question], kwargs: TypeDict[str, str]
) -> TypeDict[str, Any]:

    parsed_kwargs = {}
    for question in required:
        if question.var_name in kwargs and kwargs[question.var_name] is not None:
            value = kwargs[question.var_name]
        else:
            if question.default is not None:
                value = click.prompt(
                    question.question, type=str, default=question.default
                )
            else:
                value = click.prompt(question.question, type=str)

        value = question.validate(value=value)
        if question.cache:
            setattr(arg_cache, question.var_name, value)

        parsed_kwargs[question.var_name] = value
    return parsed_kwargs


def private_to_public_key(private_key_path: str, username: str) -> str:
    output_path = f"/tmp/hagrid_{username}_key.pub"
    cmd = f"ssh-keygen -f {private_key_path} -y > {output_path}"
    try:
        subprocess.check_call(cmd, shell=True)
    except Exception as e:
        print("failed to make ssh key", e)
        raise e
    return output_path


def check_azure_authed() -> bool:
    cmd = "az account show"
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)
        return True
    except Exception:
        pass
    return False


def login_azure() -> bool:
    cmd = "az login"
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)
        return True
    except Exception:
        pass
    return False


def create_launch_cmd(verb: GrammarVerb, kwargs: TypeDict[str, Any]) -> str:
    host_term = verb.get_named_term_type(name="host")
    host = host_term.host
    if host in ["docker"]:
        version = check_docker_version()
        if version:
            return create_launch_docker_cmd(verb=verb, docker_version=version)
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

            key_path = ask(
                Question(
                    var_name="azure_key_path",
                    question=f"Private key to access {username}@{host}?",
                    default=arg_cache.azure_key_path,
                    kind="path",
                    cache=True,
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

            auth = AuthCredentials(username=username, key_path=key_path)

            return create_launch_azure_cmd(
                verb=verb,
                resource_group=resource_group,
                location=location,
                size=size,
                username=username,
                key_path=key_path,
                repo=repo,
                branch=branch,
                auth=auth,
            )
        else:
            errors = []
            if not DEPENDENCIES["ansible-playbook"]:
                errors.append("ansible-playbook")
            raise MissingDependency(
                f"Launching a Cloud VM requires: {' '.join(errors)}"
            )
    elif host in ["aws", "gcp"]:
        print("Coming soon.")
        return
    else:
        if DEPENDENCIES["ansible-playbook"]:
            username_question = Question(
                var_name="username",
                question=f"Username for {host} with sudo privledges?",
                default=arg_cache.username,
                kind="string",
                cache=True,
            )
            key_path_question = Question(
                var_name="key_path",
                question=f"Private key to access [username]@{host}?",
                default=arg_cache.key_path,
                kind="path",
                cache=True,
            )
            repo_question = Question(
                var_name="repo",
                question="Repo to fetch source from?",
                default=arg_cache.repo,
                kind="string",
                cache=True,
            )
            branch_question = Question(
                var_name="branch",
                question="Branch to monitor for updates?",
                default=arg_cache.branch,
                kind="string",
                cache=True,
            )
            parsed_kwargs = requires_kwargs(
                required=[
                    username_question,
                    key_path_question,
                    repo_question,
                    branch_question,
                ],
                kwargs=kwargs,
            )

            auth = AuthCredentials(
                username=parsed_kwargs["username"], key_path=parsed_kwargs["key_path"]
            )
            if auth.valid:
                return create_launch_custom_cmd(
                    verb=verb, auth=auth, kwargs=parsed_kwargs
                )
        else:
            errors = []
            if not DEPENDENCIES["ansible-playbook"]:
                errors.append("ansible-playbook")
            raise MissingDependency(
                f"Launching a Custom VM requires: {' '.join(errors)}"
            )


def create_launch_docker_cmd(
    verb: GrammarVerb, docker_version: str, tail: bool = False
) -> str:
    host_term = verb.get_named_term_type(name="host")
    node_name = verb.get_named_term_type(name="node_name")
    node_type = verb.get_named_term_type(name="node_type")

    snake_name = node_name.input.lower().replace(" ", "_")
    tag = name_tag(name=node_name.input)

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
    print("\n")

    cmd = ""
    cmd += "DOMAIN_PORT=" + str(host_term.free_port)
    cmd += " TRAEFIK_TAG=" + str(tag)
    cmd += ' DOMAIN_NAME="' + snake_name + '"'
    cmd += " NODE_TYPE=" + node_type.input
    cmd += " docker compose -p " + snake_name
    cmd += " up"
    if not tail:
        cmd += " -d"
    cmd = "cd " + GRID_SRC_PATH + ";" + cmd
    return cmd


def create_launch_vagrant_cmd(verb: GrammarVerb) -> str:
    host_term = verb.get_named_term_type(name="host")
    node_name = verb.get_named_term_type(name="node_name")
    node_type = verb.get_named_term_type(name="node_type")

    snake_name = node_name.input.lower().replace(" ", "_")

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
        subprocess.check_call(cmd, shell=True)
    except Exception:
        # group doesnt exist so lets create it
        exists = False

    if not exists:
        cmd = f"az group create -l {location} -n {resource_group}"
        try:
            print(f"Creating resource group.\nRunning: {cmd}")
            subprocess.check_call(cmd, shell=True)
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
    except Exception:
        matcher = r'publicIpAddress":\s+"(.+)"'
        ips = re.findall(matcher, output)
        if len(ips) > 0:
            return ips[0]

    return None


def make_vm_azure(
    node_name: str, resource_group: str, username: str, key_path: str, size: str
) -> None:
    public_key_path = private_to_public_key(
        private_key_path=key_path, username=username
    )

    cmd = f"az vm create -n {node_name} -g {resource_group} --size {size} "
    cmd += "--image Canonical:0001-com-ubuntu-server-focal:20_04-lts:latest "
    cmd += "--public-ip-sku Standard --authentication-type ssh "
    cmd += f"--ssh-key-values {public_key_path} --admin-username {username}"
    try:
        print(f"Creating vm.\nRunning: {cmd}")
        output = subprocess.check_output(cmd, shell=True)
        host_ip = extract_host_ip(stdout=output)
    except Exception as e:
        print("failed", e)

    if host_ip is None:
        raise Exception("Failed to create vm or get VM public ip")

    try:
        # clean up temp public key
        os.unlink(public_key_path)
    except Exception:
        pass

    return host_ip


def open_port_vm_azure(resource_group: str, node_name: str, port: int) -> None:
    cmd = f"az network nsg rule create --resource-group {resource_group} "
    cmd += f"--nsg-name {node_name}NSG --name HTTP --destination-port-ranges {port} --priority 500"
    try:
        print(f"Creating ngs rule.\nRunning: {cmd}")
        output = subprocess.check_call(cmd, shell=True)
        print("output", output)
        pass
    except Exception as e:
        print("failed", e)


def create_launch_azure_cmd(
    verb: GrammarVerb,
    resource_group: str,
    location: str,
    size: str,
    username: str,
    key_path: str,
    repo: str,
    branch: str,
    auth=AuthCredentials,
) -> str:
    # resource group
    get_or_make_resource_group(resource_group=resource_group, location=location)

    # vm
    node_name = verb.get_named_term_type(name="node_name")
    snake_name = node_name.input.lower().replace(" ", "_")
    host_ip = make_vm_azure(snake_name, resource_group, username, key_path, size)

    # open port 80
    open_port_vm_azure(resource_group=resource_group, node_name=snake_name, port=80)

    # get old host
    host_term = verb.get_named_term_type(name="host")

    # replace
    host_term.parse_input(host_ip)
    verb.set_named_term_type(name="host", new_term=host_term)

    kwargs = {"repo": repo, "branch": branch}

    # provision
    return create_launch_custom_cmd(verb=verb, auth=auth, kwargs=kwargs)


def create_launch_custom_cmd(
    verb: GrammarVerb, auth: AuthCredentials, kwargs: TypeDict[str, Any]
) -> str:
    host_term = verb.get_named_term_type(name="host")
    node_name = verb.get_named_term_type(name="node_name")
    node_type = verb.get_named_term_type(name="node_type")
    # source_term = verb.get_named_term_type(name="source")

    snake_name = node_name.input.lower().replace(" ", "_")

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

    if not os.path.exists(playbook_path):
        print(f"Can't find playbook site.yml at: {playbook_path}")
    cmd = f"ANSIBLE_CONFIG={ansible_cfg_path} ansible-playbook -i {host_term.host}, {playbook_path}"
    if host_term.host != "localhost":
        cmd += f" --private-key {auth.key_path} --user {auth.username}"
    ANSIBLE_ARGS = {
        "node_type": node_type.input,
        "node_name": snake_name,
        "github_repo": kwargs["repo"],
        "repo_branch": kwargs["branch"],
    }

    if host_term.host == "localhost":
        ANSIBLE_ARGS["local"] = "true"

    # if mode == "deploy":
    #     ANSIBLE_ARGS["deploy"] = "true"

    for k, v in ANSIBLE_ARGS.items():
        cmd += f" -e \"{k}='{v}'\""

    cmd = "cd " + GRID_SRC_PATH + ";" + cmd
    return cmd


@click.command(help="Build (or re-build) PyGrid docker image.")
def build():
    check_docker_version()

    print("\n")

    cmd = ""
    cmd += " docker compose"
    cmd += " build"

    cmd = "cd " + GRID_SRC_PATH + ";" + cmd
    print(cmd)
    subprocess.call(cmd, shell=True)


@click.command(help="Stop a running PyGrid domain/network node.")
@click.argument("name", type=str, nargs=-1)
@click.option(
    "--type",
    "node_type",
    default="domain",
    required=False,
    type=click.Choice(["domain", "network"]),
    help="The type of node you would like to terminate.",
)
@click.option(
    "--port",
    default=8081,
    required=False,
    type=int,
    help="The public port your node exposes. (Default: 8081)",
)
@click.option(
    "--tag",
    default="",
    required=False,
    type=str,
    help="Optional: the underlying docker tag used (Default: 'domain_'+md5(name)",
)
def land(node_type, name, port, tag):

    _name = ""
    for word in name:
        _name += word + " "
    name = _name[:-1]

    if name == "all":
        subprocess.call("docker rm `docker ps -aq` --force", shell=True)
        return

    if tag == "" and name == "":
        raise Exception(
            "You must provide either the --tag or --name of the node you want to land!"
        )

    elif tag == "" and name != "" and node_type != "":
        tag = hashlib.md5(name.encode("utf8")).hexdigest()
        tag = node_type + "_" + tag

    elif tag != "":
        """continue"""

    else:
        raise Exception(
            "You must provide either a type and name, or you must provide a tag."
        )

    version = check_docker_version()

    motorcycle()

    print("Launching a " + str(node_type) + " PyGrid node on port " + str(port) + "!\n")
    print("  - TYPE: " + str(node_type))
    print("  - NAME: " + str(name))
    print("  - TAG: " + str(tag))
    print("  - PORT: " + str(port))
    print("  - DOCKER: " + version)

    print("\n")

    cmd = "DOMAIN_PORT=" + str(port)
    # cmd += " TRAEFIK_TAG=" + tag
    cmd += ' DOMAIN_NAME="' + name + '"'
    cmd += " NODE_TYPE=" + node_type
    cmd += " docker compose"
    cmd += ' --file "docker-compose.override.yml"'
    cmd += ' --project-name "' + tag + '"'
    cmd += " down"

    cmd = "cd " + GRID_SRC_PATH + ";export $(cat .env | sed 's/#.*//g' | xargs);" + cmd
    print(cmd)
    subprocess.call(cmd, shell=True)

    # if not keep_db:
    #     print("Deleting database for node...")
    #     subprocess.call("docker volume rm " + tag + "_app-db-data", shell=True)
    #     print()


cli.add_command(launch)
cli.add_command(build)
cli.add_command(land)
