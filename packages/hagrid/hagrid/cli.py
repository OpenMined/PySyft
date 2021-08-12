# stdlib
import json
import os
import re
import stat
import subprocess
from typing import Any
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple
from typing import cast

# third party
import click

# relative
from .art import hagrid
from .auth import AuthCredentials
from .cache import arg_cache
from .deps import DEPENDENCIES
from .deps import MissingDependency
from .deps import allowed_hosts
from .grammar import BadGrammar
from .grammar import GrammarVerb
from .grammar import parse_grammar
from .land import get_land_verb
from .launch import get_launch_verb
from .lib import GRID_SRC_PATH
from .lib import check_docker_version
from .lib import name_tag
from .lib import use_branch
from .style import RichGroup


@click.group(cls=RichGroup)
def cli() -> None:
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
@click.option(
    "--tail",
    default=None,
    required=False,
    type=str,
    help="Optional: don't tail logs on launch",
)
@click.option(
    "--cmd",
    default=None,
    required=False,
    type=str,
    help="Optional: print the cmd without running it",
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
        cmd = create_launch_cmd(verb=verb, kwargs=kwargs)
    except Exception as e:
        print(f"{e}")
        return
    print("Running: \n", cmd)
    if "cmd" not in kwargs or str_to_bool(cast(str, kwargs["cmd"])) is False:
        subprocess.call(cmd, shell=True)


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
                raise QuestionInputPathError(f"{error}")

        if self.kind == "yesno":
            if value.lower().startswith("y"):
                return "y"
            elif value.lower().startswith("n"):
                return "n"

        return value


def ask(question: Question, kwargs: TypeDict[str, str]) -> str:
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


def private_to_public_key(private_key_path: str, username: str) -> str:
    # check key permission
    fix_key_permission(private_key_path=private_key_path)
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


def str_to_bool(bool_str: Optional[str]) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


ART = str_to_bool(os.environ.get("HAGRID_ART", "True"))


def generate_key_at_path(key_path: str) -> str:
    key_path = os.path.expanduser(key_path)
    if os.path.exists(key_path):
        raise Exception(f"Can't generate key since path already exists. {key_path}")
    else:
        cmd = f"ssh-keygen -N '' -f {key_path}"
        try:
            subprocess.check_call(cmd, shell=True)
            if not os.path.exists(key_path):
                raise Exception(f"Failed to generate ssh-key at: {key_path}")
        except Exception as e:
            raise e

    return key_path


def create_launch_cmd(
    verb: GrammarVerb,
    kwargs: TypeDict[str, Any],
    ignore_docker_version_check: Optional[bool] = False,
) -> str:
    host_term = verb.get_named_term_hostgrammar(name="host")
    host = host_term.host
    auth: Optional[AuthCredentials] = None

    tail = True
    if "tail" in kwargs:
        tail = str_to_bool(kwargs["tail"])

    if host in ["docker"]:

        if not ignore_docker_version_check:
            version = check_docker_version()
        else:
            version = "n/a"

        if version:
            return create_launch_docker_cmd(
                verb=verb, docker_version=version, tail=tail
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

            key_path_question = Question(
                var_name="azure_key_path",
                question=f"Private key to access {username}@{host}?",
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
        return ""
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
            required_questions = []
            if host != "localhost":
                required_questions.append(username_question)
                required_questions.append(key_path_question)
            required_questions.append(repo_question)
            required_questions.append(branch_question)

            parsed_kwargs = requires_kwargs(
                required=required_questions,
                kwargs=kwargs,
            )

            if "branch" in parsed_kwargs:
                use_branch(branch=parsed_kwargs["branch"])

            auth = None
            if host != "localhost":
                auth = AuthCredentials(
                    username=parsed_kwargs["username"],
                    key_path=parsed_kwargs["key_path"],
                )
                if not auth.valid:
                    raise Exception(f"Login Credentials are not valid. {auth}")
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
    verb: GrammarVerb, docker_version: str, tail: bool = False
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

    cmd = ""
    cmd += "DOMAIN_PORT=" + str(host_term.free_port)
    cmd += " TRAEFIK_TAG=" + str(tag)
    cmd += ' DOMAIN_NAME="' + snake_name + '"'
    cmd += " NODE_TYPE=" + str(node_type.input)
    cmd += " docker compose -p " + snake_name
    cmd += " up"

    # if not tail:
    #     cmd += " -d"

    cmd += " --build"  # force rebuild
    cmd = "cd " + GRID_SRC_PATH + ";" + cmd
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
) -> Optional[str]:
    public_key_path = private_to_public_key(
        private_key_path=key_path, username=username
    )

    cmd = f"az vm create -n {node_name} -g {resource_group} --size {size} "
    cmd += "--image Canonical:0001-com-ubuntu-server-focal:20_04-lts:latest "
    cmd += "--public-ip-sku Standard --authentication-type ssh "
    cmd += f"--ssh-key-values {public_key_path} --admin-username {username}"
    host_ip: Optional[str] = None
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
    auth: AuthCredentials,
) -> str:
    # resource group
    get_or_make_resource_group(resource_group=resource_group, location=location)

    # vm
    node_name = verb.get_named_term_type(name="node_name")
    snake_name = str(node_name.snake_input)
    host_ip = make_vm_azure(snake_name, resource_group, username, key_path, size)

    # open port 80
    open_port_vm_azure(resource_group=resource_group, node_name=snake_name, port=80)

    # get old host
    host_term = verb.get_named_term_hostgrammar(name="host")

    # replace
    host_term.parse_input(host_ip)
    verb.set_named_term_type(name="host", new_term=host_term)

    kwargs = {"repo": repo, "branch": branch}

    # provision
    return create_launch_custom_cmd(verb=verb, auth=auth, kwargs=kwargs)


def create_launch_custom_cmd(
    verb: GrammarVerb, auth: Optional[AuthCredentials], kwargs: TypeDict[str, Any]
) -> str:
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


def create_land_cmd(verb: GrammarVerb, kwargs: TypeDict[str, Any]) -> str:
    host_term = verb.get_named_term_hostgrammar(name="host")
    host = host_term.host

    if verb.get_named_term_grammar("node_name").input == "all":
        # subprocess.call("docker rm `docker ps -aq` --force", shell=True)
        return "docker rm `docker ps -aq` --force"

    if host in ["docker"]:
        version = check_docker_version()
        if version:
            return create_land_docker_cmd(verb=verb)

    host_options = ", ".join(allowed_hosts)
    raise MissingDependency(
        f"Launch requires a correct host option, try: {host_options}"
    )


def create_land_docker_cmd(verb: GrammarVerb) -> str:
    node_name = verb.get_named_term_type(name="node_name")
    snake_name = str(node_name.snake_input)

    cmd = ""
    cmd += "docker compose"
    cmd += ' --file "docker-compose.override.yml"'
    cmd += ' --project-name "' + snake_name + '"'
    cmd += " down"

    cmd = "cd " + GRID_SRC_PATH + ";export $(cat .env | sed 's/#.*//g' | xargs);" + cmd
    return cmd


@click.command(help="Stop a running PyGrid domain/network node.")
@click.argument("args", type=str, nargs=-1)
def land(args: TypeTuple[str], **kwargs: TypeDict[str, Any]) -> None:
    verb = get_land_verb()
    try:
        grammar = parse_grammar(args=args, verb=verb)
        verb.load_grammar(grammar=grammar)
    except BadGrammar as e:
        print(e)
        return

    # if len(args) == 0:
    #     print("use interactive menu to select node?")

    try:
        cmd = create_land_cmd(verb=verb, kwargs=kwargs)
    except Exception as e:
        print(f"{e}")
        return
    print("Running: \n", cmd)
    subprocess.call(cmd, shell=True)


cli.add_command(launch)
cli.add_command(land)
