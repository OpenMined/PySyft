# stdlib
import hashlib
import subprocess
from typing import List as TypeList
from typing import Optional
from typing import Tuple as TypeTuple

# third party
import click

# relative
from .art import hagrid
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
# @click.option(
#     "--username",
#     default=None,
#     required=False,
#     type=str,
#     help="Optional: the username for provisioning the remote host",
# )
# @click.option(
#     "--password",
#     default=None,
#     required=False,
#     type=str,
#     help="Optional: the password for provisioning the remote host",
# )
# @click.option(
#     "--key_path",
#     default=None,
#     required=False,
#     type=str,
#     help="Optional: the path to the key file for provisioning the remote host",
# )
# @click.option(
#     "--mode",
#     default=None,
#     required=False,
#     type=str,
#     help="Optional: mode either provision or deploy, where deploy is a quick code update",
# )
# @click.option(
#     "--repo",
#     default="OpenMined/PySyft",
#     required=False,
#     type=str,
#     help="Optional: repo to fetch source from",
# )
# @click.option(
#     "--branch",
#     default="demo_strike_team_branch_4",
#     required=False,
#     type=str,
#     help="Optional: branch to monitor for updates",
# )
def launch(
    args: TypeTuple[str],
    # node_type,
    # port,
    # tag,
    # keep_db,
    # host,
    # username=None,
    # password=None,
    # key_path=None,
    # mode: str = "provision",
    # repo: str = "OpenMined/PySyft",
    # branch: str = "demo_strike_team_branch_4",
):
    verb = get_launch_verb()
    try:
        grammar = parse_grammar(args=args, verb=verb)
        verb.load_grammar(grammar=grammar)
    except BadGrammar as e:
        print(e)
        return

    print("launch", grammar)
    print(args, type(args))

    try:
        cmd = create_launch_cmd(verb=verb)
    except Exception as e:
        print(f"{e}")
        return
    print("Running: \n", cmd)
    subprocess.call(cmd, shell=True)


def create_launch_cmd(verb: GrammarVerb) -> str:
    host_term = verb.get_named_term_type(name="host")
    host = host_term.host
    if host in ["docker"]:
        version = check_docker_version()
        if version:
            return create_launch_docker_cmd(verb=verb, docker_version=version)
    elif host in ["vm"]:
        if DEPENDENCIES["vagrant"] and DEPENDENCIES["virtualbox"]:
            return create_launch_vagrant_cmd(verb=verb)
        else:
            errors = []
            if not DEPENDENCIES["vagrant"]:
                errors.append("vagrant")
            if not DEPENDENCIES["virtualbox"]:
                errors.append("virtualbox")
            raise MissingDependency(
                f"Launching a VM locally requires: {' '.join(errors)}"
            )
    elif host in ["aws", "azure", "gcp"]:

        print("launch @ cloud")
    else:
        print("launch @ custom host")


def create_launch_docker_cmd(
    verb: GrammarVerb, docker_version: str, tail: bool = False
) -> str:
    host_term = verb.get_named_term_type(name="host")
    node_name = verb.get_named_term_type(name="node_name")
    node_type = verb.get_named_term_type(name="node_type")
    tag = name_tag(name=node_name.input)

    hagrid()

    print(
        "Launching a "
        + str(node_type)
        + " PyGrid node on port "
        + str(host_term.free_port)
        + "!\n"
    )

    print("  - TYPE: " + str(node_type.input))
    print("  - NAME: " + str(node_name.input))
    print("  - TAG: " + str(tag))
    print("  - PORT: " + str(host_term.free_port))
    print("  - DOCKER: " + docker_version)
    print("\n")

    cmd = ""
    cmd += "DOMAIN_PORT=" + str(host_term.free_port)
    cmd += " TRAEFIK_TAG=" + str(tag)
    cmd += ' DOMAIN_NAME="' + node_name.input + '"'
    cmd += " NODE_TYPE=" + node_type.input
    cmd += " docker compose -p " + node_name.input.lower().replace(" ", "_")
    cmd += " up"
    if not tail:
        cmd += " -d"
    cmd = "cd " + GRID_SRC_PATH + ";" + cmd
    return cmd


def create_launch_vagrant_cmd(verb: GrammarVerb) -> str:
    host_term = verb.get_named_term_type(name="host")
    node_name = verb.get_named_term_type(name="node_name")
    node_type = verb.get_named_term_type(name="node_type")

    hagrid()

    print(
        "Launching a "
        + str(node_type)
        + " PyGrid node on port "
        + str(host_term.port)
        + "!\n"
    )

    print("  - TYPE: " + str(node_type.input))
    print("  - NAME: " + str(node_name.input))
    print("  - PORT: " + str(host_term.port))
    # print("  - VAGRANT: " + "1")
    # print("  - VIRTUALBOX: " + "1")
    print("\n")

    cmd = ""
    cmd += 'ANSIBLE_ARGS="'
    cmd += f"-e 'node_name={node_name.input}'"
    cmd += f"-e 'node_type={node_type.input}'"
    cmd += '" '
    cmd += "vagrant up --provision"
    cmd = "cd " + GRID_SRC_PATH + ";" + cmd
    return cmd


@click.command(help="Build (or re-build) PyGrid docker image.")
def build():
    check_docker()

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
@click.option(
    "--keep-db/--delete-db",
    default=True,
    required=False,
    type=bool,
    help="""If restarting a node that already existed, don't/do reset the database (Default: deletes the db)""",
)
def land(node_type, name, port, tag, keep_db):

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

    version = check_docker()

    # motorcycle()

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

    if not keep_db:
        print("Deleting database for node...")
        subprocess.call("docker volume rm " + tag + "_app-db-data", shell=True)
        print()


cli.add_command(launch)
cli.add_command(build)
cli.add_command(land)
