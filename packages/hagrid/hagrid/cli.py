# stdlib
import hashlib
import subprocess
from typing import Tuple as TypeTuple

# third party
import click

# relative
from .art import hagrid
from .grammar import BadGrammar
from .grammar import parse_grammar
from .lib import GRID_SRC_PATH
from .lib import check_docker
from .lib import find_available_port
from .lib import pre_process_keep_db
from .lib import pre_process_name
from .lib import pre_process_tag
from .lib import should_provision_remote
from .names import random_name
from .style import RichGroup


@click.group(cls=RichGroup)
def cli():
    pass


@click.command(help="Start a new PyGrid domain/network node!")
@click.argument("args", type=str, nargs=-1)
@click.option(
    "--type",
    "node_type",
    default="domain",
    required=False,
    type=click.Choice(["domain", "network"]),
    help="The type of node you would like to deploy.",
)
@click.option(
    "--port",
    default=8081,
    required=False,
    type=int,
    help="The public port your node should expose. (Default: 8081)",
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
    default=False,
    required=False,
    help="""If restarting a node that already existed, don't/do reset the database (Default: deletes the db)""",
)
@click.option(
    "--host",
    default="localhost",
    required=False,
    type=str,
    help="Optional: the host to provision, leave empty if localhost / docker",
)
@click.option(
    "--username",
    default=None,
    required=False,
    type=str,
    help="Optional: the username for provisioning the remote host",
)
@click.option(
    "--password",
    default=None,
    required=False,
    type=str,
    help="Optional: the password for provisioning the remote host",
)
@click.option(
    "--key_path",
    default=None,
    required=False,
    type=str,
    help="Optional: the path to the key file for provisioning the remote host",
)
@click.option(
    "--mode",
    default=None,
    required=False,
    type=str,
    help="Optional: mode either provision or deploy, where deploy is a quick code update",
)
@click.option(
    "--repo",
    default="OpenMined/PySyft",
    required=False,
    type=str,
    help="Optional: repo to fetch source from",
)
@click.option(
    "--branch",
    default="demo_strike_team_branch_4",
    required=False,
    type=str,
    help="Optional: branch to monitor for updates",
)
def launch(
    args: TypeTuple[str],
    node_type,
    port,
    tag,
    keep_db,
    host,
    username=None,
    password=None,
    key_path=None,
    mode: str = "provision",
    repo: str = "OpenMined/PySyft",
    branch: str = "demo_strike_team_branch_4",
):

    # Verb	Adjective	Object	prep	proper noun / Technology:(optional) Port	from	Github URL / Location / Branch
    # launch	slytherine	domain	to	docker:port	from	github.com/OpenMined/PySyft/tree/dev
    launch_grammar = [
        {
            "type": "verb",
            "command": "launch",
            "mappings": {
                6: [
                    "adjective",
                    "object",
                    "preposition",
                    "propernoun",
                    "preposition",
                    "propernoun",
                ]
            },
        },
        {"type": "adjective", "default": random_name, "example": "'My Domain'"},
        {"type": "object", "default": "domain", "options": ["domain", "network"]},
        {"type": "preposition", "default": "to", "options": ["to"]},
        {"type": "propernoun", "default": "docker", "example": "docker:8081+"},
        {"type": "preposition", "default": "from", "options": ["from"]},
        {
            "type": "propernoun",
            "default": "github.com/OpenMined/PySyft/tree/demo_strike_team_branch_4",
        },
    ]
    try:
        grammar = parse_grammar(args, launch_grammar)
    except BadGrammar as e:
        print(e)
        return

    print(grammar)
    # name = pre_process_name(name=name, node_type=node_type)
    print("launch")
    print(args, type(args))
    print(node_type, type(node_type))
    # print(node_type, type(node_type))
    # # run pre-processing of arguments

    # tag = pre_process_tag(tag=tag, name=name, node_type=node_type)

    # # are we deploying locally or remotely?
    # is_remote = should_provision_remote(username, password, key_path)

    # if not is_remote:

    #     version = check_docker()

    #     # check port to make sure it's not in use - if it's in use then increment until it's not.
    #     port = find_available_port(host=host, port=port)

    #     if not pre_process_keep_db(keep_db, tag):
    #         print("Deleting database for node...")
    #         subprocess.call("docker volume rm " + tag + "_app-db-data", shell=True)
    #         print()

    # hagrid()

    # if not is_remote:
    #     print(
    #         "Launching a "
    #         + str(node_type)
    #         + " PyGrid node on port "
    #         + str(port)
    #         + "!\n"
    #     )
    # else:
    #     print("Launching a " + str(node_type) + f" PyGrid node on http://{host}!\n")
    # print("  - TYPE: " + str(node_type))
    # print("  - NAME: " + str(name))
    # if not is_remote:
    #     print("  - TAG: " + str(tag))
    #     print("  - PORT: " + str(port))
    #     print("  - DOCKER: " + version)
    # else:
    #     print("  - HOST: " + host)
    #     if username is not None:
    #         print("  - USERNAME: " + username)

    #     if password is not None:
    #         print("  - PASSWORD: *************")

    #     if key_path is not None:
    #         print("  - KEY_PATH: " + key_path)

    # print("\n")

    # cmd = ""
    # if not is_remote:
    #     cmd += "DOMAIN_PORT=" + str(port)
    #     cmd += " TRAEFIK_TAG=" + tag

    # cmd += ' DOMAIN_NAME="' + name + '"'
    # cmd += " NODE_TYPE=" + node_type

    # if is_remote:
    #     # use ansible on remote host
    #     # if username is None:
    #     #     cmd += f' USERNAME="{username}"'
    #     # elif password is not None:
    #     #     cmd += f' PASSWORD="{password}"'
    #     # elif key_path is not None:
    #     #     cmd += f' KEY_PATH="{key_path}"'

    #     current_path = os.path.dirname(__file__)
    #     grid_path = Path(os.path.abspath(f"{current_path}/../../grid"))
    #     playbook_path = grid_path / "ansible/site.yml"
    #     ansible_cfg_path = grid_path / "ansible.cfg"

    #     if not os.path.exists(playbook_path):
    #         print(f"Can't find playbook site.yml at: {playbook_path}")
    #     cmd = f"ANSIBLE_CONFIG={ansible_cfg_path} ansible-playbook -i {host}, {playbook_path}"
    #     if host != "localhost":
    #         cmd += f" --private-key {key_path} --user {username}"
    #     ANSIBLE_ARGS = {
    #         "node_type": node_type,
    #         "node_name": name,
    #         "github_repo": repo,
    #         "repo_branch": branch,
    #     }
    #     if mode == "deploy":
    #         ANSIBLE_ARGS["deploy_only"] = "true"

    #     if host == "localhost":
    #         ANSIBLE_ARGS["local"] = "true"

    #     args = []
    #     for k, v in ANSIBLE_ARGS.items():
    #         args.append(f"{k}={v}")
    #     args_str = " ".join(args)
    #     cmd += f' -e "{args_str}"'
    # else:
    #     # use docker on localhost
    #     cmd += " docker compose -p " + tag
    #     cmd += " up"
    #     cmd = "cd " + GRID_SRC_PATH + ";" + cmd
    # print("Running: \n", cmd)
    # subprocess.call(cmd, shell=True)


@click.command(help="Build (or re-build) PyGrid docker image.")
def build():

    version = check_docker()

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

    """DOMAIN_PORT=$port DOMAIN_NAME=$name NODE_TYPE=$type docker compose --file "docker-compose.override.yml" --project-name "$name" down"""

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
