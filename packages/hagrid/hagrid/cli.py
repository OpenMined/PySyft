# stdlib
import hashlib
import os
from pathlib import Path
import subprocess

# third party
import click
import names
import requests

# relative
from .lib import check_docker
from .lib import motorcycle

install_path = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "../../../grid/")
)


@click.group()
def cli():
    pass


def provision_remote(username, password, key_path) -> bool:
    is_remote = username is not None or password is not None or key_path is not None
    if username and password or username and key_path:
        return is_remote
    if is_remote:
        raise Exception("--username requires either --password or --key_path")
    return is_remote


@click.command(help="Start a new PyGrid domain/network node!")
@click.argument("name", type=str, nargs=-1)
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
def launch(
    name,
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
    is_remote = provision_remote(username, password, key_path)

    _name = ""
    for word in name:
        _name += word + " "
    name = _name[:-1]

    if name == "":
        name = "The " + names.get_full_name() + " " + node_type.capitalize()

    if not is_remote:
        if tag != "":
            if " " in tag:
                raise Exception(
                    "Can't have spaces in --tag. Try something without spaces."
                )
        else:
            tag = hashlib.md5(name.encode("utf8")).hexdigest()

        tag = node_type + "_" + tag

        # check port to make sure it's not in use - if it's in use then increment until it's not.
        port_available = False
        while not port_available:
            try:
                requests.get("http://" + host + ":" + str(port))
                print(
                    str(port)
                    + " doesn't seem to be available... trying "
                    + str(port + 1)
                )
                port = port + 1
            except requests.ConnectionError as e:
                port_available = True

        if isinstance(keep_db, str):
            keep_db = True if keep_db.lower() == "true" else False
        if not keep_db:
            print("Deleting database for node...")
            subprocess.call("docker volume rm " + tag + "_app-db-data", shell=True)
            print()

        version = check_docker()

    motorcycle()

    if not is_remote:
        print(
            "Launching a "
            + str(node_type)
            + " PyGrid node on port "
            + str(port)
            + "!\n"
        )
    else:
        print("Launching a " + str(node_type) + f" PyGrid node on http://{host}!\n")
    print("  - TYPE: " + str(node_type))
    print("  - NAME: " + str(name))
    if not is_remote:
        print("  - TAG: " + str(tag))
        print("  - PORT: " + str(port))
        print("  - DOCKER: " + version)
    else:
        print("  - HOST: " + host)
        if username is not None:
            print("  - USERNAME: " + username)

        if password is not None:
            print("  - PASSWORD: *************")

        if key_path is not None:
            print("  - KEY_PATH: " + key_path)

    print("\n")

    cmd = ""
    if not is_remote:
        cmd += "DOMAIN_PORT=" + str(port)
        cmd += " TRAEFIK_TAG=" + tag

    cmd += ' DOMAIN_NAME="' + name + '"'
    cmd += " NODE_TYPE=" + node_type

    if is_remote:
        # use ansible on remote host
        # if username is None:
        #     cmd += f' USERNAME="{username}"'
        # elif password is not None:
        #     cmd += f' PASSWORD="{password}"'
        # elif key_path is not None:
        #     cmd += f' KEY_PATH="{key_path}"'

        current_path = os.path.dirname(__file__)
        grid_path = Path(os.path.abspath(f"{current_path}/../../grid"))
        playbook_path = grid_path / "ansible/site.yml"
        ansible_cfg_path = grid_path / "ansible.cfg"

        if not os.path.exists(playbook_path):
            print(f"Can't find playbook site.yml at: {playbook_path}")
        cmd = f"ANSIBLE_CONFIG={ansible_cfg_path} ansible-playbook -i {host}, {playbook_path}"
        if host != "localhost":
            cmd += f" --private-key {key_path} --user {username}"
        ANSIBLE_ARGS = {
            "node_type": node_type,
            "node_name": name,
            "github_repo": repo,
            "repo_branch": branch,
        }
        if mode == "deploy":
            ANSIBLE_ARGS["deploy_only"] = "true"

        if host == "localhost":
            ANSIBLE_ARGS["local"] = "true"

        args = []
        for k, v in ANSIBLE_ARGS.items():
            args.append(f"{k}={v}")
        args_str = " ".join(args)
        cmd += f' -e "{args_str}"'
    else:
        # use docker on localhost
        cmd += " docker compose -p " + tag
        cmd += " up"

        cmd = "cd " + install_path + ";" + cmd
    print("Running: \n", cmd)
    subprocess.call(cmd, shell=True)


@click.command(help="Build (or re-build) PyGrid docker image.")
def build():

    version = check_docker()

    print("\n")

    cmd = ""
    cmd += " docker compose"
    cmd += " build"

    cmd = "cd " + install_path + ";" + cmd
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

    if tag == "" and name == "":
        raise Exception(
            "You must provide either the --tag or --name of the node you want to land!"
        )

    elif tag == "" and name != "" and type != "":
        tag = hashlib.md5(name.encode("utf8")).hexdigest()
        tag = type + "_" + tag

    elif tag != "":
        """continue"""

    else:
        raise Exception(
            "You must provide either a type and name, or you must provide a tag."
        )

    version = check_docker()

    # motorcycle()

    print("Launching a " + str(type) + " PyGrid node on port " + str(port) + "!\n")
    print("  - TYPE: " + str(type))
    print("  - NAME: " + str(name))
    print("  - TAG: " + str(tag))
    print("  - PORT: " + str(port))
    print("  - DOCKER: " + version)

    print("\n")

    """DOMAIN_PORT=$port DOMAIN_NAME=$name NODE_TYPE=$type docker compose --file "docker-compose.override.yml" --project-name "$name" down"""

    cmd = "DOMAIN_PORT=" + str(port)
    # cmd += " TRAEFIK_TAG=" + tag
    cmd += ' DOMAIN_NAME="' + name + '"'
    cmd += " NODE_TYPE=" + type
    cmd += " docker compose"
    cmd += ' --file "docker-compose.override.yml"'
    cmd += ' --project-name "' + tag + '"'
    cmd += " down"

    install_path = os.path.abspath(
        os.path.join(os.path.realpath(__file__), "../../../grid/")
    )

    cmd = "cd " + install_path + ";export $(cat .env | sed 's/#.*//g' | xargs);" + cmd
    print(cmd)
    subprocess.call(cmd, shell=True)

    if not keep_db:
        print("Deleting database for node...")
        subprocess.call("docker volume rm " + tag + "_app-db-data", shell=True)
        print()


cli.add_command(launch)
cli.add_command(build)
cli.add_command(land)
