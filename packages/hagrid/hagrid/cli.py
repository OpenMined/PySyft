# stdlib
import hashlib
import io
import os
from pathlib import Path
import subprocess

# third party
import click
import rich

# relative
from .art import hagrid
from .deps import check_deps
from .lib import GRID_SRC_PATH
from .lib import check_docker
from .lib import find_available_port
from .lib import is_editable_mode
from .lib import pre_process_keep_db
from .lib import pre_process_name
from .lib import pre_process_tag
from .lib import should_provision_remote


class RichGroup(click.Group):
    def format_usage(self, ctx, formatter):
        DEPENDENCIES = check_deps()
        sio = io.StringIO()
        console = rich.get_console()
        mode = ""
        if is_editable_mode():
            mode = "[bold red]EDITABLE DEV MODE[/bold red] :police_car_light:"
        console.print(
            f"[bold red]HA[/bold red][bold magenta]Grid[/bold magenta]!", ":mage:", mode
        )
        table = rich.table.Table()

        table.add_column("Dependency", style="magenta")
        table.add_column("Found", justify="right")

        for dep in sorted(DEPENDENCIES.keys()):
            path = DEPENDENCIES[dep]
            installed_str = ":white_check_mark:" if path is not None else ":cross_mark:"
            dep_emoji = ":gear:"
            if dep == "docker":
                dep_emoji = ":whale:"
            if dep == "git":
                dep_emoji = ":file_folder:"
            if dep == "virtualbox":
                dep_emoji = ":ballot_box_with_ballot: "
            if dep == "vagrant":
                dep_emoji = ":person_mountain_biking:"
            if dep == "ansible-playbook":
                dep_emoji = ":blue_book:"
            table.add_row(f"{dep_emoji} {dep}", installed_str)
            # console.print(dep_emoji, dep, installed_str)
        console.print(table)
        console.print("Usage: hagrid [OPTIONS] COMMAND [ARGS]...")
        formatter.write(sio.getvalue())


@click.group(cls=RichGroup)
def cli():
    pass


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
    # run pre-processing of arguments
    name = pre_process_name(name=name, node_type=node_type)
    tag = pre_process_tag(tag=tag, name=name, node_type=node_type)

    # are we deploying locally or remotely?
    is_remote = should_provision_remote(username, password, key_path)

    if not is_remote:

        version = check_docker()

        # check port to make sure it's not in use - if it's in use then increment until it's not.
        port = find_available_port(host=host, port=port)

        if not pre_process_keep_db(keep_db, tag):
            print("Deleting database for node...")
            subprocess.call("docker volume rm " + tag + "_app-db-data", shell=True)
            print()

    hagrid()

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
        cmd = "cd " + GRID_SRC_PATH + ";" + cmd
    print("Running: \n", cmd)
    subprocess.call(cmd, shell=True)


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
