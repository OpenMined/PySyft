# stdlib
import hashlib
import os
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


@click.command(help="Start a new PyGrid domain/network node!")
@click.argument("name", type=str, nargs=-1)
@click.option(
    "--type",
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
    type=bool,
    help="""If restarting a node that already existed, don't/do reset the database (Default: deletes the db)""",
)
def launch(name, type, port, tag, keep_db, host="localhost"):

    _name = ""
    for word in name:
        _name += word + " "
    name = _name[:-1]

    if name == "":
        name = names.get_full_name() + "'s " + type.capitalize()

    if tag != "":
        if " " in tag:
            raise Exception("Can't have spaces in --tag. Try something without spaces.")
    else:
        tag = hashlib.md5(name.encode("utf8")).hexdigest()

    tag = type + "_" + tag

    # check port to make sure it's not in use - if it's in use then increment until it's not.
    port_available = False
    while not port_available:
        try:
            requests.get("http://" + host + ":" + str(port))
            print(
                str(port) + " doesn't seem to be available... trying " + str(port + 1)
            )
            port = port + 1
        except requests.ConnectionError as e:
            port_available = True

    if not keep_db:
        print("Deleting database for node...")
        subprocess.call("docker volume rm " + tag + "_app-db-data", shell=True)
        print()

    version = check_docker()

    motorcycle()

    print("Launching a " + str(type) + " PyGrid node on port " + str(port) + "!\n")
    print("  - TYPE: " + str(type))
    print("  - NAME: " + str(name))
    print("  - TAG: " + str(tag))
    print("  - PORT: " + str(port))
    print("  - DOCKER: " + version)

    print("\n")

    cmd = "DOMAIN_PORT=" + str(port)
    cmd += " TRAEFIK_TAG=" + tag
    cmd += ' DOMAIN_NAME="' + name + '"'
    cmd += " NODE_TYPE=" + type
    cmd += " docker compose -p " + tag
    cmd += " up"

    cmd = "cd " + install_path + ";" + cmd
    print(cmd)
    subprocess.call(cmd, shell=True)

@click.command(help="Build (or re-build) PyGrid docker image.")
def build():

    # # # _name = ""
    # # # for word in name:
    # # #     _name += word + " "
    # # # name = _name[:-1]
    # # #
    # # # if name == "":
    # # #     name = names.get_full_name() + "'s " + type.capitalize()
    # #
    # # if tag != "":
    # #     if " " in tag:
    # #         raise Exception("Can't have spaces in --tag. Try something without spaces.")
    # # else:
    # #     tag = hashlib.md5(name.encode("utf8")).hexdigest()
    # #
    # # tag = type + "_" + tag
    # #
    # # # check port to make sure it's not in use - if it's in use then increment until it's not.
    # port_available = False
    # while not port_available:
    #     try:
    #         requests.get("http://" + host + ":" + str(port))
    #         print(
    #             str(port) + " doesn't seem to be available... trying " + str(port + 1)
    #         )
    #         port = port + 1
    #     except requests.ConnectionError as e:
    #         port_available = True
    #
    # if not keep_db:
    #     print("Deleting database for node...")
    #     subprocess.call("docker volume rm " + tag + "_app-db-data", shell=True)
    #     print()

    version = check_docker()

    # motorcycle()

    # print("Launching a " + str(type) + " PyGrid node on port " + str(port) + "!\n")
    # print("  - TYPE: " + str(type))
    # print("  - NAME: " + str(name))
    # print("  - TAG: " + str(tag))
    # print("  - PORT: " + str(port))
    # print("  - DOCKER: " + version)

    print("\n")

    cmd = ""
    # cmd += "DOMAIN_PORT=" + str(port)
    # cmd += " TRAEFIK_TAG=" + tag
    # cmd += ' DOMAIN_NAME="' + name + '"'
    # cmd += " NODE_TYPE=" + type
    cmd += " docker compose"
    cmd += " build"

    cmd = "cd " + install_path + ";" + cmd
    print(cmd)
    subprocess.call(cmd, shell=True)


@click.command(help="Stop a running PyGrid domain/network node.")
@click.argument("name", type=str, nargs=-1)
@click.option(
    "--type",
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
def land(type, name, port, tag, keep_db):

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
