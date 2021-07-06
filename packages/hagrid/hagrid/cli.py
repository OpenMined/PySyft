# stdlib
import hashlib
import os
import subprocess

# third party
import click
import names
import requests

# relative
from .lib import check_docker, motorcycle

install_path = os.path.dirname(os.path.realpath(__file__))


@click.group()
def cli():
    pass


@click.command(help="Start a new PyGrid domain/network node!")
@click.argument("type", type=click.Choice(["domain", "network"]))
@click.option(
    "--name",
    default="",
    required=False,
    type=str,
    help="The name of your new domain/network node. (Default: <randomly generated>)",
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
    help="""Optional: the underlying docker tag used (Default: 'domain_'+md5(name)""",
)
def launch(type, name, port, tag, host="localhost"):

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

    install_path = os.path.abspath(
        os.path.join(os.path.realpath(__file__), "../../../grid/")
    )

    cmd = "cd " + install_path + ";" + cmd
    print(cmd)
    subprocess.call(cmd, shell=True)


@click.command(help="Stop a running PyGrid domain/network node.")
@click.argument("type", type=click.Choice(["domain", "network"]))
@click.option("--name", default="", required=False, type=str)
@click.option("--port", default=8081, required=False, type=int)
@click.option("--tag", default="", required=False, type=str)
def land(type, name, port, tag):

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

    motorcycle()

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


cli.add_command(launch)
cli.add_command(land)
