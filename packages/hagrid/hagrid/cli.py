import click
from .lib import some_function
from .lib import motorcycle

@click.group()
def cli():
    pass

@click.command(help="Have your new program say Hi to you!")
@click.argument('type', type=click.Choice(['domain', 'network']))
@click.option('--name', default="", required=False, type=str)
@click.option('--port', default=8081, required=False, type=int)
@click.option('--tag', default="", required=False, type=str)
def launch(type, name, port, tag):
    import os

    if  not (type == "domain" or type=="network"):
        raise Exception("Must specify either a domain or network.")

    docker, compose, word, version = os.popen('docker compose version', 'r').read().split()

    motorcycle()
    print("Launching a " + str(type) + " PyGrid node on port " + str(port) + "!\n")
    print("  - TYPE: " + str(type))
    print("  - NAME: " + str(name))
    print("  - TAG: " + str(tag))
    print("  - PORT: " + str(port))
    print("  - DOCKER: " + version)

    print("\n")

    # some_function()


cli.add_command(launch)