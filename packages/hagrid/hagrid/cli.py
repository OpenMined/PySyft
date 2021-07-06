import click
from .lib import motorcycle
import os
import subprocess

install_path = os.path.dirname(os.path.realpath(__file__))

@click.group()
def cli():
    pass

@click.command(help="Have your new program say Hi to you!")
@click.argument('type', type=click.Choice(['domain', 'network']))
@click.option('--name', default="", required=False, type=str)
@click.option('--port', default=8081, required=False, type=int)
@click.option('--tag', default="", required=False, type=str)
def launch(type, name, port, tag):

    if tag != "":
        if ' ' in tag:
            raise Exception("Can't have spaces in --tag. Try something without spaces.")

    docker, compose, word, version = os.popen('docker compose version', 'r').read().split()



    motorcycle()

    print("Launching a " + str(type) + " PyGrid node on port " + str(port) + "!\n")
    print("  - TYPE: " + str(type))
    print("  - NAME: " + str(name))
    print("  - TAG: " + str(tag))
    print("  - PORT: " + str(port))
    print("  - DOCKER: " + version)

    print("\n")

    cmd = "DOMAIN_PORT="+str(port)
    cmd += " TRAEFIK_TAG="+tag
    cmd += " DOMAIN_NAME=\""+name+"\""
    cmd += " NODE_TYPE="+type
    cmd += " docker compose -p " + tag
    cmd += " up"

    install_path =os.path.abspath(os.path.join(os.path.realpath(__file__), '../../../grid/'))

    cmd = "cd " + install_path + ";" + cmd
    print(cmd)
    subprocess.call(cmd, shell=True)

cli.add_command(launch)