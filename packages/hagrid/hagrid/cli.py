# stdlib
import hashlib
import os
import subprocess

# third party
import click
import names

# relative
from .lib import motorcycle

install_path = os.path.dirname(os.path.realpath(__file__))


@click.group()
def cli():
    pass


@click.command(help="Have your new program say Hi to you!")
@click.argument("type", type=click.Choice(["domain", "network"]))
@click.option("--name", default="", required=False, type=str)
@click.option("--port", default=8081, required=False, type=int)
@click.option("--tag", default="", required=False, type=str)
def launch(type, name, port, tag):

    docker_compose = "docker compose"

    if name == "":
        name = names.get_full_name() + "'s " + type.capitalize()

    if tag != "":
        if " " in tag:
            raise Exception("Can't have spaces in --tag. Try something without spaces.")
    else:
        tag = hashlib.md5(name.encode("utf8")).hexdigest()

    tag = type + "_" + tag

    result = (
        os.popen(docker_compose + " version", "r").read()
    )

    if "version" in result:
        version = result.split()[-1]
    else:
        print("This may be a linux machine, either that or docker compose isn't s")
        print("Result:" + result)
        out = subprocess.run(['docker', 'compose'], capture_output=True, text=True)
        if "'compose' is not a docker command" in out.stderr:
            raise Exception("""You are running an old verion of docker, possibly on Linux. You need to install v2 beta.
            Instructions for v2 beta can be found here: 
            
            https://www.rockyourcode.com/how-to-install-docker-compose-v2-on-linux-2021/
            
            At the time of writing this, if you are on linux you need to run the following:
            
            mkdir -p ~/.docker/cli-plugins
            curl -sSL https://github.com/docker/compose-cli/releases/download/v2.0.0-beta.5/docker-compose-linux-amd64 -o ~/.docker/cli-plugins/docker-compose
            chmod +x ~/.docker/cli-plugins/docker-compose
            
            ALERT: you may need to run the following command to make sure you can run without sudo.
            
            echo $USER              //(should return your username)
            sudo usermod -aG docker $USER
            
            ... now LOG ALL THE WAY OUT!!!
            
            ...and then you should be good to go. You can check your installation by running:
            
            docker compose version
            """)


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
    cmd += " "+docker_compose+" -p " + tag
    cmd += " up"

    install_path = os.path.abspath(
        os.path.join(os.path.realpath(__file__), "../../../grid/")
    )

    cmd = "cd " + install_path + ";" + cmd
    print(cmd)
    subprocess.call(cmd, shell=True)


cli.add_command(launch)
