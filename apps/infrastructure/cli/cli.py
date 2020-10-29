import json
import os
import time
from pathlib import Path

import click
import requests

from .provider_utils import aws, azure, gcp
from .utils import COLORS, Config, colored

pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option(
    "--output-file", default=f"config_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
)
@pass_config
def cli(config, output_file):
    """OpenMined CLI for Infrastructure Management.

    Example:

    >>> pygrid deploy --provider aws --app node

    >>> pygrid deploy --provider azure --app network
    """
    click.echo(colored("Welcome to OpenMined PyGrid CLI!"))

    ## ROOT Directory
    config.pygrid_root_path = str(Path.home() / ".pygrid/cli/")
    os.makedirs(config.pygrid_root_path, exist_ok=True)
    config.output_file = f"{config.pygrid_root_path}/{output_file}"


@cli.command()
@click.option(
    "--provider",
    prompt="Cloud Provider: ",
    default="AWS",
    type=click.Choice(["AWS", "GCP", "AZURE"], case_sensitive=False),
    help="The Cloud Provider for the deployment",
)
@click.option(
    "--app",
    prompt="PyGrid App: ",
    default="Node",
    type=click.Choice(["Node", "Network", "Worker"], case_sensitive=False),
    help="The PyGrid App to be deployed",
)
@pass_config
def deploy(config, provider, app):
    config.provider = provider.lower()

    credentials = Config()

    ## credentials file
    with open(
        click.prompt(
            f"Please enter path to your  {colored(f'{config.provider} credentials')} json file",
            type=str,
            default=f"{Path.home()}/.{config.provider}/credentials.json",
        ),
        "r",
    ) as f:
        credentials.cloud = json.load(f)

    ## Get app config and arguments
    config.app = Config(name=app.lower())

    ## Deployment type
    config.deployment_type = (
        "serverless"
        if click.confirm(f"Do you want to deploy serverless?")
        else "serverfull"
    )

    ## Websockets
    config.websockets = (
        True if click.confirm(f"Will you need to support Websockets?") else False
    )

    get_app_arguments(config)

    ## Prompting user to provide configuration for the selected cloud
    if config.provider == "aws":
        config.vpc = aws.get_vpc_config()
    elif config.provider == "gcp":
        pass
    elif config.provider == "azure":
        pass

    ## Database
    credentials.db = aws.get_db_config()

    if click.confirm(
        f"""Your current configration are: \n\n{colored((json.dumps(vars(config), indent=2, default=lambda o: o.__dict__)))} \n\nContinue?"""
    ):

        config.credentials = credentials

        url = "http://localhost:5000/"
        data = json.dumps(vars(config), indent=2, default=lambda o: o.__dict__)
        r = requests.post(url, json=data)

        if r.status_code == 200:
            print(f"Your PyGrid {config.app.name} was deployed successfully")
        else:
            print(
                f"There was an issue with deploying your Pygrid {config.app.name}. Please try again."
            )


def get_app_arguments(config):
    if config.app.name == "node":
        config.app.id = click.prompt(
            f"PyGrid Node ID", type=str, default=os.environ.get("NODE_ID", None)
        )
        config.app.port = click.prompt(
            f"Port number of the socket.io server",
            type=str,
            default=os.environ.get("GRID_NODE_PORT", 5000),
        )
        config.app.host = click.prompt(
            f"Grid node host",
            type=str,
            default=os.environ.get("GRID_NODE_HOST", "0.0.0.0"),
        )
        config.app.network = click.prompt(
            f"Grid Network address (e.g. --network=0.0.0.0:7000)",
            type=str,
            default=os.environ.get("NETWORK", None),
        )
        # TODO: Validate if this is related to data-centric or model-centric and is it requried?
        # config.app.num_replicas = click.prompt(
        #     f"Number of replicas to provide fault tolerance to model hosting",
        #     type=int,
        #     default=os.environ.get("NUM_REPLICAS", None),
        # )
    elif config.app.name == "network" and config.deployment_type == "serverfull":
        config.app.port = click.prompt(
            f"Port number of the socket.io server",
            type=str,
            default=os.environ.get("GRID_NETWORK_PORT", "7000"),
        )
        config.app.host = click.prompt(
            f"Grid Network host",
            type=str,
            default=os.environ.get("GRID_NETWORK_HOST", "0.0.0.0"),
        )
    else:
        # TODO: Workers arguments
        pass


@cli.resultcallback()
@pass_config
def logging(config, results, **kwargs):
    click.echo(f"Writing configs to {config.output_file}")
    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(vars(config), f, indent=2, default=lambda o: o.__dict__)
