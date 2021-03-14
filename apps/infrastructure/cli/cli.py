import glob
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin

import click
import requests

from apps.infrastructure.cli.providers import aws, azure, gcp
from apps.infrastructure.utils import COLORS, Config, colored

config_exist = glob.glob(str(Path.home() / ".pygrid/cli/*.json")) or None
prev_config = (
    max(config_exist, key=os.path.getmtime) if config_exist is not None else None
)
pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option("--api", required=True, type=str)
@click.option(
    "--output-file", default=f"config_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
)
@pass_config
def cli(config: SimpleNamespace, output_file: str, api: str):
    """OpenMined CLI for Infrastructure Management.

    Example:

    >>> pygrid --api <api-endpoint> deploy --provider aws --app domain

    >>> pygrid --api <api-endpoint> deploy --provider azure --app network
    """
    try:
        config.api_url = api
        response = requests.get(config.api_url)
        if response.status_code == 200:
            click.echo(colored(response.json()["message"]))
            click.echo(colored("Welcome to OpenMined PyGrid CLI", color=COLORS.blue))
    except:
        click.echo(colored("Please enter a valid API URL", color=COLORS.red))
        quit()

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
    default="Domain",
    type=click.Choice(["Domain", "Network", "Worker"], case_sensitive=False),
    help="The PyGrid App to be deployed",
)
@pass_config
def deploy(config: SimpleNamespace, prev_config: str, provider: str, app: str):

    config.provider = provider.lower()

    # Store credentials in a separate object, thus not logging it in output
    # when asking the user to confirm the current configuration
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
    config.serverless = False
    if not config.app.name in ["domain", "worker"]:
        config.serverless = click.confirm(f"Do you want to deploy serverless?")

    ## Websockets
    if not config.serverless:
        config.websockets = click.confirm(f"Will you need to support Websockets?")

    if not config.serverless:
        get_app_arguments(config)

    ## Prompting user to provide configuration for the selected cloud
    if config.provider == "aws":
        config.vpc = aws.get_vpc_config()
        if not config.serverless:
            config.vpc.instance_type = aws.get_instance_type(config.vpc.region)
    elif config.provider == "gcp":
        pass
    elif config.provider == "azure":
        pass

    ## Database
    if config.app.name != "worker":
        credentials.db = aws.get_db_config()

    if click.confirm(
        f"""Your current configration are:
        \n\n{colored((json.dumps(vars(config),
                        indent=2, default=lambda o: o.__dict__)))}
        \n\nContinue?"""
    ):

        config.credentials = credentials

        url = urljoin(config.api_url, "/deploy")

        data = json.dumps(vars(config), indent=2, default=lambda o: o.__dict__)
        r = requests.post(url, json=data)

        if r.status_code == 200:
            click.echo(colored(json.dumps(json.loads(r.text), indent=2)))
        else:
            click.echo(
                colored(json.dumps(json.loads(r.text), indent=2)), color=COLORS.red
            )


def get_app_arguments(config):
    config.app.count = click.prompt(
        f"How many apps do you want to deploy", type=int, default=1
    )
    apps = []
    for count in range(1, config.app.count + 1):
        if config.app.name == "domain":
            id = click.prompt(
                f"#{count}: PyGrid Domain ID",
                type=str,
                default=os.environ.get("DOMAIN_ID", None),
            )
            port = click.prompt(
                f"#{count}: Port number of the socket.io server",
                type=str,
                default=os.environ.get("GRID_DOMAIN_PORT", 5000),
            )
            host = click.prompt(
                f"#{count}: Grid DOMAIN host",
                type=str,
                default=os.environ.get("GRID_DOMAIN_HOST", "0.0.0.0"),
            )
            network = click.prompt(
                f"#{count}: Grid Network address (e.g. --network=0.0.0.0:7000)",
                type=str,
                default=os.environ.get("NETWORK", None),
            )
            app = Config(id=id, port=port, host=host, network=network)
        elif config.app.name == "network":
            port = click.prompt(
                f"#{count}: Port number of the socket.io server",
                type=str,
                default=os.environ.get("GRID_NETWORK_PORT", f"{7000 + count}"),
            )
            host = click.prompt(
                f"#{count}: Grid Network host",
                type=str,
                default=os.environ.get("GRID_NETWORK_HOST", "0.0.0.0"),
            )
            app = Config(port=port, host=host)
        else:
            port = click.prompt(
                f"#{count}: Port number of the socket.io server",
                type=str,
                default=os.environ.get("GRID_WORKER_PORT", 5000),
            )
            host = click.prompt(
                f"#{count}: Grid DOMAIN host",
                type=str,
                default=os.environ.get("GRID_WORKER_HOST", "0.0.0.0"),
            )
            app = Config(port=port, host=host)

        apps.append(app)
    config.apps = apps


@cli.resultcallback()
@pass_config
def logging(config, results, **kwargs):
    click.echo(f"Writing configs to {config.output_file}")
    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(vars(config), f, indent=2, default=lambda o: o.__dict__)
