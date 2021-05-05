# stdlib
import glob
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from urllib.parse import urljoin

# third party
import click

# grid relative
from .providers import AWS_Serverfull
from .providers import AWS_Serverless
from .providers import AZURE
from .providers import GCP
from .providers.aws import utils as aws_utils
from .providers.azure import utils as azure_utils
from .providers.gcp import utils as gcp_utils
from .utils import COLORS
from .utils import Config
from .utils import colored

pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option(
    "--output-file", default=f"config_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
)
@pass_config
def cli(config: SimpleNamespace, output_file: str):
    """OpenMined CLI for Infrastructure Management.

    Example:

    >>> pygrid deploy --provider aws --app domain

    >>> pygrid deploy --provider azure --app network
    """
    click.echo(colored("Welcome to OpenMined PyGrid CLI", color=COLORS.blue))
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
    type=click.Choice(["Domain", "Network"], case_sensitive=False),
    help="The PyGrid App to be deployed",
)
@pass_config
def deploy(config: SimpleNamespace, provider: str, app: str):
    """CLI PyGrid Apps Deployment.

    Example:

    >>> pygrid deploy --provider aws --app domain

    >>> pygrid deploy --provider azure --app network
    """

    config.provider = provider.lower()

    # Store credentials in a separate object, thus not logging it in output
    # when asking the user to confirm the current configuration
    credentials = Config()

    # credentials file
    cred_prompt = f"Please enter path to your  {colored(f'{config.provider} credentials')} json file"
    cred_default_path = f"{Path.home()}/.{config.provider}/credentials.json"
    with open(click.prompt(cred_prompt, type=str, default=cred_default_path), "r") as f:
        credentials.cloud = Config(**json.load(f))

    ## Get app config and arguments
    config.app = Config(name=app.lower())

    config.root_dir = os.path.join(
        str(Path.home()), ".pygrid", "apps", str(config.provider), str(config.app.name)
    )

    ## Deployment type
    # config.serverless = False
    # TODO: Uncomment this after updating serverless deployment
    # if config.app.name == "network":
    config.serverless = click.confirm(f"Do you want to deploy serverless?")

    ## Websockets
    if not config.serverless:
        config.websockets = click.confirm(f"Will you need to support Websockets?")

    if not config.serverless:
        get_app_arguments(config)

    ## Prompting user to provide configuration for the selected cloud
    if config.provider == "aws":
        vpc, instance_type = None, None
        while vpc is None or instance_type is None:
            try:
                vpc = aws_utils.get_vpc_config()
                if not config.serverless:
                    instance_type = aws_utils.get_instance_type(vpc.region)
            except Exception as e:
                click.echo(colored(str(e), color=COLORS.red))
        config.vpc = vpc
        config.vpc.instance_type = instance_type
    elif config.provider == "gcp":
        config.gcp = gcp_utils.get_gcp_config()
    elif config.provider == "azure":
        config.azure = azure_utils.get_azure_config()

    ## Database
    credentials.db = aws_utils.get_db_config()

    ## TODO(amr): [clean] For quick dev stuff
    # with open("/Users/amrmkayid/.pygrid/cli/azure.json", "rb") as config_json:
    #     config = Config(**json.load(config_json))

    config_json = json.dumps(vars(config), indent=2, default=lambda o: o.__dict__)
    if click.confirm(
        f"""Your current configration are:\n\n{colored(config_json)}\n\nContinue?"""
    ):

        config.credentials = credentials
        click.echo(colored("STARTING DEPLOYMENT... 🔃", color=COLORS.green))

        deployed, output = _deploy(config)

        msg, color = (
            ("\nDEPLOYMENT SUCCESSFUL\n", COLORS.green)
            if deployed
            else ("\nDEPLOYMENT FAILED\n", COLORS.red)
        )
        click.echo(colored(msg, color=color))


def _deploy(config):
    deployment = None
    if config.provider == "aws":
        deployment = (
            AWS_Serverless(config)
            if config.serverless
            else AWS_Serverfull(config=config)
        )
    elif config.provider == "azure":
        deployment = AZURE(config)
    elif config.provider == "gcp":
        deployment = GCP(config)

    if deployment.validate():
        # deployed, output = True, {}
        deployed, output = deployment.deploy()
    else:
        deployed, output = (
            False,
            {"failure": f"Your attempt to deploy PyGrid {config.app.name} failed"},
        )
    return deployed, output


def get_app_arguments(config):
    config.app.count = click.prompt(
        f"How many servers do you wish to deploy? (All are managed under the load balancer)",
        type=int,
        default=1,
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
            app = Config(id=id, port=port)
        else:  # Network
            port = click.prompt(
                f"#{count}: Port number of the socket.io server",
                type=str,
                default=os.environ.get("GRID_NETWORK_PORT", f"{7000 + count}"),
            )
            app = Config(port=port)
        apps.append(app)
    config.apps = apps


@cli.resultcallback()
@pass_config
def logging(config, results, **kwargs):
    click.echo(f"Writing configs to {config.output_file}")
    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(vars(config), f, indent=2, default=lambda o: o.__dict__)
