# stdlib
from getpass import getpass
import json
import logging
import sys
import time
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
import requests

# relative
from .. import GridURL
from ... import __version__
from ...core.io.connection import ClientConnection
from ...core.io.route import SoloRoute
from ...core.node.common.client import Client
from ...core.node.domain_client import DomainClient
from ...core.node.network_client import NetworkClient
from ...util import bcolors
from ...util import verify_tls
from .grid_connection import GridHTTPConnection

DEFAULT_PYGRID_PORT = 80
DEFAULT_PYGRID_ADDRESS = f"http://127.0.0.1:{DEFAULT_PYGRID_PORT}"


def connect(
    url: Union[str, GridURL] = DEFAULT_PYGRID_ADDRESS,
    conn_type: Type[ClientConnection] = GridHTTPConnection,
    credentials: Optional[Dict] = None,
    user_key: Optional[SigningKey] = None,
    timeout: Optional[float] = None,
) -> Client:
    # Use Server metadata
    # to build client route
    conn = conn_type(url=GridURL.from_url(url))  # type: ignore

    # get metadata and check for https redirect so that login is sent over TLS
    metadata = conn._get_metadata(timeout=timeout)  # type: ignore

    if not user_key:
        credentials = credentials if credentials is not None else {}
        metadata, _user_key = conn.login(credentials=credentials)  # type: ignore
        _user_key = SigningKey(_user_key.encode(), encoder=HexEncoder)
    else:
        # metadata = conn.auth_using_key(user_key=user_key)  # type: ignore
        _user_key = user_key

    # Check node client type based on metadata response
    client_type: Union[Type[DomainClient], Type[NetworkClient]]
    if metadata.node_type == "Domain":
        client_type = DomainClient
    else:
        client_type = NetworkClient

    (
        spec_location,
        name,
        client_id,
    ) = client_type.deserialize_client_metadata_from_node(metadata=metadata)

    # Create a new Solo Route using the selected connection type
    route = SoloRoute(destination=spec_location, connection=conn)

    kwargs = {
        "name": name,
        "routes": [route],
        "signing_key": _user_key,
        "version": metadata.version,
    }

    if client_type is NetworkClient:
        kwargs["network"] = spec_location
    elif client_type is DomainClient:
        kwargs["domain"] = spec_location
    else:
        raise NotImplementedError

    # Create a new client using the selected client type
    node = client_type(**kwargs)

    return node


def login(
    url: Optional[Union[str, GridURL]] = None,
    port: Optional[int] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    conn_type: Type[ClientConnection] = GridHTTPConnection,
    verbose: Optional[bool] = True,
    timeout: Optional[float] = None,
    retry: Optional[int] = None,
) -> Client:

    retry = 5 if retry is None else retry  # Default to 5 retries
    timeout = 10 if timeout is None else timeout  # Default to 10 seconds

    if password == "changethis":  # nosec

        if email == "info@openmined.org":
            print(
                f"{bcolors.YELLOW}WARNING:{bcolors.ENDC} CHANGE YOUR USERNAME AND PASSWORD!!! \n\n"
                + "Anyone can login as an admin to your node"
                + " right now because your password is still the default PySyft username and password!!!\n"
            )
        else:
            print(
                f"{bcolors.YELLOW}WARNING:{bcolors.ENDC} CHANGE YOUR PASSWORD!!! \n\n"
                + "Anyone can login into your account"
                + " right now because your password is the default PySyft password!!!\n"
            )

    # TRASK: please keep this so that people will stop putting their passwords in notebooks.
    if password == "secret":  # nosec
        print("Welcome " + str(email) + "!")
        password = getpass(prompt="Please enter you password:")

    if port is None and not url:  # if url is used, we can ignore port
        port = int(input("Please specify the port of the domain you're logging into:"))

    # TODO: build multiple route objects and let the Client decide which one to use
    if isinstance(url, GridURL):
        grid_url = url
    elif url is None:
        grid_url = GridURL(host_or_ip="docker-host", port=port, path="/api/v1/status")
        try:
            requests.get(str(grid_url), verify=verify_tls())
        except Exception:
            grid_url.host_or_ip = "localhost"
    else:
        grid_url = GridURL(host_or_ip=url, port=port)

    grid_url = grid_url.with_path("/api/v1")

    if verbose:
        sys.stdout.write("\rConnecting to " + str(grid_url.host_or_ip) + "...")

    if email is None or password is None:
        credentials = {}
        logging.info(
            "\n\nNo email and password defined in login() - connecting as anonymous user!!!\n"
        )
    else:
        credentials = {"email": email, "password": password}

    # connecting to domain
    node = None
    timeout_btw_retries = timeout
    retry_attempt = 1

    while node is None and retry_attempt <= retry:
        try:
            node = connect(
                url=grid_url,
                credentials=credentials,
                conn_type=conn_type,
                timeout=timeout,
            )
        except requests.ReadTimeout:
            print(
                f"\n{bcolors.BOLD}{bcolors.RED}ReadTimeout:{bcolors.ENDC}\n"
                f"\tConnection to node with url: {grid_url.host_or_ip}:{grid_url.port} "
                f"timed out after {timeout} seconds.\n"
                "\tPlease try the following options:\n"
                "\t- Please try increasing the timeout by passing it as an argument to the login method.\n"
                "\te.g. `sy.login(email='my@email.com', password='password', url='localhost', timeout=30)`\n"
                "\t- The domain/network node you're trying to connect could be offline "
                "at the current moment. Please try again later.\t"
            )
            return  # type: ignore

        except requests.ConnectionError as e:
            if retry_attempt <= retry:
                print(
                    f"\r{bcolors.BOLD}ConnectionError{bcolors.ENDC}: Retrying again.... Attempt: {retry_attempt}",
                    end="\r",
                )
                time.sleep(timeout_btw_retries)
            else:
                raise e
        retry_attempt += 1

    if node is None:
        print(
            f"\n{bcolors.BOLD}{bcolors.RED}ConnectionError:{bcolors.ENDC}\n"
            f"\tOops !!! We can't seem to connect to the node: '{grid_url.host_or_ip}:{grid_url.port}'\n"
            "\tPlease try the following options:\n"
            f"\t- Are you sure the server at '{grid_url.host_or_ip}:{grid_url.port}' is running? "
            "Please check the `url`/`port` you entered are correct.\n"
            f"\t- Are you sure you can connect to the server at '{grid_url.host_or_ip}:{grid_url.port}'? "
            "Perhaps there's a firewall between you and the server?\n"
            "\t- The domain/network node you're trying to connect could be offline "
            "at the current moment. Please try again later.\n"
        )
        return  # type: ignore

    if verbose:
        # bit of fanciness
        sys.stdout.write("\rConnecting to " + str(grid_url.host_or_ip) + "...")
        sys.stdout.write(" done! \t Logging into")
        sys.stdout.write(" " + str(node.name) + "... ")
        if email is None or password is None:
            sys.stdout.write("as GUEST...")
        time.sleep(1)  # ok maybe too fancy... but c'mon don't you want to be fancy?
        print("done!")
    else:
        print("Logging into", str(node.name), "... done!")

    if __version__ != node.version:
        print(
            "\n**Warning**: The syft version on your system and the node are different."
        )
        print(
            f"Version on your system: {__version__}\nVersion on the node: {node.version}"
        )
        print()

    return node


def register(
    name: Optional[str] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    url: Optional[str] = None,
    port: Optional[int] = None,
    verbose: Optional[bool] = True,
) -> Client:
    if name is None:
        name = input("Please enter your name:")

    if email is None:
        email = input("Please enter your email:")

    if password is None:
        password = getpass("Please enter your password")

    if url is None:
        url = input("Please enter URL of domain (ex: 'localhost'):")

    if port is None:
        port = int(input("Please enter the port your domain is running on:"))

    grid_url = GridURL(host_or_ip=url, port=port)

    register_url = grid_url.url + "/api/v1/register"
    myobj = {"name": name, "email": email, "password": password}

    response = requests.post(register_url, data=json.dumps(myobj))

    if "error" not in json.loads(response.text):
        if verbose:
            print("Successfully registered! Logging in...")

        return login(
            url=grid_url, port=port, email=email, password=password, verbose=verbose
        )

    raise Exception(response.text)
