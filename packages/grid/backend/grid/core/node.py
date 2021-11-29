# stdlib
import logging
import threading
import time
from typing import Optional

# third party
from nacl.signing import SigningKey

# syft absolute
from syft import Domain  # type: ignore
from syft import Network  # type: ignore
from syft.core.node.common.client import Client
from syft.core.node.common.node_table.utils import seed_db

# grid absolute
from grid.core.config import settings
from grid.db.session import get_db_engine
from grid.db.session import get_db_session


def thread_function(*args, **kwargs) -> None:

    time.sleep(60)

    # third party
    from requests import get

    ip = get("https://api.ipify.org").content.decode("utf8")
    print(f"My public IP address is: {ip}")

    try:

        NETWORK_PUBLIC_HOST = "http://" + ip + ":80"
        # syft absolute
        import syft as sy

        network_root = sy.login(
            email="info@openmined.org",
            password="changethis",
            url="http://" + ip,
            port=80,
        )
    except Exception as e:
        NETWORK_PUBLIC_HOST = "http://localhost:80"

        network_root = sy.login(
            email="info@openmined.org",
            password="changethis",
            url="http://localhost",
            port=80,
        )

    network_root.join_network(host_or_ip=NETWORK_PUBLIC_HOST)


if settings.NODE_TYPE.lower() == "domain":
    node = Domain("Domain", db_engine=get_db_engine(), settings=settings)
elif settings.NODE_TYPE.lower() == "network":
    node = Network("Network", db_engine=get_db_engine(), settings=settings)
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread")
    x = threading.Thread(target=thread_function)
    logging.info("Main    : before running thread")
    x.start()
    logging.info("Main    : wait for the thread to finish")
    # x.join()
    logging.info("Main    : all done")

else:
    raise Exception(
        "Don't know NODE_TYPE "
        + str(settings.NODE_TYPE)
        + ". Please set "
        + "NODE_TYPE to either 'Domain' or 'Network'."
    )

node.loud_print()

if len(node.setup):  # Check if setup was defined previously
    node.name = node.setup.node_name

# Moving this to get called WITHIN Domain and Network so that they can operate in standalone mode
if not len(node.roles):  # Check if roles were registered previously
    seed_db(get_db_session())


def get_client(signing_key: Optional[SigningKey] = None) -> Client:
    return node.get_client(signing_key=signing_key)
