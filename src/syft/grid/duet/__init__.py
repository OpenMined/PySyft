# stdlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from typing import Generator
from typing import Optional

# third party
import nest_asyncio
import requests

# syft relative
from ...core.node.domain.domain import Domain
from .bcolors import bcolors
from .exchange_ids import DuetCredentialExchanger
from .exchange_ids import OpenGridTokenFileExchanger
from .exchange_ids import OpenGridTokenManualInputExchanger
from .om_signaling_client import register
from .webrtc_duet import Duet as WebRTCDuet  # noqa: F811

try:
    nest_asyncio.apply()
except RuntimeError as e:
    # this happens when pytest-xdist parallel threaded tests are run on some systems
    print("Nothing to patch", e)

ADDR_REPOSITORY = (
    "https://raw.githubusercontent.com/OpenMined/OpenGridNodes/master/network_address"
)

LOGO_URL = os.path.abspath(Path(__file__) / "../../../img/logo.png")


try:
    # third party
    from IPython.core.display import Image
    from IPython.core.display import display

    jupyter = True
except ImportError:
    jupyter = False


def get_available_network() -> str:
    network_addr = json.loads(requests.get(ADDR_REPOSITORY).content)
    for addr in network_addr:
        try:
            requests.get(addr + "/metadata")
            return addr
        except Exception:
            continue
    raise Exception("Couldn't find any available network.")


def begin_duet_logger(my_domain: Domain) -> None:
    # stdlib
    from contextlib import contextmanager
    import sys
    import threading
    import time

    # we need a lock, so that other threads don't snatch control
    # while we have set a temporary parent
    stdout_lock = threading.Lock()

    @contextmanager
    def set_stdout_parent(parent: Any) -> Generator:
        """a context manager for setting a particular parent for sys.stdout
        the parent determines the destination cell of output
        """
        save_parent = sys.stdout.parent_header  # type: ignore
        with stdout_lock:
            sys.stdout.parent_header = parent  # type: ignore
            try:
                yield
            finally:
                # the flush is important, because that's when the parent_header actually has its effect
                sys.stdout.flush()
                sys.stdout.parent_header = save_parent  # type: ignore

    class counterThread(threading.Thread):
        def run(self) -> None:
            # record the parent when the thread starts
            thread_parent = sys.stdout.parent_header  # type: ignore
            iterator = 0
            while True:
                time.sleep(0.1)
                # then ensure that the parent is the same as when the thread started
                # every time we print
                with set_stdout_parent(thread_parent):

                    n_objects = len(my_domain.store)
                    n_requests = len(my_domain.requests)
                    n_messages = my_domain.message_counter

                    blink_on = (int(iterator / 5) % 2) == 0

                    if blink_on and n_requests > 0:
                        left_blink = bcolors.BOLD + ">" + bcolors.ENDC
                        right_blink = bcolors.BOLD + "<" + bcolors.ENDC
                        left_color = bcolors.FAIL
                        right_color = bcolors.ENDC
                    else:
                        left_blink = " "
                        right_blink = " "
                        left_color = ""
                        right_color = ""

                    if blink_on:
                        star = "*"
                    else:
                        star = "-"

                    out = (
                        "♫♫♫ > DUET LIVE STATUS  "
                        + star
                        + "  Objects: "
                        + str(n_objects)
                        + "  "
                        + left_color
                        + "Requests:"
                        + right_color
                        + left_blink
                        + str(n_requests)
                        + right_blink
                        + "  Messages: "
                        + str(n_messages)
                    )
                    out += "                                "
                    sys.stdout.write("\r" + out)
                iterator += 1

    if hasattr(sys.stdout, "parent_header"):
        counterThread().start()


def duet(
    target_id: Optional[str] = None,
    logging: bool = True,
    network_url: str = "",
    loopback: bool = False,
    db_path: Optional[str] = None,
) -> WebRTCDuet:
    if target_id is not None:
        return join_duet(
            target_id=target_id, loopback=loopback, network_url=network_url
        )
    else:
        return launch_duet(
            logging=logging, network_url=network_url, loopback=loopback, db_path=db_path
        )


def launch_duet(
    logging: bool = True,
    network_url: str = "",
    loopback: bool = False,
    credential_exchanger: DuetCredentialExchanger = OpenGridTokenManualInputExchanger(),
    db_path: Optional[str] = None,
) -> WebRTCDuet:
    if os.path.isfile(LOGO_URL) and jupyter:
        display(
            Image(
                LOGO_URL,
                width=400,
                unconfined=True,
            )
        )
    print("🎤  🎸  ♪♪♪ Starting Duet ♫♫♫  🎻  🎹\n")
    sys.stdout.write(
        "♫♫♫ >\033[93m" + " DISCLAIMER" + "\033[0m"
        ":"
        + "\033[1m"
        + " Duet is an experimental feature currently in beta.\n"
        + "♫♫♫ >             Use at your own risk.\n"
        + "\033[0m"
    )

    print("♫♫♫ >")

    if not network_url:
        network_url = get_available_network()
    print("♫♫♫ > Punching through firewall to OpenGrid Network Node at:")
    print("♫♫♫ > " + str(network_url))
    print("♫♫♫ >")
    sys.stdout.write("♫♫♫ > ...waiting for response from OpenGrid Network... ")

    signaling_client = register(url=network_url)

    print(bcolors.OKGREEN + "DONE!" + bcolors.ENDC)

    my_domain = Domain(name="Launcher", db_path=db_path)

    if loopback:
        credential_exchanger = OpenGridTokenFileExchanger()
    target_id = credential_exchanger.run(credential=signaling_client.duet_id)

    print("♫♫♫ > Connecting...")

    _ = WebRTCDuet(
        node=my_domain,
        target_id=target_id,
        signaling_client=signaling_client,
        offer=True,
    )
    print()
    print("♫♫♫ > " + bcolors.OKGREEN + "CONNECTED!" + bcolors.ENDC)
    #     return duet, my_domain.get_root_client()
    out_duet = my_domain.get_root_client()

    if logging:
        begin_duet_logger(my_domain=my_domain)
    print()

    return out_duet


def join_duet(
    target_id: str = "",
    network_url: str = "",
    loopback: bool = False,
    credential_exchanger: DuetCredentialExchanger = OpenGridTokenManualInputExchanger(),
) -> WebRTCDuet:
    if os.path.isfile(LOGO_URL) and jupyter:
        display(
            Image(
                LOGO_URL,
                width=400,
                unconfined=True,
            )
        )
    print("🎤  🎸  ♪♪♪ Joining Duet ♫♫♫  🎻  🎹\n")
    sys.stdout.write(
        "♫♫♫ >\033[93m" + " DISCLAIMER" + "\033[0m"
        ":"
        + "\033[1m"
        + " Duet is an experimental feature currently in beta.\n"
        + "♫♫♫ >             Use at your own risk.\n"
        + "\033[0m"
    )

    print("♫♫♫ >")

    if not network_url:
        network_url = get_available_network()
    print("♫♫♫ > Punching through firewall to OpenGrid Network Node at:")
    print("♫♫♫ > " + str(network_url))
    print("♫♫♫ >")
    sys.stdout.write("♫♫♫ > ...waiting for response from OpenGrid Network... ")

    signaling_client = register(url=network_url)

    print(bcolors.OKGREEN + "DONE!" + bcolors.ENDC)

    my_domain = Domain(name="Joiner")

    if loopback:
        credential_exchanger = OpenGridTokenFileExchanger()
        credential_exchanger.set_role(join=True)
    else:
        credential_exchanger.set_role(join=True).run(credential=target_id)

    if target_id != "":
        credential_exchanger.set_responder_id(credential=target_id)

    target_id = credential_exchanger.run(credential=signaling_client.duet_id)

    duet = WebRTCDuet(
        node=my_domain,
        target_id=target_id,
        signaling_client=signaling_client,
        offer=False,
    )
    print()
    print("♫♫♫ > " + bcolors.OKGREEN + "CONNECTED!" + bcolors.ENDC)
    # begin_duet_client_logger(duet.node)

    return duet
