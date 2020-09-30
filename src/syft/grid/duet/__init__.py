# stdlib
import json
import sys
from typing import Any
from typing import Generator

# third party
import nest_asyncio
import requests

# syft relative
from ...core.node.domain.domain import Domain
from .duet import Duet  # noqa: F401
from .om_signaling_client import register
from .webrtc_duet import Duet as WebRTCDuet  # noqa: F811

nest_asyncio.apply()

ADDR_REPOSITORY = (
    "https://raw.githubusercontent.com/OpenMined/OpenGridNodes/master/network_address"
)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


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

    counterThread().start()


def launch_duet(
    logging: bool = True,
    network_url: str = "",
) -> WebRTCDuet:
    print("🎤  🎸  ♪♪♪ starting duet ♫♫♫  🎻  🎹\n")
    sys.stdout.write(
        "♫♫♫ >\033[93m" + " DISCLAIMER" + "\033[0m"
        ":"
        + "\033[1m"
        + " Duet is an experimental feature currently \n♫♫♫ > "
        + "in alpha. Do not use this to protect real-world data.\n"
        + "\033[0m"
    )

    print("♫♫♫ >")
    print("♫♫♫ > Punching through firewall to OpenGrid Network Node at network_url: ")

    if not network_url:
        network_url = get_available_network()
    print("♫♫♫ > " + str(network_url))
    print("♫♫♫ >")
    sys.stdout.write("♫♫♫ > ...waiting for response from OpenGrid Network... ")

    signaling_client = register(url=network_url)

    print(bcolors.OKGREEN + "DONE!" + bcolors.ENDC)

    #     print("♫♫♫ >")
    #     print("♫♫♫ > Your Duet Id: " + signaling_client.duet_id)

    print("♫♫♫ >")
    print(
        "♫♫♫ > "
        + bcolors.HEADER
        + "STEP 1:"
        + bcolors.ENDC
        + " Send the following code to your duet partner!"
    )
    #         print(f"♫♫♫ > Duet Node ID:{domain.id.value}")

    print("\nimport syft as sy")
    print("sy.VERBOSE=False")
    print(
        "duet = sy.join_duet('"
        + bcolors.BOLD
        + signaling_client.duet_id
        + bcolors.ENDC
        + "')"
    )

    my_domain = Domain(name="Launcher")

    print(
        "\n♫♫♫ > "
        + bcolors.HEADER
        + "STEP 2:"
        + bcolors.ENDC
        + " The code above will print out a 'Client Id'. Have"
    )
    print("♫♫♫ >         your duet partner send it to you and enter it below!")
    print()
    target_id = input("♫♫♫ > Duet Partner's Client Id:")  # nosec
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
        begin_duet_logger(my_domain)
    print()

    return out_duet
    # return duet


def join_duet(
    target_id: str,
    network_url: str = "",
) -> WebRTCDuet:
    print("🎤  🎸  ♪♪♪ joining duet ♫♫♫  🎻  🎹\n")
    sys.stdout.write(
        "♫♫♫ >\033[93m" + " DISCLAIMER" + "\033[0m"
        ":"
        + "\033[1m"
        + " Duet is an experimental feature currently \n♫♫♫ > "
        + "in alpha. Do not use this to protect real-world data.\n"
        + "\033[0m"
    )

    print("♫♫♫ >")
    print("♫♫♫ > Punching through firewall to OpenGrid Network Node at network_url: ")
    if not network_url:
        network_url = get_available_network()
    print("♫♫♫ > " + str(network_url))
    print("♫♫♫ >")
    sys.stdout.write("♫♫♫ > ...waiting for response from OpenGrid Network... ")

    signaling_client = register(url=network_url)

    print(bcolors.OKGREEN + "DONE!" + bcolors.ENDC)

    my_domain = Domain(name="Joiner")
    print()
    print(
        "♫♫♫ > Duet Client ID: "
        + bcolors.BOLD
        + signaling_client.duet_id
        + bcolors.ENDC
    )
    print()
    print(
        "♫♫♫ > "
        + bcolors.HEADER
        + "STEP 1:"
        + bcolors.ENDC
        + " Send the Duet Client ID to your duet partner!"
    )
    print()
    print("♫♫♫ > ...waiting for partner to connect...")
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
