# stdlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from typing import Generator
from typing import Optional

# third party
import nest_asyncio
import requests

# syft relative
from ...core.node.domain.domain import Domain
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


class bcolors:
    FAIL = "\033[91m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    OKBLUE = "\033[94m"
    HEADER = "\033[95m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def generate_donation_msg(name: str) -> str:
    donate_url = "https://github.com/sponsors/OpenMined"
    donate_msg = (
        f"\n    > ❤️ {bcolors.FAIL}Love{bcolors.ENDC} {bcolors.OKGREEN}{name}{bcolors.ENDC}? "
        + f"{bcolors.WARNING}Please{bcolors.ENDC} {bcolors.OKBLUE}consider{bcolors.ENDC} "
        + f"{bcolors.HEADER}supporting{bcolors.ENDC} {bcolors.FAIL}our{bcolors.ENDC} "
        + f"{bcolors.WARNING}community!{bcolors.ENDC}"
        + f"\n    > {donate_url}"
    )
    return donate_msg


DUET_DONATE_MSG = generate_donation_msg(name="Duet")

try:
    # third party
    from IPython.core.display import Image
    from IPython.core.display import display

    jupyter = True
except ImportError:
    jupyter = False


# for local debugging
def get_loopback_path() -> str:
    loopback_file = "duet_loopback.json"
    return str(Path(tempfile.gettempdir()) / loopback_file)


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
    print(bcolors.BOLD + DUET_DONATE_MSG + bcolors.BOLD + "\n")

    if not network_url:
        network_url = get_available_network()
    print("♫♫♫ > Punching through firewall to OpenGrid Network Node at:")
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
        + " Send the following code to your Duet Partner!"
    )
    #         print(f"♫♫♫ > Duet Node ID:{domain.id.value}")

    print("\nimport syft as sy")
    print(
        'duet = sy.duet("'
        + bcolors.BOLD
        + signaling_client.duet_id
        + bcolors.ENDC
        + '")'
    )

    # use a local file to automatically exchange duet ids
    if loopback is True:
        loopback_config = {}
        loopback_config["server_id"] = signaling_client.duet_id
        with open(get_loopback_path(), "w") as f:
            f.write(json.dumps(loopback_config))

    my_domain = Domain(name="Launcher", db_path=db_path)

    print(
        "\n♫♫♫ > "
        + bcolors.HEADER
        + "STEP 2:"
        + bcolors.ENDC
        + " Running the code above will print out a 'Client ID'."
    )
    print("♫♫♫ >         Have your duet partner send it to you and enter it below!")
    print()
    if loopback is False:
        while True:
            target_id = input("♫♫♫ > Duet Partner's Client ID: ")  # nosec
            if len(target_id) == 32:
                break
            else:
                print("    > Error: Invalid Client ID. Please try again.")

    else:
        target_id = ""
        print(
            "Running loopback mode. Use sy.join_duet(loopback=True) on the other side."
        )
        while target_id == "":
            try:
                with open(get_loopback_path(), "r") as f:
                    loopback_config = json.loads(f.read())
                    if "client_id" in loopback_config:
                        target_id = str(loopback_config["client_id"])
                    else:
                        time.sleep(0.5)
            except Exception as e:
                print(e)
                break

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
) -> WebRTCDuet:
    if os.path.isfile(LOGO_URL) and jupyter:
        display(
            Image(
                LOGO_URL,
                width=400,
                unconfined=True,
            )
        )
    if target_id == "" and loopback is False:
        cmd = 'join_duet("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")'
        raise Exception(f"You must enter a Duet Server ID like this: {cmd}")
    print("🎤  🎸  ♪♪♪ Joining Duet ♫♫♫  🎻  🎹\n")
    sys.stdout.write(
        "♫♫♫ >\033[93m" + " DISCLAIMER" + "\033[0m"
        ":"
        + "\033[1m"
        + " Duet is an experimental feature currently in beta.\n"
        + "♫♫♫ >             Use at your own risk.\n"
        + "\n♫♫♫ > \n"
        + "\033[0m"
    )

    print("♫♫♫ >")
    print(bcolors.BOLD + DUET_DONATE_MSG + bcolors.BOLD + "\n")

    if not network_url:
        network_url = get_available_network()
    print("♫♫♫ > Punching through firewall to OpenGrid Network Node at:")
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
    if loopback:
        loopback_config = {}
        target_id = ""
        while target_id == "":
            try:
                with open(get_loopback_path(), "r") as f:
                    loopback_config = json.loads(f.read())
                    # only continue once the server has overwritten the file
                    # with only its new server_id
                    if (
                        "server_id" in loopback_config
                        and "client_id" not in loopback_config
                    ):
                        target_id = str(loopback_config["server_id"])
                    else:
                        time.sleep(0.5)
            except Exception as e:
                print(e)
                break

        loopback_config["client_id"] = signaling_client.duet_id

        with open(get_loopback_path(), "w") as f:
            f.write(json.dumps(loopback_config))

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
