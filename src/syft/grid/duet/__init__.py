# syft relative
from ...core.node.domain.domain import Domain
from .om_signaling_client import register
from .duet import Duet  # noqa: F401

# third party libraries
import nest_asyncio
import sys

nest_asyncio.apply()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def begin_duet_logger(my_domain):
    import sys
    import threading
    import time
    from contextlib import contextmanager

    # we need a lock, so that other threads don't snatch control
    # while we have set a temporary parent
    stdout_lock = threading.Lock()

    @contextmanager
    def set_stdout_parent(parent):
        """a context manager for setting a particular parent for sys.stdout

        the parent determines the destination cell of output
        """
        save_parent = sys.stdout.parent_header
        with stdout_lock:
            sys.stdout.parent_header = parent
            try:
                yield
            finally:
                # the flush is important, because that's when the parent_header actually has its effect
                sys.stdout.flush()
                sys.stdout.parent_header = save_parent

    class counterThread(threading.Thread):
        def run(self):
            # record the parent when the thread starts
            thread_parent = sys.stdout.parent_header
            iterator = 0
            while (True):
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

                    out = "♫♫♫ > DUET LIVE STATUS  " + star + "  Objects: " + str(
                        n_objects) + "  " + left_color + "Requests:" + right_color + left_blink + str(
                        n_requests) + right_blink + "  Messages: " + str(n_messages)
                    out += "                                "
                    sys.stdout.write("\r" + out)
                iterator += 1

    counterThread().start()


def launch_duet(logging=True, network_url="http://ec2-18-191-23-46.us-east-2.compute.amazonaws.com:5000"):
    from .webrtc_duet import Duet
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
    print("♫♫♫ > " + str(network_url))
    print("♫♫♫ >")
    sys.stdout.write("♫♫♫ > ...waiting for response from OpenGrid Network... ")

    signaling_client = register(url=network_url)

    print(bcolors.OKGREEN + "DONE!" + bcolors.ENDC)

    #     print("♫♫♫ >")
    #     print("♫♫♫ > Your Duet Id: " + signaling_client.duet_id)

    print("♫♫♫ >")
    print("♫♫♫ > " + bcolors.HEADER + "STEP 1:" + bcolors.ENDC + " Send the following code to your duet partner!")
    #         print(f"♫♫♫ > Duet Node ID:{domain.id.value}")

    print("\n import syft as sy")
    print("\n sy.join_duet('" + signaling_client.duet_id + "')")

    my_domain = Domain(name="Launcher")

    print("\n♫♫♫ > " + bcolors.HEADER + "STEP 2:" + bcolors.ENDC + " The code above will print out a 'Client Id'. Have")
    print("♫♫♫ >         your duet partner send it to you and enter it below!")
    print()
    target_id = input("♫♫♫ > Duet Partner's Client Id:")
    print("♫♫♫ > Connecting...")

    duet = Duet(node=my_domain,
                target_id=target_id,
                signaling_client=signaling_client,
                offer=True)
    print()
    print("♫♫♫ > " + bcolors.OKGREEN + "CONNECTED!" + bcolors.ENDC)
    #     return duet, my_domain.get_root_client()
    out_duet = my_domain.get_root_client()

    if (logging):
        begin_duet_logger(my_domain)
    print()
    return out_duet