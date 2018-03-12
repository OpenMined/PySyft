from grid import workers
import argparse
import time
from colorama import Fore, Style

title = f"""{Fore.GREEN}   ____                             _                __   ______     _     __
  / __ \____  ___  ____  ____ ___  (_____  ___  ____/ /  / _________(_____/ /
 / / / / __ \/ _ \/ __ \/ __ `__ \/ / __ \/ _ \/ __  /  / / __/ ___/ / __  /
/ /_/ / /_/ /  __/ / / / / / / / / / / / /  __/ /_/ /  / /_/ / /  / / /_/ /
\____/ .___/\___/_/ /_/_/ /_/ /_/_/_/ /_/\___/\__,_/   \____/_/  /_/\__,_/
    /_/          {Style.RESET_ALL}{Fore.YELLOW}A distributed compute grid{Style.RESET_ALL}
"""

print(title)

program_desc = f"""
"""

# print(title)

parser = argparse.ArgumentParser(description=program_desc)

parser.add_argument(
    '--compute',
    dest='compute',
    action='store_const',
    const=True,
    default=True,
    help='Run grid in compute mode')

parser.add_argument(
    '--tree',
    dest='tree',
    action='store_const',
    const=True,
    default=False,
    help='Run grid in tree mode')

parser.add_argument(
    '--anchor',
    dest='anchor',
    action='store_const',
    const=True,
    default=False,
    help='Run grid in anchor mode')

args = parser.parse_args()
"""
TODO: modify Client to store the source code for the model in IPFS.
      (think through logistics; introduces
      hurdles for packaging model source code)
TODO: figure out a convenient way to make robust training procedure for torch
      -- will probably want to use ignite for this
"""

logf = open("openmined.errors", "w")


def run():
    try:
        print("\n\n")

        if (args.tree):
            workers.tree.GridTree()
        elif (args.anchor):
            workers.anchor.GridAnchor()
        else:
            workers.compute.GridCompute()

    except Exception as e:  # most generic exception you can catch
        print(e)
        time.sleep(1000)
        logf.write("Failed to download {0}: {1}\n".format(
            str(download), str(e)))
        run()


run()
