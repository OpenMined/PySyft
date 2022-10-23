# stdlib
import locale
import os
import secrets

# third party
import ascii_magic
import rich
from rich.emoji import Emoji


def motorcycle() -> None:
    print(
        """
                                             `
                                         `.+yys/.`
                                       ``/NMMMNNs`
                                    `./shNMMMMMMNs``    `..`
                                  `-smNMMNNMMMMMMN/.``......`
                                `.yNMMMMNmmmmNNMMm/.`....`
                              `:sdNMMMMMMNNNNddddds-`.`` `--. `
                           `.+dNNNNMMMMMMMMMNNNNmddohmh//hddy/.```..`
                          `-hNMMMMMMMMMMMMNNdmNNMNNdNNd:sdyoo+/++:..`
                        ../mMMMMMMMMMMMMMMNNmmmmNMNmNNmdmd/hNNNd+:`
                        `:mMNNMMMMMMMMMMMMNMNNmmmNNNNNdNNd/NMMMMm::
                       `:mMNNNMMMMMMMMMMMMMMMNNNNdNMNNmmNd:smMMmh//
                     ``/mMMMMMMMMMMMMMMMMMMMMMMNmdmNNMMNNNy/osoo/-`
                    `-sNMMMMMMMMMMMMMMMMMMMMMMMMNNmmMMMMNh-....`
                   `/dNMMMMMMMMMMMMMMMMMMMMMMMMMMMNNMMMNy.`
                ``.omNNMMMMMMMMMMMMNMMMMMMMNmmmmNNMMMMN+`
                `:hmNNMMMMMMMMMMMNo/ohNNNNho+os+-+hNys/`
                -mNNNNNNMMMMMMMMm+``-yNdd+/mMMMms.-:`
                .+dmNNNNMMMMMMNd:``:dNN+y`oMMMMMm-.`
                `+dmmmNNNmmmmy+.   `-+m/s/+MMMMm/--
               `+mmmhNy/-...```     ``-.-sosyys+/-`
            ``.smmmsoo``               .oh+-:/:.
          `.:odmdh/````             `.+d+``````
     ```/sydNdhy+.`              ``-sds.
    `:hdmhs::-````               `oNs.`
```.sdmh/``                    `-ym+`
 ``ssy+`                     `-yms.`
   ``                      `:hNy-``
   `                     `-yMN/```
                       `-yNhy-
                     `/yNd/`
                   `:dNMs``
                 `.+mNy/.`
              `.+hNMMs``
             `:dMMMMh.`"""  # noqa: W605
    )


def hold_on_tight() -> None:
    out = os.popen("stty size", "r").read().split()  # nosec
    if len(out) == 2:
        rows, columns = out
    else:
        """not running in a proper command line (probably a unit test)"""
        return

    if int(columns) >= 91:

        print(
            """
 _   _       _     _                 _   _       _     _     _   _                       _
| | | |     | |   | |               | | (_)     | |   | |   | | | |                     | |
| |_| | ___ | | __| |   ___  _ __   | |_ _  __ _| |__ | |_  | |_| | __ _ _ __ _ __ _   _| |
|  _  |/ _ \| |/ _` |  / _ \| '_ \  | __| |/ _` | '_ \| __| |  _  |/ _` | '__| '__| | | | |
| | | | (_) | | (_| | | (_) | | | | | |_| | (_| | | | | |_  | | | | (_| | |  | |  | |_| |_|
\_| |_/\___/|_|\__,_|  \___/|_| |_|  \__|_|\__, |_| |_|\__| \_| |_/\__,_|_|  |_|   \__, (_)
                                            __/ |                                   __/ |
                                           |___/                                   |___/
            """  # noqa: W605
        )
    else:
        print(
            """
 _   _       _     _                 _   _                       _
| | | |     | |   | |               | | | |                     | |
| |_| | ___ | | __| |   ___  _ __   | |_| | __ _ _ __ _ __ _   _| |
|  _  |/ _ \| |/ _` |  / _ \| '_ \  |  _  |/ _` | '__| '__| | | | |
| | | | (_) | | (_| | | (_) | | | | | | | | (_| | |  | |  | |_| |_|
\_| |_/\___/|_|\__,_|  \___/|_| |_| \_| |_/\__,_|_|  |_|   \__, (_)
                                                            __/ |
                                                           |___/
        """  # noqa: W605
        )


def hagrid1() -> None:
    # relative
    from .lib import asset_path

    try:
        ascii_magic.to_terminal(
            ascii_magic.from_image_file(
                img_path=str(asset_path()) + "/img/hagrid.png", columns=83
            )
        )
    except Exception:  # nosec
        pass


def hagrid2() -> None:
    # relative
    from .lib import asset_path

    try:
        ascii_magic.to_terminal(
            ascii_magic.from_image_file(
                img_path=str(asset_path()) + "/img/hagrid2.png", columns=83
            )
        )
    except Exception:  # nosec
        pass


def quickstart_art() -> None:
    text = """
888    888        d8888  .d8888b.          d8b      888
888    888       d88888 d88P  Y88b         Y8P      888
888    888      d88P888 888    888                  888
8888888888     d88P 888 888        888d888 888  .d88888
888    888    d88P  888 888  88888 888P"   888 d88" 888
888    888   d88P   888 888    888 888     888 888  888
888    888  d8888888888 Y88b  d88P 888     888 Y88b 888
888    888 d88P     888  "Y8888P88 888     888  "Y88888


 .d88888b.           d8b          888               888                     888
d88P" "Y88b          Y8P          888               888                     888
888     888                       888               888                     888
888     888 888  888 888  .d8888b 888  888 .d8888b  888888  8888b.  888d888 888888
888     888 888  888 888 d88P"    888 .88P 88K      888        "88b 888P"   888
888 Y8b 888 888  888 888 888      888888K  "Y8888b. 888    .d888888 888     888
Y88b.Y8b88P Y88b 888 888 Y88b.    888 "88b      X88 Y88b.  888  888 888     Y88b.
 "Y888888"   "Y88888 888  "Y8888P 888  888  88888P'  "Y888 "Y888888 888      "Y888
       Y8b
"""
    console = rich.get_console()
    console.print(
        text,
        style="bold",
        justify="left",
        new_line_start=True,
    )


def hagrid() -> None:
    """Print a random hagrid image with the caption "hold on tight harry" """
    options = [motorcycle, hagrid1, hagrid2]
    i = secrets.randbelow(3)
    options[i]()
    hold_on_tight()


class RichEmoji(Emoji):
    def to_str(self) -> str:
        return self._char.encode("utf-8").decode(locale.getpreferredencoding())
