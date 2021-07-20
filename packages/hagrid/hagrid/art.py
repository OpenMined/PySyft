# stdlib
import os
import random

# third party
import ascii_magic

# relative
from .lib import asset_path


def motorcycle():
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
             `:dMMMMh.`"""
    )


def hold_on_tight():

    rows, columns = os.popen("stty size", "r").read().split()

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
            """
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
        """
        )


def hagrid1():
    ascii_magic.to_terminal(
        ascii_magic.from_image_file(
            img_path=str(asset_path()) + "/img/hagrid.png", columns=83
        )
    )


def hagrid2():
    ascii_magic.to_terminal(
        ascii_magic.from_image_file(
            img_path=str(asset_path()) + "/img/hagrid2.png", columns=83
        )
    )


options = [motorcycle, hagrid1, hagrid2]


def hagrid():
    """Print a random hagrid image with the caption "hold on tight harry" """
    i = random.randint(0, 2)

    options[i]()

    hold_on_tight()
