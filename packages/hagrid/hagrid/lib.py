import os
import subprocess


def some_function():
    print("Running lib.some_function")


def check_docker():
    result = os.popen("docker compose version", "r").read()

    if "version" in result:
        version = result.split()[-1]
    else:
        print("This may be a linux machine, either that or docker compose isn't s")
        print("Result:" + result)
        out = subprocess.run(["docker", "compose"], capture_output=True, text=True)
        if "'compose' is not a docker command" in out.stderr:
            raise Exception(
                """You are running an old verion of docker, possibly on Linux. You need to install v2 beta.
                Instructions for v2 beta can be found here:

                https://www.rockyourcode.com/how-to-install-docker-compose-v2-on-linux-2021/

                At the time of writing this, if you are on linux you need to run the following:

                mkdir -p ~/.docker/cli-plugins
                curl -sSL https://github.com/docker/compose-cli/releases/download/v2.0.0-beta.5/docker-compose-linux-amd64 -o ~/.docker/cli-plugins/docker-compose
                chmod +x ~/.docker/cli-plugins/docker-compose

                ALERT: you may need to run the following command to make sure you can run without sudo.

                echo $USER              //(should return your username)
                sudo usermod -aG docker $USER

                ... now LOG ALL THE WAY OUT!!!

                ...and then you should be good to go. You can check your installation by running:

                docker compose version
                """
            )

    return version


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
