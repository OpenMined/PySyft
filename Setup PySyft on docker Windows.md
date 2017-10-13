# Setup PySyft on Docker Windows
Using the pre-assembled Docker image is the fastest way to get started with PySyft. However, setting up process is different in Windows:
- Most windows (win 7, 8, 10 home edition) do not support docker directly. We have to use Docker toolbox on Windows to work around.
- **_make_** command is not recognized on windows. Therefore, we cannot run _make docker-build_ or _make docker-run_. 

Following guide is made to assist setting up PySyft on docker windows.

[//]: # (Image References)
[desktop]: ./assets/desktop.PNG
[docker_images]: ./assets/docker_images.PNG
[docker_start]: ./assets/docker_start.PNG

## Install docker on windows
if you already installed **docker toolbox on windows** skip this installation part.
1. Download and install the [Docker toolbox on windows](https://docs.docker.com/toolbox/toolbox_install_windows/). You should see following 3 components if the installation is successful.

      ![alt text][desktop]

2. start _docker quickstart terminal_ and run following command:
```bash
docker -v
```
if docker version is displayed, you are good to go. Please keep the IP address in your mind. We will use it when we use jupyter notebook in the browser.

   ![alt text][docker_start]


## Setup PySyft on the docker windows
Following steps are executed on the _docker quickstart terminal_.

1. Get the newest PySyft repo and go into the repo.
```bash
    git clone https://github.com/OpenMined/PySyft.git
    cd PySyft
```
2. Get the base image by following command (DO NOT Forget the DOT in the end).
```bash
    docker build -f dockerfiles/Dockerfile.base -t pysyft-base:local .
```
3. Run following command to create PySyft image.
```bash
    docker build -f dockerfiles/Dockerfile -t openmined/pysyft:local .
```
4. Run following command to create PySyft dev image (includes Capsule installation).
```bash
    docker build -f dockerfiles/Dockerfile.dev -t openmined/pysyft-dev:local .
```
5. Check if images are installed by following command
```bash
    docker images
```
if you see 3 images listed as bellow, you are good to go to next step. 

   ![alt text][docker_images]

6. Run PySyft-dev image and SSH into the container with following command:
```bash
    docker run -it --rm -v "$(PWD)":/PySyft -p 8888:8888 openmined/pysyft-dev:local sh
```
7. In the container, start the Capsule and run the jupyter notebook.
```bash
     redis-server & FLASK_APP=/usr/bin/Capsule/capsule/local_server.py flask run & cd notebooks && jupyter notebook --allow-root --ip=0.0.0.0
```
you should see the URL for the notebook: http://0.0.0.0:8888/?token=xxxxxxxxxxxxxxxxxxxxx.  Now you need to replace the IP - 0.0.0.0 with docker-machine’s IP address (the one we see when starting the _docker quickstart terminal_).

8. Start the browser and input above URL. You should be able to run the notebook. 
