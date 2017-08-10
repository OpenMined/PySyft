# Using PySyft via Docker

This directory contains `Dockerfile` to make it easy to get up and running with
PySyft via [Docker](http://www.docker.com/).

## Installing Docker

Follow the instructions from the official website:

* [Windows](https://www.docker.com/docker-windows)
* [Mac](https://www.docker.com/docker-mac)
* [Ubuntu](https://www.docker.com/docker-ubuntu)

## Running the container

We are using `Makefile` to simplify docker commands within make commands.

Build the container and start a jupyter notebook (accessible from http://HOSTNAME:8888/)

    $ make notebook

Build the container and start an iPython shell

    $ make ipython

Build the container and start a bash session

    $ make bash

Build the container for testing purposes

    $ make DOCKERFILE=Dockerfile.test bash # start a bash session in test mode
    $ make DOCKERFILE=Dockerfile.test test # executes tests in test mode

Prints all make tasks

    $ make help
