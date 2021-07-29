# hagrid

Use this cli to deploy PyGrid Domain and Network nodes on your local machine.

A Hagrid is a HAppy GRID!

## Installation Linux and MacOS

Python

```
$ pip install hagrid
```

Docker

```
$ docker run -it -v ~/:/root openmined/hagrid:latest hagrid
```

Then simply run hagrid as you would normally:

```
$ docker run -it -v ~/:/root openmined/hagrid:latest hagrid launch slytherin to azure
```

## Installation Windows

Install Docker Desktop: https://www.docker.com/products/docker-desktop

```powershell
PS docker run -it -v "$($env:USERPROFILE):/root" openmined/hagrid:latest hagrid
```

Then simply run hagrid as you would normally:

```powershell
PS docker run -it -v "$($env:USERPROFILE):/root" openmined/hagrid:latest hagrid launch slytherin to azure
```

## Development

#### Step 1 Dev Setup

If you want hagrid to launch nodes based on your live-updating codebase, then install it using one of the live-updating install commands. This will mean that the codebase will hot-reload based on the current project.

```bash
pip install -e .
```

## Launch a Node

![alt text](cli2.png)

## A Few Example Commands

Start a node with:

```bash
hagrid launch slytherin
```

... and then stop it with:

```bash
hagrid land slytherin
```

You can specify ports if you want to:

```bash
hagrid launch hufflepuff_house to docker:8081+
```

... but if you don't it'll find an open one for you

```bash
// finds hufflepuff already has 8081... tries 8082
hagrid launch ravenclaw
```

You can also specify the node type (domain by default)

```bash
hagrid launch gryffendor network to docker
```

## Credits

**Super Cool Code Images** by [Carbon](https://carbon.now.sh/)
