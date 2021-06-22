# Hyperledger Aries Credential Exchange

## Using Verifiable Credentials to add Authentication to a Duet Session

This example was developed as part of the
[Foundations in Private Computation: Public Key Infrastructures](https://github.com/OpenMined/PyDentity/tree/master/tutorials/5.%20OM%20FoPC%20Course%20-%20Public%20Key%20Infrastructures) course.

## Actors

- **OpenMined Duet Authority**
  Will connect with and issue the Data Owner and Data Scientist the respective credentials

- **Data Scientist**
  A Scientist first applies to become a Duet Data Scientist. Then uses this credential to establish a Duet session with a Data Owner

- **Data Owner**
  A person with data they wish to allow computation on, without the Data Scientist seeing it

## Requirements

- Install Docker
- Install docker-compose
- Install Source2Image

## Configuration

- First setup the .env files for each actor
- Customise the files if you wish, but they are meant to work as is
- Note: playground/actor is not used and is included in case you want to extend this example

### Authority

```
$ cd playground/om-authority
$ cp om-authority-example.env .env
$ cd -
```

### DS (Data Scientist)

```
$ cd playground/data-scientist
$ cp data-scientist-example.env .env
$ cd -
```

### DO (Data Owner)

```
$ cd playground/data-owner
$ cp data-owner-example.env .env
$ cd -
```

## Run

_From inside `/packages/syft/examples/hyperledger-aries`_

- Spin up the docker containers in a terminal

```bash
./manage.sh start
```

## Open Notebooks

To get the urls to the notebooks inside the containers run:

```bash
./scripts/get-urls.sh
```

This should print 3 urls to a jupyter lab session for each actor.

Work through the notebook code.
You should start with the OpenMined Duet Authority notebooks (port 8890).

**NOTE:** Don't forget to stop the containers when you are done.

Top stop the containers run:

```bash
$ ./manage.sh stop
```

## Running Remotely

If you are struggling to run this on your machine (mostly we have had issues with people
installing docker and docker-compose) we recommend you run this on a virtual machine.
The above steps should be the same, accept you need to replace localhost with the ip
address of your virtual machine when it comes to accessing the jupyter notebooks.

## Using a deployed Full Stack OM Duet Authority Application

Hosted OM Duet Authority - http://139.162.224.50
Repo: https://github.com/wip-abramson/fpc-om-authority-aries-application

This example has been designed to work with a full stack application replacing the
OM Duet Authority notebooks. This application may or may not be running when you come
to work through this example.

## Designing You Own Credential System

After working through this example you might have ideas for you own credential system for a specific context/domain you are knowledgable about. With a set of actors and roles and a purpose where digital credentials can be justified and bring real value.

If so great!

We have designed a playground that should provide you with the perfect starting place when attempting to design and experiment which what such a system might look like.

https://github.com/wip-abramson/aries-jupyter-playground
