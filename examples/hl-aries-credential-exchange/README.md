# Hyperledger Aries Credential Exchange

## Using Verifiable Credentials to add Authentication to a Duet Session

#### This example was developed as part of the [Foundations in Private Computation: Public Key Infrastructures](https://github.com/OpenMined/PyDentity/tree/master/tutorials/5.%20OM%20FoPC%20Course%20-%20Public%20Key%20Infrastructures) course.

## Actors

* OpenMined Duet Authority - Will connect with and issue the Data Owner and Data Scientist the respective credentials
* Data Scientist - A Scientist first applies for to become a Duet Data Scientist. Then uses this credential to establish a Duet session with a Data Owner
* Data Owner - A Data 


## Requirements

* Install Docker
* Install docker-compose
* Install Source2Image

## Configuration

* First setup the .env files for each actor. E.g.
    * cd playground/data-owner
    * cp data-owner-example.env .env
* Customise the files if you wish, but they are meant to work as is
* Note: playground/actor is not used and is included in case you want to extend this example

## Run

*Taking the hl-aries-credential-exchange as the root folder*

* Spin up the docker containers in a terminal
```bash
./manage start
```
* Get Notebook urls
```bash
./scripts/get_URLS.sh
```

This should print 3 urls to a jupyter lab session for each actor.

Work through the notebook code. You should start with the OpenMined Duet Authority notebooks (port 8890)

* Stop containers when finished

`./manage stop`

## Running Remotely

If you are struggling to run this on your machine (mostly we have had issues with people installing docker and docker-compose) we recommend you run this on a virtual machine. The above steps should be the same, accept you need to replace localhost with the ip address of your virtual machine when it comes to accessing the jupyter notebooks.

## Using a Deployed Full Stack OM Duet Authority Application

TODO: This still needs to be deployed
Repo: https://github.com/wip-abramson/fpc-om-authority-aries-application

This example has been designed to work with a full stack application replacing the OM Duet Authority notebooks. This application may or may not be running when you come to work through this example. 


    