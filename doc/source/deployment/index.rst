Deploy with Hagrid
==================

`Hagrid` (HAppy GRID!) is a command-line tool that speeds up the deployment of PyGrid, the software providing a peer-to-peer network of data owners and data scientists who can collectively train AI models using PySyft.
Hagrid is able to orchestrate a collection of PyGrid Domain and Network nodes and scale them in a local development environment (based on a docker-compose file). By stacking multiple copies of this docker, you can simulate multiple entities (e.g countries) that collaborate over data and experiment with more complicated data flows such as SMPC.
Similarly to the local deployment, Hagrid can bootstrap docker on a Vagrant VM or on a cloud VM, helping you deploy in an user-friendly way on Azure, AWS* and GCP*.


