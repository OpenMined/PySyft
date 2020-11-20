# Aries StagingNet Example

This example takes you through the process of moving from a local development environment into a more realistic implementation of SSI with credential schema and definitions written to the public Sovrin StagingNet.

In this example, **you** will play the role of holder using a mobile agent of your choice. The notebooks represent an issuer and a verifier, the example takes you through issuing a credential to your wallet using the issuer notebook and then presenting attributes from this credential to the verifier on a different notebook. This begins to show the power of this architecture, the individual becomes the point of interoperability between systems for data about them.

## Notebooks:

* [Issuer](http://localhost:8888) - run `docker logs aries-stagingnet_issuer-notebook_1` for the token.
* [Verifier](http://localhost:8889) - run `docker logs aries-stagingnet_verifier-notebook_1` for the token.
