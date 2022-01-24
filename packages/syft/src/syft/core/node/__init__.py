"""Syft's node submodule is responsible for both the interface and implementations of
different node types. The structure of the submodule is:

* abstract: the interfaces that need to be implemented by a node.

* common: services, database handlers or managers commonly used by nodes.

* device: An experimental type of node to identify resources on a vm.

* domain: A domain node can be identified as a user, the node being able to do
    computations or to interact with other domains or networks.

* network: A network node is responsible on connecting multiple domains together, making
    already connected domains discoverable by new domains.

* vm: An experimental type of node used to identify a compute unit in a domain.
"""
