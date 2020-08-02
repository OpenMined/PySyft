"""Route is an abstraction between Connection and Client.
A Client represents an API for interfacing with a Node of
a particular type at a specific location. We assume that
a Client may have multiple ways of interacting with a node
(such as multiple protocols, http, websockets, broadcast server
, etc.). We assume that a Client also has knowledge of the
full variety of messages and old_message types which can be sent
to a Node. It is the highest level of abstraction.

A connection is the lowest level of abstraction, representing
a real network interface to a single destination. Importantly,
this destination may not even be the ultimate destination
for a old_message. In this sense, a connection is just a single "hop"
in a potentially long series of hops between a Client and
a Node.

However, it's important to note that a Client doesn't need to
have direct access to the node it communicates with because
Nodes have a variety of services for forwarding messages
around. For example, if I have two clients, one pointing
to a Domain node and another pointing to a Device within
the Domain, both might use the same connection object
because all messages to a device have to go through a domain.
This is because, for permissions reasons, the domain represents
an intentional information bottleneck.

This might beg the question, what is the point of Route if
the right combination of Clients and Connnections can always
get a old_message to where it needs to go?

The answer is that neither a Connection, Client, or Node are
appropriate for the nature of four combined constraints.

1) the existence of multiple connections from client -> node
with varying tradeoffs (websocket, http, webRTC, gRPC, etc.)
2) the existence of multiple clients which share the same
connection because of intentional information bottlenecks.
3) the existence of 1->many connection types (pub-sub)
4) the varying cost of different networks (edge cellular, and
cloud networking costs being the biggest)

We need the ability for an arbitrary client (which represents
an abstract, indirect connection to a specific remote node of
a specific type) to be able to consider a old_message, it's address
, and other metadata, and make a logical decision on which connection
should be used. (For example, some messages block for a reply)

We need the connection decision to take into account the latency,
connection health, and overall cost of each connection option as
measured by fixed constants as well as previous uses of the
connection.

This necessitates an abstraction which lets us store connection
information both about hops we know about and about hops we don't
know about.

This abstraction is called a "Route" and each hop is called a "Hop"
where any collection of hops within the route is called a 
"RouteSegment". Note that a Route need only be initialized with 
it's source and destination(s). 

When a route is used by a client, it is used to decide which route
would be best to take for a particular old_message.

When a route is attached to a old_message, it implies the expected
journey the old_message is to take to the address. A node may or may
not actually respect this journey.

Routes can be created locally, but they are best created by asking
the network for a good route to a remote worker which allows for
things like network speed to be interrogated in the creation of
a route.

While in theory this data-structure could allow for very sophisticated
network analysis, in the beginning it will mostly exist to choose
between Pub-sub, Request-Reponse, and Streaming options. In the
long run it will allow us to design routes which explicitly
avoid the most costly network bottlenecks. For example, if Grid is
advertising a new Federated Learning job, instead of sending the
model directly to 10,000 clients (lots of bandwidth costs on the
cloud node), it could send it to one cellphone with WiFi which
would have instructions for how to forward to a binary tree of other
node, propagating the model to all node which asked for it.
"""

from ..common.object import ObjectWithID
from .location import Location
from typing import Set
from syft.core.common.message import SyftMessageWithReply
from syft.core.common.message import SyftMessageWithoutReply
from syft.core.common.message import EventualSyftMessageWithoutReply
from .connection import ClientConnection
from typing import List


class RouteSchema(ObjectWithID):
    """An object which contains the IDs of the origin node and
    set of destination node. Multiple routes can subscribe
    to the same RouteSchema and routing logic is thus split into
    two groups of functionality:

    1) Discovering new routes
    2) Comparing known routes to find the best one for a old_message
    """

    def __init__(self, source: Location, destination: Location):
        self.source = source
        self.destination = destination


class Route(ObjectWithID):
    def __init__(self, schema: RouteSchema, stops: List[Location] = list()):
        self.schema = schema
        self.stops = stops

    def send_immediate_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:
        raise NotImplementedError


class BroadcastRoute(Route):
    def send_immediate_msg_with_reply(
        self, msg: SyftMessageWithReply
    ) -> Set[SyftMessageWithoutReply]:
        raise NotImplementedError


class SoloRoute(Route):
    def __init__(
        self, source: Location, destination: Location, connection: ClientConnection
    ):
        self.schema = RouteSchema(source=source, destination=destination)
        self.connection = connection

    def send_immediate_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:
        self.connection.send_immediate_msg_without_reply(msg=msg)

    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        self.connection.send_eventual_msg_without_reply(msg=msg)

    def send_immediate_msg_with_reply(
        self, msg: SyftMessageWithReply
    ) -> SyftMessageWithoutReply:
        return self.connection.send_immediate_msg_with_reply(msg=msg)
