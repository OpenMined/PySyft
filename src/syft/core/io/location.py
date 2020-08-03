from typing import Set

from ..common.object import ObjectWithID


class Location(ObjectWithID):
    """This represents the location of a node, including
    location-relevant metadata (such as how long it takes
    for us to communicate with this location, etc.)"""

    def __init__(self):
        super().__init__()


class LocationGroup(Location):

    """This represents a group of multiple locations. There
    are two kinds of LocationGroups, RegistryBackedLocationGroup
    and SubscriptionBackedLocationGroup. A registry backed group
    has an official owner with an official list while a subscription
    backed group is open to whomever knows the key to the group and
    chooses to subscribe. Both groups can have inclusion criteria
    which a location does or doesn't meet."""

    def __init__(self, known_group_members: Set[Location]):
        self.known_group_members = known_group_members


class SubscriptionBackedLocationGroup(LocationGroup):
    def __init__(self, topic: str, known_group_members: Set[Location]):
        super().__init__(known_group_members=known_group_members)
        self.topic = topic


class RegistryBackedLocationGroup(LocationGroup):

    """This is a location group where membership of the group is
    determined by an official owner who has an official list. The
    best and most practical example of this is a Node and its group
    of children (or grand children, etc.). Sending messages to this
    group requires sending a old_message to its owner and asking it
    to redistribute to the group. However, we can still maintain a list
    of workers which we believe to be in the group which allows us
    to logically reason about the group itself."""

    def __init__(self, group_owner: Location, known_group_members: Set[Location]):
        super().__init__(known_group_members=known_group_members)
        self.group_owner = group_owner
