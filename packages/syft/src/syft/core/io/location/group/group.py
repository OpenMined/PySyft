# stdlib
from typing import Set

# relative
from ..location import Location


class LocationGroup(Location):

    """This represents a group of multiple locations. There
    are two kinds of LocationGroups, RegistryBackedLocationGroup
    and SubscriptionBackedLocationGroup. A registry backed group
    has an official owner with an official list while a subscription
    backed group is open to whoever knows the key to the group and
    chooses to subscribe. Both groups can have inclusion criteria
    which a location does or doesn't meet."""

    def __init__(self, known_group_members: Set[Location]):
        self.known_group_members = known_group_members
