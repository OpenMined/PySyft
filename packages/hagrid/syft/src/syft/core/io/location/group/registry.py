# stdlib
from typing import Set

# relative
from ..location import Location
from .group import LocationGroup


class RegistryBackedLocationGroup(LocationGroup):

    """This is a location group where members of the group is
    determined by an official owner who has an official list. The
    best and most practical example of this is a Node and its group
    of children (or grandchildren, etc.). Sending messages to this
    group requires sending a message to its owner and asking it
    to redistribute to the group. However, we can still maintain a list
    of workers which we believe to be in the group which allows us
    to logically reason about the group itself."""

    def __init__(self, group_owner: Location, known_group_members: Set[Location]):
        super().__init__(known_group_members=known_group_members)
        self.group_owner = group_owner
