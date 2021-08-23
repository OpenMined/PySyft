# stdlib
from typing import Set

# relative
from ..location import Location
from .group import LocationGroup


class SubscriptionBackedLocationGroup(LocationGroup):
    def __init__(self, topic: str, known_group_members: Set[Location]):
        super().__init__(known_group_members=known_group_members)
        self.topic = topic
