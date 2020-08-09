from .group import LocationGroup
from ..location import Location
from typing import Set


class SubscriptionBackedLocationGroup(LocationGroup):
    def __init__(self, topic: str, known_group_members: Set[Location]):
        super().__init__(known_group_members=known_group_members)
        self.topic = topic
