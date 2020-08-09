from .location import Location
from ...common.object import ObjectWithID


class SpecificLocation(Location, ObjectWithID):
    """This represents the location of a single Node object
    represented by a single UID. It may not have any functionality
    beyond Location but there is logic which interprets it differently."""

    def __init__(self):
        super().__init__()
