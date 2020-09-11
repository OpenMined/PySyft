"""This represents the communication "edge" connecting two
Locations in a route. """

# syft relative
from .location import Location


class Hop:
    def __init__(self, a: Location, b: Location):
        self.a = a
        self.b = b
