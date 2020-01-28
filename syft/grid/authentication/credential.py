from abc import ABC, abstractmethod


class AbstractCredential(ABC):
    """ AbstractCredential is an abstract class that defines generic methods
        used by all types of authentications defined in this module. """

    @abstractmethod
    def parse(self):
        """ Read, parse and load credential files."""
        raise NotImplementedError("Parse not specified!")

    @abstractmethod
    def json(self):
        """ Convert credential instances into a JSON structure. """
        raise NotImplementedError("JSON not specified!")
