from ...logger import error
from ...logger import info
from .compressor import sync_compression
from .compressor import get_connection_speed

class CompressionParams:
    def __init__(self) -> None:
        self._bytes = ListenerDict({
            'compress': True,
            'lib': 'lzma',
            'cname': 'zlib',
            'compression_lvl': 8,
        })
        self._tensor = ListenerDict({
            'compress': True,
            'compressors': ['DgcCompressor'],
        })
        self._dgc_compressor = ListenerDict({
            'ratio': 0.8
        })
        self._deep_reduce = ListenerDict({
            'compress_ratio': 0.5, 
            'deepreduce':'index', 
            'index':'bloom',
        })
        self._connection = ListenerDict({
            'speed': 0.0,
            'tested': False,
        })

    def process_value(self, value) -> "ListenerDict":
        if isinstance(value, dict):
            value = ListenerDict(value)
            return value
        error('TypeError: Expected dict/ListenerDict but received', type(value))

    def __copy__(self):
        duplicate = type(self)()
        duplicate._bytes = ListenerDict(self.bytes)
        duplicate._tensor = ListenerDict(self.tensor)
        duplicate._dgc_compressor = ListenerDict(self.dgc_compressor)
        duplicate._deep_reduce = ListenerDict(self.deep_reduce)
        duplicate._connection = ListenerDict(self.connection)
        if hasattr(self, 'client'):
            duplicate._client = self._client
        return duplicate

    @property
    def bytes(self):
        return self._bytes

    @bytes.setter
    def bytes(self, value):
        updated = self.__copy__()
        value = self.process_value(value)
        updated._bytes = value
        success = sync_compression(updated)
        if success:
            self._bytes = value

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        updated = self.__copy__()
        value = self.process_value(value)
        updated._tensor = value
        success = sync_compression(updated)
        if success:
            self._tensor = value

    @property
    def dgc_compressor(self):
        return self._dgc_compressor

    @dgc_compressor.setter
    def dgc_compressor(self, value):
        updated = self.__copy__()
        value = self.process_value(value)
        updated._dgc_compressor = value
        success = sync_compression(updated)
        if success:
            self._dgc_compressor = value

    @property
    def deep_reduce(self):
        return self._deep_reduce

    @deep_reduce.setter
    def deep_reduce(self, value):
        updated = self.__copy__()
        value = self.process_value(value)
        updated._deep_reduce = value
        success = sync_compression(updated)
        if success:
            self._deep_reduce = value

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, value):
        updated = self.__copy__()
        value = self.process_value(value)
        updated._connection = value
        success = sync_compression(updated)
        if success:
            self._connection = value

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value):
        speed = get_connection_speed(self, value)
        self.set_params_by_speed(speed)
        success = sync_compression(self, value)
        if success:
            info('Automatically set compression params using connection speed.')
            self._client = value

    def set_params_by_speed(self, speed):
        self._connection = ListenerDict({
            'speed': speed,
            'tested': True,
        })
        pass

    def fix_dict_listener(self):
        self._bytes = ListenerDict(self.bytes)
        self._tensor = ListenerDict(self.tensor)
        self._dgc_compressor = ListenerDict(self.dgc_compressor)
        self._deep_reduce = ListenerDict(self.deep_reduce)
        self._connection = ListenerDict(self.connection)

class ListenerDict(dict):
    def __setitem__(self, item, value):
        try:
            curr_value = dict.__getitem__(self, item)
        except:
            curr_value = None
        dict.__setitem__(self, item, value)
        updated = compression_params.__copy__()
        if curr_value is not None:
            dict.__setitem__(self, item, curr_value)
        else:
            dict.pop(self, item, None)
        success = sync_compression(updated)
        if success:
            dict.__setitem__(self, item, value)

compression_params = CompressionParams()
