from sys import byteorder
import pyarrow as pa
from pyarrow.lib import tobytes
from pyarrow.util import pathlib, find_free_port
from pyarrow.tests import util

from pyarrow import flight
from pyarrow.flight import (
    FlightClient, FlightServerBase,
    ServerAuthHandler, ClientAuthHandler,
    ServerMiddleware, ServerMiddlewareFactory,
    ClientMiddleware, ClientMiddlewareFactory,
)

from ...core.common.uid import UID

class FlightServerDuet(FlightServerBase):
    def __init__(self, flight_args):
        location = '{}://{}:{}'.format(flight_args['scheme'], flight_args['host'], flight_args['port'])
        tls_certificates = []
        if flight_args['tls']:
            NotImplementedError
        verify_client = flight_args['verify_client']
        root_certificates = flight_args['root_certificates']
        auth_handler = flight_args['auth_handler']
        
        self.accessible = dict()

        super(FlightServerDuet, self).__init__(
            location, auth_handler, tls_certificates, verify_client,
            root_certificates)

    def do_exchange(self, context, descriptor, reader, writer):
        if descriptor.descriptor_type != flight.DescriptorType.CMD:
            raise pa.ArrowInvalid("Must provide a command descriptor")
        obj_id_str = descriptor.command.decode('utf-8')
        obj_id = UID.from_string(obj_id_str[3:])

        if obj_id_str[:3] == 'get':
            return self.do_exchange_write(obj_id, reader, writer, obj_id_str)
        elif obj_id_str[:3] == 'put':
            return self.do_exchange_read(obj_id, reader, writer)
        else:
            raise Exception

    def do_exchange_read(self, obj_id, reader, writer):
        data = reader.read_all()
        self.accessible[obj_id] = data

    def do_exchange_write(self, obj_id, reader, writer, obj_id_str):
        #TODO (flight): use appropriate arrow types
        data = pa.Table.from_arrays([
                pa.array(self.accessible[obj_id])
            ], names=[obj_id_str[3:]])
        writer.begin(data.schema)
        writer.write_table(data)

    def add_accessible(self, obj, id_at_location):
        self.accessible[id_at_location] = obj
    
    def retrieve_accessible(self, id_at_location):
        #TODO (flight): fix this mess (use appropriate arrow types)
        return self.accessible.get(id_at_location, None).to_pandas()[str(id_at_location.value)].to_numpy()