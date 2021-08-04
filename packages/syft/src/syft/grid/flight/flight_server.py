from sys import byteorder
import pyarrow as pa
from pyarrow.lib import tobytes
from pyarrow.util import pathlib, find_free_port
from pyarrow.tests import util
import torch
from pyarrow import flight
from pyarrow.flight import (
    FlightClient, FlightServerBase,
    ServerAuthHandler, ClientAuthHandler,
    ServerMiddleware, ServerMiddlewareFactory,
    ClientMiddleware, ClientMiddlewareFactory,
)

from ...core.common.uid import UID

class FlightServerDuet(FlightServerBase):
    def __init__(self, flight_args, node):
        location = '{}://{}:{}'.format(flight_args['scheme'], flight_args['host'], flight_args['port'])
        location = flight_args['location']
        tls_certificates = []
        if flight_args['tls']:
            NotImplementedError
        verify_client = flight_args['verify_client']
        root_certificates = flight_args['root_certificates']
        auth_handler = flight_args['auth_handler']
        
        self.accessible = dict()
        self.node = node
        self.node.flight_server = self

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
        elif obj_id_str[:3] == 'dgt':
            return self.do_exchange_dim_write(obj_id, reader, writer, obj_id_str)
        elif obj_id_str[:3] == 'dpt':
            return self.do_exchange_dim_read(obj_id, reader, writer)
        else:
            raise Exception

    def do_exchange_read(self, obj_id, reader, writer):
        data = reader.read_all()
        # print(type(data))
        self.accessible[obj_id] = data[str(obj_id.value)]

    def do_exchange_write(self, obj_id, reader, writer, obj_id_str):
        data = pa.RecordBatch.from_arrays([
                self.accessible[obj_id]
            ], names=[obj_id_str[3:]])
        writer.begin(data.schema)
        writer.write_batch(data)
        writer.close()

    def do_exchange_dim_write(self, obj_id, reader, writer, obj_id_str):
        data = pa.RecordBatch.from_arrays([
                self.accessible['dim'+str(obj_id.value)]
            ], names=[obj_id_str[3:]])
        writer.begin(data.schema)
        writer.write_batch(data)
        writer.close()

    def do_exchange_dim_read(self, obj_id, reader, writer):
        data = reader.read_all()
        self.accessible['dim'+str(obj_id.value)] = data[str(obj_id.value)]

    def add_accessible(self, obj, id_at_location):
        self.accessible[id_at_location] = pa.array(obj.numpy().reshape(-1))
        self.accessible['dim'+str(id_at_location.value)] = pa.array(obj.numpy().shape)
    
    def retrieve_accessible(self, id_at_location):
        raw_data = self.accessible.get(id_at_location, None)
        if raw_data is not None:
            obj_dim = self.accessible.get('dim'+str(id_at_location.value), None)
            if obj_dim is not None:
                return torch.from_numpy(raw_data.to_numpy().reshape(obj_dim.to_numpy()))
            return torch.from_numpy(raw_data.to_numpy())
        return raw_data