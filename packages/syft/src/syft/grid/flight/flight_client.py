import pyarrow as pa
import torch
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

class FlightClientDuet(FlightClient):
    def __init__(self, *args, **kwargs):
        super(FlightClientDuet, self).__init__(*args, **kwargs)

    def get_object(self, obj_id):
        obj = self.get_object_action(obj_id)
        obj_dim = self.get_object_dim(obj_id).to_numpy()
        return torch.from_numpy(obj.reshape(obj_dim))

    def put_object(self, obj_id, obj):
        self.put_object_action(obj_id, obj)
        self.put_object_dim(obj_id, obj)

    def get_object_action(self, obj_id):
        descriptor = flight.FlightDescriptor.for_command(str('get' + str(obj_id.value)).encode('utf-8'))
        writer, reader = super().do_exchange(descriptor)
        data = reader.read_all()
        return data.to_pandas()[str(obj_id.value)].to_numpy()

    def put_object_action(self, obj_id, obj):
        obj_id_str = str('put' + str(obj_id.value))
        descriptor = flight.FlightDescriptor.for_command(obj_id_str.encode('utf-8'))
        print("PUT OBJECT", type(obj))

        data = pa.RecordBatch.from_arrays([
                pa.array(obj.numpy().reshape(-1))
            ], names=[obj_id_str[3:]])

        writer, _ = super().do_exchange(descriptor)
        writer.begin(data.schema)
        writer.write_batch(data)
        writer.close()

    def get_object_dim(self, obj_id):
        descriptor = flight.FlightDescriptor.for_command(str('dgt' + str(obj_id.value)).encode('utf-8'))
        writer, reader = super().do_exchange(descriptor)
        data = reader.read_all()
        return data[str(obj_id.value)]

    def put_object_dim(self, obj_id, obj):
        obj_id_str = str('dpt' + str(obj_id.value))
        descriptor = flight.FlightDescriptor.for_command(obj_id_str.encode('utf-8'))

        data = pa.RecordBatch.from_arrays([
                pa.array(obj.numpy().shape)
            ], names=[obj_id_str[3:]])

        writer, _ = super().do_exchange(descriptor)
        writer.begin(data.schema)
        writer.write_batch(data)
        writer.close()