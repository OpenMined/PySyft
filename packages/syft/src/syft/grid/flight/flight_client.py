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

class FlightClientDuet(FlightClient):
    def __init__(self, *args, **kwargs):
        super(FlightClientDuet, self).__init__(*args, **kwargs)

    def get_object(self, obj_id):
        descriptor = flight.FlightDescriptor.for_command(str('get' + str(obj_id.value)).encode('utf-8'))
        writer, reader = super().do_exchange(descriptor)
        return reader.read_all()

    def put_object(self, obj_id, obj):
        obj_id_str = str('put' + str(obj_id.value))
        descriptor = flight.FlightDescriptor.for_command(obj_id_str.encode('utf-8'))
        writer, reader = super().do_exchange(descriptor)

        #TODO (flight): use appropriate arrow types
        data = pa.Table.from_arrays([
                pa.array(obj)
            ], names=[obj_id_str[3:]])
        writer.begin(data.schema)
        writer.write_table(data)