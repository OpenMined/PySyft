# # stdlib
# from typing import List
# from typing import Optional
#
# # third party
# import numpy as np
# import pyarrow as pa
#
# # relative
# from ..common.serde.serializable import bind_protobuf
# from .schema import Schema
#
#
# @bind_protobuf
# class Dataset:
#     def __init__(self, schema: Schema):
#         self.schema = schema
#         self._table = None
#
#     def add_pandas(data) -> "Dataset":
#         pass
#
#     def add_numpy(numpy_array: np.ndarray) -> "Dataset":
#         pass
#
#     def add_table(self, table: pa.Table):
#         if self._table is None:
#             self._table = table
#         else:
#             self._table = pa.concat_tables([self._table, table])
#
#     def add_pylist(self, python_list) -> None:
#         new_record_batch = pa.record_batch(python_list, **self.schema.get_constraints())
#         table = pa.table(python_list, **self.schema.get_constraints())
#         self.add_table(table)
#
#     def _object2proto(self):
#         pass
#
#     def _proto2object(self):
#         pass
#
#     @staticmethod
#     def get_protobuf_schema():
#         return type("Myprotobuf", tuple(), {})
