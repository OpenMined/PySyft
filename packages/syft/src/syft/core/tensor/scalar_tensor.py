# # future
# from __future__ import annotations
#
# # stdlib
# from typing import List as TypeList
# from typing import Union
#
# # third party
# import names
# import numpy as np
#
# # relative
# from ..adp.entity import Entity
# from ..adp.publish import publish
# from ..adp.scalar import PhiScalar
# from .autodp.single_entity_phi import SingleEntityPhiTensor
# from .autodp.row_entity_phi import RowEntityPhiTensor
#
#
#
# def make_entities(n: int = 100) -> TypeList[Entity]:
#     ents: TypeList[Entity] = list()
#     for i in range(n):
#         ents.append(Entity(name=names.get_full_name().replace(" ", "_")))
#     return ents
#
#
# def private(
#     input_data,
#     min_val,
#     max_val,
#     entities=None,
#     one_entity_per_row=True,
#     is_discrete=False,
# ) -> Union[SingleEntityPhiTensor, RowEntityPhiTensor, ]:
#     self = input_data
#
#     if one_entity_per_row and entities is None:
#         entities = make_entities(n=len(input_data))
#
#     if entities is None:
#         flat_data = self.flatten()
#         entities = make_entities(n=len(flat_data))
#
#         scalars = list()
#         for i in flat_data:
#             value = max(min(float(i), max_val), min_val)
#             s = PhiScalar(
#                 value=value,
#                 min_val=min_val,
#                 max_val=max_val,
#                 entity=entities[len(scalars)],
#                 #                 is_discrete=is_discrete
#             )
#             scalars.append(s)
#
#         return to_values(np.array(scalars)).reshape(input_data.shape)
#
#     elif isinstance(entities, list):
#         if len(entities) == len(self):
#             output_rows = list()
#             for row_i, row in enumerate(self):
#                 row_of_entries = list()
#                 for item in row.flatten():
#                     s = PhiScalar(
#                         value=item,
#                         min_val=min_val,
#                         max_val=max_val,
#                         entity=entities[row_i],
#                         #                         is_discrete=is_discrete
#                     )
#                     row_of_entries.append(s)
#                 output_rows.append(np.array(row_of_entries).reshape(row.shape))
#             return to_values(np.array(output_rows)).reshape(self.shape)
#         else:
#             print(len(entities))
#             print(len(self))
#             raise Exception("len(entities) must equal len(self)")
#
#
# class GradLedger:
#     def __init__(self):
#         self.grad_parents = list()
#         self.total_symbols = set()
#
#     def add(self, dict_of_gradient_parents, accumulate_grads=False):
#
#         self.grad_parents.append(dict_of_gradient_parents)
#
#         # if accumulate grads is false, raise exception if any of the results
#         # set gradients for the same variables (which should have been accumulated)
#
#         for symbol_id in dict_of_gradient_parents.keys():
#             if not accumulate_grads:
#                 if symbol_id in self.total_symbols:
#                     exception_msg = (  # nosec
#                         "You had multiple gradients attempting to update the same"
#                         + "scalar but accumulate_grads was set to False. This means that"
#                         + "some grads were overwritten by new ones and these gradients"
#                         + "would be false. Please re-run the computation with"
#                         + "accumulate_grads set to True and don't forget to zero grads"
#                         + "out each time you're done."
#                     )
#                     raise Exception(exception_msg)
#             self.total_symbols.add(symbol_id)
#
#     def zero_grads(self):
#         for x in self.total_symbols:
#             x._grad = None
#
#
# class ScalarTensor(np.ndarray):
#     def __new__(
#         cls,
#         input_array,
#         min_val=None,
#         max_val=None,
#         entities=None,
#         info=None,
#         is_discrete=False,
#     ):
#
#         is_private = False
#
#         if min_val is not None and max_val is not None:
#             input_array = private(
#                 input_array,
#                 min_val=min_val,
#                 max_val=max_val,
#                 entities=entities,
#                 is_discrete=is_discrete,
#             )
#             is_private = True
#         else:
#             input_array = to_values(input_array)
#
#         obj = np.asarray(input_array).view(cls)
#         obj.info = info
#         obj.is_private = is_private
#
#         return obj
#
#     def __array_finalize__(self, obj):
#         if obj is None:
#             return
#         self.info = getattr(obj, "info", None)
#         self.is_private = getattr(obj, "is_private", None)
#
#     def __array_wrap__(self, out_arr, context=None):
#
#         output = out_arr.view(ScalarTensor)
#
#         is_private = False
#         if context is not None:
#             for arg in context[1]:
#                 if hasattr(arg, "is_private") and arg.is_private:
#                     is_private = True
#
#         output.is_private = is_private
#
#         return output
#
#     def backward(self, accumulate_grads=False) -> GradLedger:
#         ledger = GradLedger()
#         for entry in self.flatten():
#             ledger.add(
#                 grad(entry, accumulate=accumulate_grads),
#                 accumulate_grads=accumulate_grads,
#             )
#         return ledger
#
#     @property
#     def grad(self):
#         grads = list()
#         for val in self.flatten().tolist():
#             grads.append(val._grad)
#         return ScalarTensor(grads).reshape(self.shape)
#
#     def slow_publish(self, **kwargs):
#         grads = list()
#         for val in self.flatten().tolist():
#             grads.append(val.value.publish(**kwargs))
#         return np.array(grads).reshape(self.shape)
#
#     def publish(self, **kwargs):
#         grads = list()
#         for val in self.flatten().tolist():
#             grads.append(val.value)
#         grads = publish(scalars=grads, **kwargs)
#         return np.array(grads).reshape(self.shape)
#
#     @property
#     def value(self):
#         values = list()
#         for val in self.flatten().tolist():
#             if hasattr(val.value, "value"):
#                 values.append(val.value.value)
#             else:
#                 values.append(val.value)
#         return np.array(values).reshape(self.shape)
#
#     def private(self, min_val, max_val, entities=None, is_discrete=False):
#         if self.is_private:
#             raise Exception("Cannot call .private() on tensor which is already private")
#
#         return ScalarTensor(
#             self.value,
#             min_val=min_val,
#             max_val=max_val,
#             entities=entities,
#             is_discrete=is_discrete,
#         )
