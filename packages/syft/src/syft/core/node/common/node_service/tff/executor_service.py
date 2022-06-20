# # stdlib
# from typing import Optional

# # third party
# from tensorflow_federated.python.core.impl.executors import executor_base
# from tensorflow_federated.python.core.impl.executors import executor_value_base as evb


# class PySyftExecutorService(executor_base.Executor):
#     def close(self):
#         raise NotImplementedError

#     async def create_value(self, value, type_spec=None) -> evb.ExecutorValue:
#         raise NotImplementedError

#     async def create_call(
#         self, comp: evb.ExecutorValue, arg: Optional[evb.ExecutorValue] = None
#     ) -> evb.ExecutorValue:
#         raise NotImplementedError

#     async def create_struct(self, elements) -> evb.ExecutorValue:
#         raise NotImplementedError

#     async def create_selection(self, source, index) -> evb.ExecutorValue:
#         raise NotImplementedError
