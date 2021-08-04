# # stdlib
# import logging
# from multiprocessing import Process
# import socket
# from time import time
# from typing import Any as TypeAny
# from typing import Dict as TypeDict
# from typing import Generator
# from typing import List as TypeList
#
# # third party
# import _pytest
# import pytest
#
# # syft absolute
# import syft as sy
# syft absolute
# from syft import logger
#
# logger.remove()
#
#
#
#
# # @pytest.fixture(scope="session")
# # def node() -> sy.VirtualMachine:
# #     return sy.VirtualMachine(name="Bob")
#
#
# # this is not working anymore
# # @pytest.fixture(autouse=True)
# # def node_store(node: sy.VirtualMachine) -> None:
# #     node.store.clear()
#
# #
# # @pytest.fixture(scope="session")
# # def client(node: sy.VirtualMachine) -> sy.VirtualMachineClient:
# #     return node.get_client()
# #
# #
# # @pytest.fixture(scope="session")
# # def root_client(node: sy.VirtualMachine) -> sy.VirtualMachineClient:
# #     return node.get_root_client()
