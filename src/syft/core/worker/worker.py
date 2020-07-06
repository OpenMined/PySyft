# -*- coding: utf-8 -*-
"""
stuff
"""

from ..message.message import RunClassMethodMessage
from ..message.message import SaveObjectMessage
from ..message.message import GetObjectMessage
from ..message.message import DeleteObjectMessage
from ..message.message import RunFunctionOrConstructorMessage

from ..store.store import ObjectStore

from ...ast.globals import Globals

from ..pointer.pointer import Pointer


class Worker:
    def __init__(self, id, supported_frameworks):
        self.id = id
        self.store = ObjectStore()
        self.frameworks = Globals()
        for fw in supported_frameworks:
            for name, ast in fw.ast.attrs.items():
                if name in self.frameworks.attrs:
                    raise KeyError(
                        "Framework already imported. Why are you importing it twice?"
                    )
                self.frameworks.attrs[name] = ast

        self.msg_router = {}
        self.msg_router[RunClassMethodMessage] = self.process_run_class_method_message
        self.msg_router[SaveObjectMessage] = self.process_save_object_message
        self.msg_router[GetObjectMessage] = self.process_get_object_message
        self.msg_router[DeleteObjectMessage] = self.process_delete_object_message
        self.msg_router[
            RunFunctionOrConstructorMessage
        ] = self.process_run_function_or_constructor_message

    def process_run_class_method_message(self, msg):

        self_possibly_pointer = msg._self
        args_with_pointers = msg.args
        kwargs_with_pointers = msg.kwargs

        result_id_at_location = msg.id_at_location

        # Step 1: Replace Pointers with Objects in self, args, and kwargs
        self_is_object = None
        args_with_objects = list()
        kwargs_with_objects = {}

        # Step 1a: set self_is_object to be the object self_possibly_pointer points to
        if issubclass(type(self_possibly_pointer), Pointer):
            self_is_object = self.store.get_object(self_possibly_pointer.id_at_location)
        else:
            self_is_object = self_possibly_pointer

        # Step 1b: replace arg pointers with objects the pointers point to
        for arg in args_with_pointers:
            if issubclass(type(arg), Pointer):
                args_with_objects.append(self.store.get_object(arg.id_at_location))
            else:
                args_with_objects.append(arg)

        # Step 1c: replace kwarg pointers with objects the pointers point to
        for name, kwarg in kwargs_with_objects.items():
            if issubclass(type(kwarg), Pointer):
                args_with_objects[name] = self.store.get_object(kwarg.id_at_location)
            else:
                args_with_objects[name] = kwarg

        # Step 2: Execute method
        result = self.frameworks(msg.path)(
            self_is_object, *args_with_objects, **kwargs_with_objects
        )

        # Step 3: Store result
        self.store.store_object(result_id_at_location, result)

        return True

    def process_run_function_or_constructor_message(self, msg):
        ""

    def process_save_object_message(self, msg):
        self.store.store_object(msg.id, msg.obj)

    def process_get_object_message(self, msg):
        return self.store.get_object(msg.id)

    def process_delete_object_message(self, msg):
        return self.store.delete_object(msg.id)

    def recv_msg(self, msg):
        return self.msg_router[type(msg)](msg)

    def __repr__(self):
        return f"<Worker id:{self.id}>"
