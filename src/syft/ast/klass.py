from .. import ast
from .. import message as msg
from .. import pointer as ptr

from forbiddenfruit import curse


class Class(ast.callable.Callable):

    ""

    def __init__(self, name, path_and_name, ref):
        super().__init__(name, path_and_name, ref)
        self.pointer_name = self.path_and_name + "Pointer"

    def __repr__(self):
        return f"<Class:{self.name}>"

    def create_pointer_class(self):
        def run_class_method(attr, attr_path_and_name, __self, args, kwargs):
            # TODO: lookup actual return type instead of just guessing that it's identical
            result = getattr(self, self.pointer_name)(location=__self.location)

            cmd = msg.RunClassMethodMessage(
                attr_path_and_name, __self, args, kwargs, result.id_at_location
            )
            __self.location.send_msg(cmd)

            return result

        attrs = {}
        for attr_name, attr in self.attrs.items():
            attrs[attr_name] = lambda _self, *args, **kwargs: run_class_method(
                attr, attr.path_and_name, _self, args, kwargs
            )

        klass_pointer = type(self.pointer_name, (ptr.pointer.Pointer,), attrs)

        setattr(self, self.pointer_name, klass_pointer)

    def create_send_method(outer_self):
        def send(self, location):
            # Step 1: create pointer which will point to result
            ptr = getattr(outer_self, outer_self.pointer_name)(location=location)

            # Step 2: create message which contains object to send
            obj_msg = msg.SaveObjectMessage(id=ptr.id_at_location, obj=self)

            # Step 3: send message
            location.send_msg(obj_msg)

            # STep 4: return pointer
            return ptr

        # using curse because Numpy tries to lock down custom attributes
        curse(outer_self.ref, "send", send)
