class ObjectStore:
    """Logic to store and retrieve objects within a node"""

    def __init__(self):
        self._objects = {}

    def has_object(self, id):
        return id in self._objects

    def store_object(self, id, obj):
        self._objects[id] = obj

    def get_object(self, id):
        return self._objects[id]

    def get_objects_of_type(self, obj_type: type):
        out = list()
        for id, obj in self._objects.items():
            if isinstance(obj, obj_type):
                out.append(obj)
        return out

    def delete_object(self, id):
        del self._objects[id]

    def __repr__(self):
        return f"ObjectStore:{self._objects}"
