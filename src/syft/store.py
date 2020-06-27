# class ObjectStore:
#     """Logic to store and retrieve objects within a worker"""
#
#     def __init__(self):
#         self._objects = {}
#
#     def store_object(self, id, obj):
#         self._objects[id] = obj
#
#     def get_object(self, id):
#         return self._objects[id]
#
#     def delete_object(self, id):
#         del self._objects[id]