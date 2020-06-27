# class RunClassMethodMessage:
#
#     def __init__(self, path, _self, args, kwargs, id_at_location):
#         self.path = path
#         self._self = _self
#         self.args = args
#         self.kwargs = kwargs
#         self.id_at_location = id_at_location
#
#
# class RunFunctionOrConstructorMessage:
#
#     def __init__(self, path, args, kwargs):
#         self.path = path
#         self.args = args
#         self.kwargs = kwargs
#
#
# class SaveObjectMessage:
#
#     def __init__(self, id, obj):
#         self.id = id
#         self.obj = obj
#
#
# class GetObjectMessage:
#
#     def __init__(self, id):
#         self.id = id
#
#
# class DeleteObjectMessage:
#
#     def __init__(self, id):
#         self.id = id