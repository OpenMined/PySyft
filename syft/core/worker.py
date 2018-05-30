from .torch_ import utils
import json

class BaseWorker(object):

    def __init__(self, id=0):

        self.id = id
        self.objects = {}

    def send(message,recipient):
        raise NotImplementedError


class LocalWorker(BaseWorker):
    
    def __init__(self,id=0):
        super().__init__(id)  
        
    def send(self, message, recipient):
        recipient.receive(message._ser())

    def receive(self, message):

        message_obj = json.loads(message)
        obj_type = utils.types_guard(message_obj['torch_type'])
        obj = obj_type._deser(obj_type,message_obj['data'])
        self.objects[obj.id] = obj

    