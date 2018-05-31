from .torch_ import utils
import json

class BaseWorker(object):

    def __init__(self, id=0):

        self.id = id
        self.objects = {}

    def send_obj(message,recipient):
        raise NotImplementedError


class LocalWorker(BaseWorker):
    
    def __init__(self,id=0):
        super().__init__(id)  
        
    def send_obj(self, obj, recipient):
        print(obj)
        recipient.receive_obj(obj._ser())

    def receive_obj(self, message):

        message_obj = json.loads(message)
        obj_type = utils.types_guard(message_obj['torch_type'])
        obj = obj_type._deser(obj_type,message_obj['data'])

        self.objects[message_obj['id']] = obj
        obj.id = message_obj['id']

    def request_obj(self,obj_id,sender):
        
        sender.send_obj(sender.objects[obj_id],self)
        return self.objects[obj_id]