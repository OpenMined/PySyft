
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
        recipient.receive(message)

    def receive(self, message):
        self.objects[message.id] = message

    