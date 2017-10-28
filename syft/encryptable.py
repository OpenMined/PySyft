class Encryptable():
    def __init__(self):
        self.pubkey = None
    
    def encrypt(self, pubkey):
        for name in self.encryptables:
            obj = getattr(self, name)
            if isinstance(obj, Encryptable):
                setattr(self, name, obj.encrypt(pubkey))     
            elif hasattr(obj,'__iter__'):
                setattr(self, name, [o.encrypt(pubkey) for o in obj])
        self.pubkey = pubkey
        return self
        
    def decrypt(self, seckey):
        for name in self.encryptables:
            obj = getattr(self, name)
            if isinstance(obj, Encryptable):
                setattr(self, name, obj.decrypt(seckey))
            elif hasattr(obj,'__iter__'):
                setattr(self, name, [o.decrypt(seckey) for o in obj])
        self.pubkey = None
        return self