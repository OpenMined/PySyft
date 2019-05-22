


class Layer(object):
    def __init__(self, *args, **kargs):
        
        self.arguments = kargs

class Dense(Layer):
    def __init__(self,*args, **kargs):
        super(Dense, self).__init__(*args, **kargs)
        
        self.name = "Dense"
        self.arguments = kargs

class Conv2d(Layer):
    def __init__(self, **kargs):
        super(Dense, self).__init__(*args, **kargs)

        self.name = "Conv2D"
        self.arguments = kargs

class MaxPooling2D(Layer):
    def __init__(self, **kargs):
        super(Dense, self).__init__(*args, **kargs)

        self.name = "MaxPooling2D"
        self.arguments = kargs