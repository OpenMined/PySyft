

class Layer(object):
    def __init__(self, *args, **kargs):
        
        self.args = args
        self.kargs = kargs

class Dense(Layer):
    def __init__(self,*args, **kargs):
        super(Dense, self).__init__(*args, **kargs)
        
        self.name = "Dense"
        self.args = args
        self.kargs = kargs

class Activation(Layer):
    def __init__(self,*args, **kargs):
        super(Activation, self).__init__(*args, **kargs)

        self.name = "Activation"
        self.args = args
        self.kargs = kargs

class Conv2D(Layer):
    def __init__(self,*args, **kargs):
        super(Conv2D, self).__init__(*args, **kargs)

        self.name = "Conv2D"
        self.args = args
        self.kargs = kargs

class MaxPooling2D(Layer):
    def __init__(self,*args, **kargs):
        super(MaxPooling2D, self).__init__(*args, **kargs)

        self.name = "MaxPooling2D"
        self.args = args
        self.kargs = kargs

class AveragePooling2D(Layer):
    def __init__(self,*args, **kargs):
        super(AveragePooling2D, self).__init__(*args, **kargs)

        self.name = "AveragePooling2D"
        self.args = args
        self.kargs = kargs


class Flatten(Layer):
    def __init__(self,*args, **kargs):
        super(Flatten, self).__init__(*args, **kargs)

        self.name = "Flatten"
        self.args = args
        self.kargs = kargs

class ReLU(Layer):
    def __init__(self,*args, **kargs):
        super(ReLU, self).__init__(*args, **kargs)

        self.name = "ReLU"
        self.args = args
        self.kargs = kargs