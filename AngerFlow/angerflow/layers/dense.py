import numpy as np
from angerflow.layers.base import Layer

class Dense(Layer):
    def __init__(self,units,input_dim=None,activation=None):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
        self.activation = activation

        self.W = None
        self.b = None
        self.inputs = None

        if self.input_dim is not None:
            self.build(self.input_dim)

    def build(self, input_dim):
        self.input_dim = input_dim
        self.W = (np.random.randn(input_dim, self.units)*0.01).astype(np.float32)
        self.params = [self.W, self.b]

    def forward(self, inputs, training=True):
        if self.W is None:
            self.build(inputs.shape[1])

        self.inputs = inputs
        output = np.dot(inputs,self.W) + self.b
        return output

    def backward(self, grad_output):
        pass
    