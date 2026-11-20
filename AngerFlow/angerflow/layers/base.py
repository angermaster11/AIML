from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.params = []
        self.grads = []
        self.trainable = True

    @abstractmethod
    def forward(self,inputs,training=True):
        """
        inputs: np.ndarray shape (batch_input_dim)
        RETURNS: output np.ndarray
        """
        pass

    @abstractmethod
    def backward(self,grad_output):
        """
        grad_output: dL/dy from the next layer
        RETRUNS : dL/dx to pass the previous layer
        """
        pass
