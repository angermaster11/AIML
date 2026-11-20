import numpy as np
from angerflow.layers.base import Layer


class Sigmoid(Layer):
    """
    Sigmoid Activation: S(x) = 1 / (1 + exp(-x))
    Maps input values to (0, 1) range. Useful for binary classification output.
    """
    def __init__(self):
        super().__init__()
        self.out = None  # Cache output for backprop: dS/dx = S(x) * (1 - S(x))

    def forward(self, inputs, training=True):
        # Clip inputs to [-500, 500] to prevent exp overflow (inf / NaN)
        safe_inputs = np.clip(inputs, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-safe_inputs))
        return self.out

    def backward(self, grad_output):
        pass


class ReLU(Layer):
    """
    Rectified Linear Unit: R(x) = max(0, x)
    Standard activation for hidden layers. Non-saturating gradient for x > 0.
    """
    def __init__(self):
        super().__init__()
        self.inputs = None  # Cache inputs: derivative is 1 if x > 0, else 0

    def forward(self, inputs, training=True):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        pass


class Tanh(Layer):
    """
    Hyperbolic Tangent: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Maps input to (-1, 1). Zero-centered.
    """
    def __init__(self):
        super().__init__()
        self.out = None  # Cache output: dTanh/dx = 1 - Tanh(x)^2

    def forward(self, inputs, training=True):
        # np.tanh is numerically stable internally
        self.out = np.tanh(inputs)
        return self.out

    def backward(self, grad_output):
        pass


class Softmax(Layer):
    """
    Softmax Activation: Softmax(x_i) = exp(x_i) / sum(exp(x_j))
    Converts logits into a probability distribution summing to 1 across the last axis.
    """
    def __init__(self):
        super().__init__()
        self.out = None  # Cache output probabilities

    def forward(self, inputs, training=True):
        # Subtract max per sample (axis=-1) for numerical stability (prevents overflow in exp)
        shifted = inputs - np.max(inputs, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        # Normalize across classes: keepdims=True preserves shape (batch_size, 1) for broadcasting
        self.out = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        return self.out

    def backward(self, grad_output):
        pass
