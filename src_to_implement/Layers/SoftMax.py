import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.prediction = None

    def forward(self, input_tensor):
        x = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_x = np.exp(x)
        self.prediction = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.prediction

    def backward(self, error_tensor):
        # We compute the gradient of the loss w.r.t input
        # element-wise: dx = y * (dy - sum(y*dy))
        sum_y_dy = np.sum(error_tensor * self.prediction, axis=1, keepdims=True)
        return self.prediction * (error_tensor - sum_y_dy)
