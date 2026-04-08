import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # Weights shape includes bias
        self.weights = np.random.rand(input_size + 1, output_size) 
        self._optimizer = None
        self.gradient_weights = None
        self.input_tensor = None

    def initialize(self, weights_initializer, bias_initializer):
        weight_matrix = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias_vector = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((weight_matrix, bias_vector))

    def forward(self, input_tensor):
        # Add bias column of 1s
        batch_size = input_tensor.shape[0]
        self.input_tensor = np.hstack((input_tensor, np.ones((batch_size, 1))))
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            
        return np.dot(error_tensor, self.weights[:-1, :].T)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
