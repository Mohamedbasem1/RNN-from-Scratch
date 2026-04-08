import numpy as np
from .Base import BaseLayer
import copy

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        
        # Learnable parameters (gamma and beta)
        self.weights = np.ones(channels)  # gamma
        self.bias = np.zeros(channels)     # beta
        
        # Moving averages for test time
        self.moving_mean = None
        self.moving_var = None
        self.decay = 0.8  # decay rate for moving average
        
        # Cache for backward pass
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None
        self.input_tensor = None
        self.epsilon = 1e-10
        
        # Gradients
        self.gradient_weights = None
        self.gradient_bias = None
        
        # Optimizer
        self._optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None
        
        # To track input shape for reformat
        self.input_shape = None
        
    def initialize(self, weights_initializer, bias_initializer):
        # Always initialize gamma (weights) to 1 and beta (bias) to 0
        # Ignore the provided initializers
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
    
    def reformat(self, tensor):
        """
        Reformats between image tensor (B, C, H, W) and vector tensor (B*H*W, C)
        """
        if len(tensor.shape) == 4:
            # Image to vector: (B, C, H, W) -> (B*H*W, C)
            batch_size, channels, height, width = tensor.shape
            self.input_shape = tensor.shape
            # Transpose to (B, H, W, C) then reshape
            tensor = np.transpose(tensor, (0, 2, 3, 1))
            return tensor.reshape(-1, channels)
        else:
            # Vector to image: (B*H*W, C) -> (B, C, H, W)
            if self.input_shape is not None:
                batch_size, channels, height, width = self.input_shape
                # Reshape to (B, H, W, C)
                tensor = tensor.reshape(batch_size, height, width, channels)
                # Transpose to (B, C, H, W)
                return np.transpose(tensor, (0, 3, 1, 2))
            return tensor
    
    def forward(self, input_tensor):
        # Store original input
        self.input_tensor = input_tensor
        
        # Reformat if needed (4D -> 2D)
        is_image = len(input_tensor.shape) == 4
        if is_image:
            input_tensor = self.reformat(input_tensor)
        
        if self.testing_phase:
            # Test mode: use moving averages
            if self.moving_mean is None:
                self.moving_mean = np.zeros(self.channels)
                self.moving_var = np.ones(self.channels)
            
            normalized = (input_tensor - self.moving_mean) / np.sqrt(self.moving_var + self.epsilon)
            output = self.weights * normalized + self.bias
        else:
            # Training mode: use batch statistics
            self.batch_mean = np.mean(input_tensor, axis=0)
            self.batch_var = np.var(input_tensor, axis=0)
            
            # Update moving averages
            if self.moving_mean is None:
                self.moving_mean = self.batch_mean
                self.moving_var = self.batch_var
            else:
                self.moving_mean = self.decay * self.moving_mean + (1 - self.decay) * self.batch_mean
                self.moving_var = self.decay * self.moving_var + (1 - self.decay) * self.batch_var
            
            # Normalize
            self.normalized = (input_tensor - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            
            # Scale and shift
            output = self.weights * self.normalized + self.bias
        
        # Reformat back if needed (2D -> 4D)
        if is_image:
            output = self.reformat(output)
        
        return output
    
    def backward(self, error_tensor):
        # Reformat if needed
        is_image = len(error_tensor.shape) == 4
        if is_image:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(self.input_tensor)
        else:
            input_tensor = self.input_tensor
        
        # Compute gradients for weights and bias
        self.gradient_weights = np.sum(error_tensor * self.normalized, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        
        # Update weights and bias if optimizer is set
        if self._optimizer:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)
        
        # Compute gradient w.r.t input using the helper function
        from . import Helpers
        grad_input = Helpers.compute_bn_gradients(
            error_tensor, 
            input_tensor, 
            self.weights, 
            self.batch_mean, 
            self.batch_var, 
            self.epsilon
        )
        
        # Reformat back if needed
        if is_image:
            grad_input = self.reformat(grad_input)
        
        return grad_input
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        if optimizer:
            self._optimizer_weights = copy.deepcopy(optimizer)
            self._optimizer_bias = copy.deepcopy(optimizer)
