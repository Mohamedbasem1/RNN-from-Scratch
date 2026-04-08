import numpy as np
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
import copy


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Hidden state
        self.hidden_state = np.zeros(hidden_size)
        
        # Memorize flag for TBPTT (Truncated Backpropagation Through Time)
        self._memorize = False
        
        # FC layer for computing hidden state: [h_t-1, x_t] -> h_t
        # Input: concatenated [hidden_state, input] of size (hidden_size + input_size)
        # Output: hidden_state of size hidden_size
        self.fc_hidden = FullyConnected(hidden_size + input_size, hidden_size)
        
        # Activation for hidden state
        self.tanh_hidden = TanH()
        
        # FC layer for output: h_t -> y_t
        self.fc_output = FullyConnected(hidden_size, output_size)
        
        # Optimizer
        self._optimizer = None
        
        # Cache for backward pass
        self.input_tensor_sequence = None
        self.hidden_states = None
        self.concatenated_inputs = None
        
    @property
    def memorize(self):
        return self._memorize
    
    @memorize.setter
    def memorize(self, value):
        self._memorize = value
    
    @property
    def weights(self):
        """Return weights of the FC layer that computes hidden state"""
        return self.fc_hidden.weights
    
    @weights.setter
    def weights(self, weights):
        """Set weights of the FC layer that computes hidden state"""
        self.fc_hidden.weights = weights
    
    @property
    def gradient_weights(self):
        """Return gradient of weights"""
        return self.fc_hidden.gradient_weights
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        if optimizer:
            # Set optimizers for internal layers
            self.fc_hidden.optimizer = copy.deepcopy(optimizer)
            self.fc_output.optimizer = copy.deepcopy(optimizer)
    
    def initialize(self, weights_initializer, bias_initializer):
        """Initialize weights of internal FC layers"""
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)
    
    def calculate_regularization_loss(self):
        """Calculate regularization loss from internal layers"""
        reg_loss = 0
        if self._optimizer and hasattr(self._optimizer, 'regularizer') and self._optimizer.regularizer:
            reg_loss += self._optimizer.regularizer.norm(self.fc_hidden.weights)
            reg_loss += self._optimizer.regularizer.norm(self.fc_output.weights)
        return reg_loss
    
    def forward(self, input_tensor):
        """
        Forward pass through RNN
        
        Args:
            input_tensor: shape (batch_size, input_size)
                         batch_size is treated as time dimension
        
        Returns:
            output_tensor: shape (batch_size, output_size)
        """
        batch_size = input_tensor.shape[0]  # This is the time dimension
        
        # Initialize hidden state
        if not self._memorize:
            self.hidden_state = np.zeros(self.hidden_size)
        
        # Store for backward pass
        self.input_tensor_sequence = input_tensor
        self.hidden_states = []
        self.concatenated_inputs = []
        
        outputs = []
        
        # Process each time step
        for t in range(batch_size):
            # Get input at time t
            x_t = input_tensor[t:t+1, :]  # shape: (1, input_size)
            
            # Concatenate hidden state and input
            h_prev = self.hidden_state.reshape(1, -1)  # shape: (1, hidden_size)
            concat_input = np.hstack([h_prev, x_t])  # shape: (1, hidden_size + input_size)
            self.concatenated_inputs.append(concat_input)
            
            # Compute new hidden state: h_t = tanh(W * [h_t-1, x_t])
            hidden_fc_output = self.fc_hidden.forward(concat_input)
            hidden_activated = self.tanh_hidden.forward(hidden_fc_output)
            
            # Update hidden state
            self.hidden_state = hidden_activated.flatten()
            self.hidden_states.append(self.hidden_state.copy())
            
            # Compute output: y_t = W_out * h_t
            y_t = self.fc_output.forward(hidden_activated)
            outputs.append(y_t)
        
        # Stack outputs
        output_tensor = np.vstack(outputs)  # shape: (batch_size, output_size)
        
        return output_tensor
    
    def backward(self, error_tensor):
        """
        Backward pass through RNN (Backpropagation Through Time)
        
        Args:
            error_tensor: shape (batch_size, output_size)
        
        Returns:
            gradient_input: shape (batch_size, input_size)
        """
        batch_size = error_tensor.shape[0]
        
        # Initialize gradients
        gradient_input = []
        gradient_hidden = np.zeros(self.hidden_size)
        
        # Initialize accumulators for gradient_weights
        accumulated_gradient_hidden = None
        accumulated_gradient_output = None
        
        # Backpropagate through time (from T-1 to 0)
        for t in reversed(range(batch_size)):
            # Error at time t
            error_t = error_tensor[t:t+1, :]  # shape: (1, output_size)
            
            # Restore hidden state at time t
            h_t = self.hidden_states[t].reshape(1, -1)  # shape: (1, hidden_size)
            
            # Backward through output layer
            self.fc_output.input_tensor = np.hstack([h_t, np.ones((1, 1))])  # Restore input with bias
            error_hidden = self.fc_output.backward(error_t)  # shape: (1, hidden_size)
            
            # Accumulate gradient_weights for output layer
            if accumulated_gradient_output is None:
                accumulated_gradient_output = self.fc_output.gradient_weights.copy()
            else:
                accumulated_gradient_output += self.fc_output.gradient_weights
            
            # Add gradient from next time step
            error_hidden += gradient_hidden.reshape(1, -1)
            
            # Backward through tanh activation
            self.tanh_hidden.activations = h_t  # Restore activations
            error_hidden_pre_activation = self.tanh_hidden.backward(error_hidden)
            
            # Backward through hidden FC layer
            concat_input = self.concatenated_inputs[t]
            self.fc_hidden.input_tensor = np.hstack([concat_input, np.ones((1, 1))])  # Restore input with bias
            error_concat = self.fc_hidden.backward(error_hidden_pre_activation)
            
            # Accumulate gradient_weights for hidden layer
            if accumulated_gradient_hidden is None:
                accumulated_gradient_hidden = self.fc_hidden.gradient_weights.copy()
            else:
                accumulated_gradient_hidden += self.fc_hidden.gradient_weights
            
            # Split gradient for hidden state and input
            gradient_hidden = error_concat[0, :self.hidden_size]  # Gradient w.r.t. h_t-1
            gradient_x_t = error_concat[0, self.hidden_size:]  # Gradient w.r.t. x_t
            
            # Store input gradient
            gradient_input.append(gradient_x_t)
        
        # Assign accumulated gradients to the layers
        self.fc_hidden.gradient_weights = accumulated_gradient_hidden
        self.fc_output.gradient_weights = accumulated_gradient_output
        
        # Reverse to get gradients in correct order (time 0 to T-1)
        gradient_input = np.array(list(reversed(gradient_input)))
        
        return gradient_input
