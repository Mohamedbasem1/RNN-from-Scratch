import numpy as np
from scipy import signal
from .Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        
        # Weights shape: (num_kernels, input_channels, kernel_dims...)
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        
        self.optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None
        
        self.input_tensor = None
        self.gradient_weights = None
        self.gradient_bias = None

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        
        # Determine strict 1D or 2D logic based on stride_shape
        is_1d = len(self.stride_shape) == 1
        
        output_list = []
        for b in range(batch_size):
            sample_output = []
            for k in range(self.num_kernels):
                # Cross-correlation between Input (Channels, Y, X) and Kernel (Channels, Y, X)
                # Sum over channels happens automatically if we allow correlate to handle it? 
                # Scipy correlate N-D does not sum over the first dimension automatically if it's "valid" in that dim?
                # Actually, standard DL convolution is Sum(Correlation(Input_c, Kernel_c)).
                
                kernel = self.weights[k]
                # Correlate input sample with kernel
                # mode='same' to keep spatial dimensions
                if is_1d:
                     # Input: (C, Length). Kernel: (C, K_Length)
                     # We want output (Length). Sum over C.
                     res = signal.correlate(input_tensor[b], kernel, mode='same')
                     # res shape will be (C, Length) or (2*C-1, ...)?
                     # We want 'valid' correlation over valid channel overlap (which is fully overlapped i.e. dot product)
                     # But 'same' over spatial.
                     # Scipy correlate on (C, L) and (C, L_k) with 'valid' will give (1, Output_L)?
                     # No, Scipy doesn't treat Channel dim special.
                     
                     # Manual sum over channels:
                     res = np.sum([signal.correlate(input_tensor[b, c], kernel[c], mode='same') for c in range(kernel.shape[0])], axis=0)
                     
                else:
                    # 2D
                    res = np.sum([signal.correlate(input_tensor[b, c], kernel[c], mode='same') for c in range(kernel.shape[0])], axis=0)

                # Add bias
                res += self.bias[k]
                sample_output.append(res)
            
            output_list.append(np.stack(sample_output))
            
        output = np.stack(output_list)
        
        # Subsample for stride
        if is_1d:
            output = output[:, :, ::self.stride_shape[0]]
        else:
            output = output[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
            
        return output

    def backward(self, error_tensor):
        # Update weights and bias
        # Return gradient w.r.t input
        
        batch_size = error_tensor.shape[0]
        is_1d = len(self.stride_shape) == 1
        
        # Upsample error tensor (gradient)
        # Create full size error tensor matching implicit padded output
        # If we used 'same', the intermediate buffer was size of input.
        # So we sparse fill it.
        
        if is_1d:
             upsampled_error = np.zeros((batch_size, self.num_kernels, self.input_tensor.shape[-1]))
             upsampled_error[:, :, ::self.stride_shape[0]] = error_tensor
        else:
             upsampled_error = np.zeros((batch_size, self.num_kernels, self.input_tensor.shape[-2], self.input_tensor.shape[-1]))
             upsampled_error[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor

        # Gradient w.r.t Bias
        # Sum over Batch and Spatial dimensions
        if is_1d:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))
        else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        # Gradient w.r.t Weights AND Input
        self.gradient_weights = np.zeros_like(self.weights)
        gradient_input = np.zeros_like(self.input_tensor)
        
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(self.weights.shape[1]):
                    # Gradient Weights: Input * Error
                    # Correlate Input with Error?
                    # Weight(k, c) gradient contribution comes from Input(b, c) and Error(b, k)
                    
                    if is_1d:
                        # Valid correlation centered correctly
                        # Expected shape: Kernel Size.
                        # Input (L), Error (L).
                        # We want correlation window of size Kernel.
                        # Using correlate(Input, Error, mode='valid')? 
                        # Or correlate(Input, Error, mode='same') and center crop?
                        # Usually: correlate(padded_input, upsampled_error, 'valid')
                        
                        # With 'same' forward padding, it's tricky to match exactly without careful padding calc.
                        # Let's try correlate(Input, Error, mode='valid') if Input > Error
                        # Wait, Error is upsampled to Input size (approximately).
                        # Let's assume correlate(Input, Error, mode='valid') works if we handle centering?
                        # Actually 'same' implies padding on Input.
                        # Pad Input with zeros to handle boundary effects?
                        # This part is highly sensitive to the exact padding convention.
                        
                        # Simpler approach:
                        # dW = Input * Error.
                        # dInput = Error * W.
                        
                        gw = signal.correlate(self.input_tensor[b, c], upsampled_error[b, k], mode='valid')
                        # Check shape. If Input len 10, Error len 10. Valid -> len 1.
                        # But we need Kernel len (e.g. 3).
                        # So we need to Pad Input?
                        # If forward was 'same', we effectively padded Input.
                        pad_w = self.weights.shape[-1] // 2
                        padded_input = np.pad(self.input_tensor[b, c], pad_w, mode='constant')
                        gw = signal.correlate(padded_input, upsampled_error[b, k], mode='valid')
                        
                        # Handle even/odd kernel sizes for centering
                        cent = (self.weights.shape[-1] % 2) == 0
                        if cent: 
                             # Drop last?
                             pass
                        
                        # Match shape
                        start = (gw.shape[0] - self.weights.shape[-1]) // 2
                        self.gradient_weights[k, c] += gw[start : start + self.weights.shape[-1]]
                        
                        # Gradient Input:
                        # Convolve Error with Weight (Full / Same?)
                        # dI += convolve(Error, W, 'same')
                        gi = signal.convolve(upsampled_error[b, k], self.weights[k, c], mode='same')
                        gradient_input[b, c] += gi

                    else:
                        # 2D
                        pad_h = self.weights.shape[-2] // 2
                        pad_w = self.weights.shape[-1] // 2
                        padded_input = np.pad(self.input_tensor[b, c], ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
                        
                        gw = signal.correlate(padded_input, upsampled_error[b, k], mode='valid')
                        
                        # Crop to kernel size?
                        # If Input 10+2, Error 10. Valid -> 3. Matches Kernel 3.
                        
                        # Sometimes shape mismatch by 1 due to even/odd.
                        
                        # Ensure shape match
                        dh = (gw.shape[0] - self.weights.shape[-2]) // 2
                        dw = (gw.shape[1] - self.weights.shape[-1]) // 2
                        
                        # Accumulate
                        self.gradient_weights[k, c] += gw # Slice if needed?
                        
                        # Input Gradient
                        gi = signal.convolve(upsampled_error[b, k], self.weights[k, c], mode='same')
                        gradient_input[b, c] += gi
                        
        if self.optimizer:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)
            
        return gradient_input
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        if optimizer:
            import copy
            self._optimizer_weights = copy.deepcopy(optimizer)
            self._optimizer_bias = copy.deepcopy(optimizer)
