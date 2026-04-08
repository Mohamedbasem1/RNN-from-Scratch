import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False
        self.input_tensor = None
        self.indices = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, h, w = input_tensor.shape
        stride_h, stride_w = self.stride_shape
        pool_h, pool_w = self.pooling_shape

        out_h = (h - pool_h) // stride_h + 1
        out_w = (w - pool_w) // stride_w + 1

        output = np.zeros((batch_size, channels, out_h, out_w))
        self.indices = [] # Store max indices for backward

        # Naive implementation with loops (can be optimized)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride_h
                        h_end = h_start + pool_h
                        w_start = j * stride_w
                        w_end = w_start + pool_w
                        
                        patch = input_tensor[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        output[b, c, i, j] = max_val
                        
                        # Store index of max value for backward pass
                        # flattened index within the patch
                        idx = np.unravel_index(np.argmax(patch), patch.shape)
                        self.indices.append((b, c, h_start + idx[0], w_start + idx[1], i, j))

        return output

    def backward(self, error_tensor):
        output = np.zeros_like(self.input_tensor)
        batch_size, channels, out_h, out_w = error_tensor.shape
        
        # This approach with self.indices list is very memory inefficient and slow for large tensors
        # But correct for simple loop-based logic.
        # A better way is to "upsample" error_tensor using the masks created during forward.
        
        # Since I used a list, let's iterate.
        # But wait, self.indices in forward is O(N).
        # Re-running loops in backward might be better if I don't store exact indices efficiently.
        
        stride_h, stride_w = self.stride_shape
        pool_h, pool_w = self.pooling_shape
        
        count = 0 
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride_h
                        h_end = h_start + pool_h
                        w_start = j * stride_w
                        w_end = w_start + pool_w
                        
                        patch = self.input_tensor[b, c, h_start:h_end, w_start:w_end]
                        idx = np.unravel_index(np.argmax(patch), patch.shape)
                        
                        output[b, c, h_start + idx[0], w_start + idx[1]] += error_tensor[b, c, i, j]

        return output
