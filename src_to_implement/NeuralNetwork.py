import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = False

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase
        for layer in self.layers:
            layer.testing_phase = phase

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        
        # Calculate data loss
        loss = self.loss_layer.forward(output, label_tensor)
        
        # Add regularization loss from all trainable layers
        regularization_loss = 0
        for layer in self.layers:
            if layer.trainable and hasattr(layer, 'optimizer') and layer.optimizer:
                if hasattr(layer.optimizer, 'regularizer') and layer.optimizer.regularizer:
                    # Get the weights from the layer
                    if hasattr(layer, 'weights'):
                        regularization_loss += layer.optimizer.regularizer.norm(layer.weights)
        
        # Total loss = data loss + regularization loss
        total_loss = loss + regularization_loss
        self.loss.append(total_loss)
        return output, label_tensor

    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False
        for _ in range(iterations):
            _, label_tensor = self.forward()
            self.backward(label_tensor)

    def test(self, input_tensor):
        self.phase = True
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output
