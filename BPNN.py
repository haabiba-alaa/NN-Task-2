import numpy as np

class MNN:
    def __init__(self, input_size, hidden_layers, neurons, output_size, learning_rate, include_bias, activation_function):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.include_bias = include_bias
        self.activation_function = activation_function
        self.input_data = None
        
        
        self.weights = []
        self.biases = []
        prev_layer_size = input_size
        
        '''
        # Initialize weights using Xavier initialization
        for layer_size in neurons:
            self.weights.append(np.random.randn(prev_layer_size, layer_size) * np.sqrt(2. / prev_layer_size))
            self.biases.append(np.zeros((1, layer_size)))
            prev_layer_size = layer_size

        self.weights.append(np.random.randn(prev_layer_size, output_size) * np.sqrt(2. / prev_layer_size))
        self.biases.append(np.zeros((1, output_size)))
        '''
        #Random_init
        for layer_size in neurons:
            self.weights.append(np.random.randn(prev_layer_size, layer_size) * 0.1)
            self.biases.append(np.zeros((1, layer_size)))
            prev_layer_size = layer_size

        self.weights.append(np.random.randn(prev_layer_size, output_size) * 0.1)
        self.biases.append(np.zeros((1, output_size)))

    def activation(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)  
        
    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return x * (1 - x)  
        elif self.activation_function == 'tanh':
            return 1 - x**2  
        
    def forward(self, x):
        self.layer_outputs = [x]
        self.layer_inputs = []  
        for i in range(self.hidden_layers):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)
            a = self.activation(z)
            self.layer_outputs.append(a)
        
        final_input = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(final_input)
        
        final_output = self.activation(final_input)
        self.layer_outputs.append(final_output)
        
        return final_output

    def backward(self, D):
        output_error = self.layer_outputs[-1] - D 
        output_delta = output_error * self.activation_derivative(self.layer_outputs[-1])

        self.deltas = [output_delta]

        # in hidden layers
        for i in range(self.hidden_layers - 1, -1, -1): 
            layer_error = np.dot(self.deltas[0], self.weights[i + 1].T)
            layer_delta = layer_error * self.activation_derivative(self.layer_outputs[i + 1])
            self.deltas.insert(0, layer_delta)

    def weight_update(self):
        for i in range(self.hidden_layers):
            if i == 0:  
                weight_grad = np.dot(self.input_data.T, self.deltas[i])
            else:
                weight_grad = np.dot(self.layer_outputs[i].T, self.deltas[i])

            self.weights[i] -= self.learning_rate * weight_grad
            self.biases[i] -= self.learning_rate * np.sum(self.deltas[i], axis=0, keepdims=True)

        weight_grad = np.dot(self.layer_outputs[-2].T, self.deltas[-1])
        self.weights[-1] -= self.learning_rate * weight_grad
        self.biases[-1] -= self.learning_rate * np.sum(self.deltas[-1], axis=0, keepdims=True)

    def predict(self, X):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions

    def train(self, X_train, Y_train, epochs):
        for epoch in range(epochs):
            for i in range(X_train.shape[0]):
                self.input_data = X_train[i:i+1]  
                D = Y_train[i:i+1]  
                output = self.forward(self.input_data)
                self.backward(D)
                self.weight_update()
            print(f"Epoch {epoch+1}/{epochs} completed.")