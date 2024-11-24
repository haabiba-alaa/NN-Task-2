import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np

# Function to preprocess data
def preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)

        # Filling missing 'gender' values with mode (most frequent)
        mode_gender = df['gender'].mode()[0]
        df['gender'] = df['gender'].fillna(mode_gender)

        # Encoding 'gender' and 'bird category'
        df['gender'] = df['gender'].map({'male': 0, 'female': 1})
        df['bird category'] = df['bird category'].map({'A': 0, 'B': 1, 'C': 2})

        # Normalizing continuous features
        def min_max_normalize(column):
            min_value = column.min()
            max_value = column.max()
            return (column - min_value) / (max_value - min_value)

        df['body_mass'] = min_max_normalize(df['body_mass'])
        df['beak_length'] = min_max_normalize(df['beak_length'])
        df['beak_depth'] = min_max_normalize(df['beak_depth'])
        df['fin_length'] = min_max_normalize(df['fin_length'])

        return df
    except Exception as e:
        messagebox.showerror("File Error", f"Error processing the file: {e}")
        return None

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, neurons, output_size, learning_rate, include_bias, activation_function):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.include_bias = include_bias
        self.activation_function = activation_function
        self.input_data = None
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        prev_layer_size = input_size
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
        # output error signal
        output_error = self.layer_outputs[-1] - D 
        output_delta = output_error * self.activation_derivative(self.layer_outputs[-1])

        
        self.deltas = [output_delta]

        # in hidden layers
        for i in range(self.hidden_layers - 1, -1, -1): 
            layer_error = np.dot(self.deltas[0], self.weights[i + 1].T)
            layer_delta = layer_error * self.activation_derivative(self.layer_outputs[i + 1])
            self.deltas.insert(0, layer_delta)

    def weight_update(self):
        """
        Update weights and biases 
        """
        for i in range(self.hidden_layers):
            if i == 0:  #for first hidden layer
                weight_grad = np.dot(self.input_data.T, self.deltas[i])  
            else:  #for other layers
                weight_grad = np.dot(self.layer_outputs[i].T, self.deltas[i])  

            
            self.weights[i] -= self.learning_rate * weight_grad
           
            self.biases[i] -= self.learning_rate * np.sum(self.deltas[i], axis=0, keepdims=True)

        #update weights for the output Layer
        weight_grad = np.dot(self.layer_outputs[-2].T, self.deltas[-1])
        self.weights[-1] -= self.learning_rate * weight_grad
        self.biases[-1] -= self.learning_rate * np.sum(self.deltas[-1], axis=0, keepdims=True)

    '''
    def weight_update(self):
        
        for i in range(len(self.weights)):
            if i == 0:
                # for the first layer use input and delta
                self.weights[i] -= self.learning_rate * np.dot(self.input_data.T, self.deltas[i])
            else:
                # for other layers use the previous layer's output and delta
                self.weights[i] -= self.learning_rate * np.dot(self.layer_outputs[i].T, self.deltas[i])
            
            if self.include_bias:
                # Update biases for all layers
                self.biases[i] -= self.learning_rate * np.sum(self.deltas[i], axis=0, keepdims=True)
    '''

    '''
    def feed_forward_2(self, x):
        """
        A second feedforward pass to update weights based on the most recent activations.
        This method also stores activations (y) after each layer for backpropagation.
        """
        self.layer_outputs_2 = [x]  # Store activations (y) of neurons during the second forward pass
        self.layer_inputs_2 = []    # Store inputs (z) before activation during the second pass
        for i in range(self.hidden_layers):
            z = np.dot(self.layer_outputs_2[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs_2.append(z)
            a = self.activation(z)
            self.layer_outputs_2.append(a)

        final_input_2 = np.dot(self.layer_outputs_2[-1], self.weights[-1]) + self.biases[-1]
        self.layer_inputs_2.append(final_input_2)
        final_output_2 = self.activation(final_input_2)
        self.layer_outputs_2.append(final_output_2)

        # Update weights and biases based on the most recent activations (you would add the update logic here)
        # Example: Gradient Descent update rule can be applied after the second pass.
        self.update_weights()

        return final_output_2
'''

'''
    def update_weights(self):
        """
        Update weights and biases using gradient descent.
        """
        # Compute output error
        output_error = self.loss_grad(self.layer_outputs_2[-1], self.target)
        delta = output_error * self.activation_derivative(self.layer_inputs_2[-1])

        # Gradients for final layer weights and biases
        weight_grad = np.dot(self.layer_outputs_2[-2].T, delta)
        bias_grad = np.sum(delta, axis=0, keepdims=True)

        # Update final layer weights and biases
        self.weights[-1] -= self.learning_rate * weight_grad
        self.biases[-1] -= self.learning_rate * bias_grad

        # Backpropagate through hidden layers
        for i in reversed(range(self.hidden_layers)):
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(self.layer_inputs_2[i])
            weight_grad = np.dot(self.layer_outputs_2[i].T, delta)
            bias_grad = np.sum(delta, axis=0, keepdims=True)

            # Update weights and biases for current layer
            self.weights[i] -= self.learning_rate * weight_grad
            self.biases[i] -= self.learning_rate * bias_grad
'''

# GUI Class
class NNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Backpropagation GUI")

        # Input fields
        self.hidden_layers_label = tk.Label(root, text="Number of Hidden Layers:")
        self.hidden_layers_label.grid(row=0, column=0)
        self.hidden_layers_entry = tk.Entry(root)
        self.hidden_layers_entry.grid(row=0, column=1)

        self.neurons_label = tk.Label(root, text="Neurons in Each Hidden Layer (comma-separated):")
        self.neurons_label.grid(row=1, column=0)
        self.neurons_entry = tk.Entry(root)
        self.neurons_entry.grid(row=1, column=1)

        self.learning_rate_label = tk.Label(root, text="Learning Rate:")
        self.learning_rate_label.grid(row=2, column=0)
        self.learning_rate_entry = tk.Entry(root)
        self.learning_rate_entry.grid(row=2, column=1)

        self.epochs_label = tk.Label(root, text="Number of Epochs:")
        self.epochs_label.grid(row=3, column=0)
        self.epochs_entry = tk.Entry(root)
        self.epochs_entry.grid(row=3, column=1)

        self.bias_var = tk.BooleanVar(value=True)
        self.bias_label = tk.Label(root, text="Include Bias?")
        self.bias_label.grid(row=4, column=0)
        self.bias_checkbox = tk.Checkbutton(root, variable=self.bias_var)
        self.bias_checkbox.grid(row=4, column=1)

        self.activation_label = tk.Label(root, text="Activation Function:")
        self.activation_label.grid(row=5, column=0)
        self.activation_var = tk.StringVar(value="sigmoid")
        self.sigmoid_rb = tk.Radiobutton(root, text="Sigmoid", variable=self.activation_var, value="sigmoid")
        self.sigmoid_rb.grid(row=5, column=1)
        self.tanh_rb = tk.Radiobutton(root, text="Tanh", variable=self.activation_var, value="tanh")
        self.tanh_rb.grid(row=6, column=1)

        self.train_button = tk.Button(root, text="Train Network", command=self.train_network)
        self.train_button.grid(row=7, column=0, columnspan=2)

        self.result_label = tk.Label(root, text="Training Status:")
        self.result_label.grid(row=8, column=0)
        self.result_output = tk.Label(root, text="")
        self.result_output.grid(row=8, column=1)
        


        self.test_button = tk.Button(root, text="Test Network", command=self.test_network)
        self.test_button.grid(row=7, column=1)



        self.test_status_label = tk.Label(root, text="Testing Status:")
        self.test_status_label.grid(row=9, column=0)
        self.test_status_output = tk.Label(root, text="")
        self.test_status_output.grid(row=9, column=1)


    def get_inputs(self):
        try:
            hidden_layers = int(self.hidden_layers_entry.get())
            neurons = [int(x.strip()) for x in self.neurons_entry.get().split(",")]
            learning_rate = float(self.learning_rate_entry.get())
            epochs = int(self.epochs_entry.get())
            include_bias = self.bias_var.get()
            activation_function = self.activation_var.get()

            if len(neurons) != hidden_layers:
                raise ValueError("Number of neurons must match hidden layers.")

            return hidden_layers, neurons, learning_rate, epochs, include_bias, activation_function
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid inputs: {e}")
            return None



    def train_network(self):
        file_path = r"D:\UNI\NN\Task 2\birds.csv"
        # Preprocess the data
        preprocessed_df = preprocess_data(file_path)
        if preprocessed_df is None:
            return
        inputs = self.get_inputs()
        if inputs:
            hidden_layers, neurons, learning_rate, epochs, include_bias, activation_function = inputs
            X = preprocessed_df[['gender', 'body_mass', 'beak_length', 'beak_depth', 'fin_length']].values
            X = X.reshape(X.shape[0], -1)  # Ensuring it's 2D if necessary
            D = pd.get_dummies(preprocessed_df['bird category']).values  # One-hot encode targets

            # Create NeuralNetwork instance
            self.NN = NeuralNetwork(
                input_size=5, hidden_layers=hidden_layers, neurons=neurons,
                output_size=3, learning_rate=learning_rate,
                include_bias=include_bias, activation_function=activation_function
            )
        
            # Training loop
            for epoch in range(epochs):
                self.NN.input_data = X
                predictions = self.NN.forward(X)
                self.NN.backward(D)  # Compute gradients
                self.NN.weight_update()  # Update weights and biases

            self.result_output.config(text="Training Completed!", fg="green")


    def test_network(self):
        file_path = r"D:\UNI\NN\Task 2\birds.csv"

        # Preprocess the test data
        preprocessed_df = preprocess_data(file_path)
        

        if preprocessed_df is None:
            return

        X = preprocessed_df[['gender', 'body_mass', 'beak_length', 'beak_depth', 'fin_length']].values
        true_labels = preprocessed_df['bird category'].values

        # Use the trained NeuralNetwork instance for predictions
        predictions = self.NN.forward(X)  # Forward pass using the trained NN instance
        predicted_classes = np.argmax(predictions, axis=1)

        accuracy = np.mean(predicted_classes == true_labels)
        self.test_status_output.config(
            text=f"Testing Accuracy: {accuracy:.2%}", fg="blue"
        )



# Run the GUI
root = tk.Tk()
app = NNApp(root)
root.mainloop()
