import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from testing import *  
from BPNN import MNN

df = pd.read_csv("/Users/habibaalaa/NN-Task-2/birds_preprocessed.csv")

def show_confusion_matrix(cm):
    window = tk.Tk()
    window.title("Confusion Matrix")

    title_label = tk.Label(window, text="Confusion Matrix", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)

    cm_text = "\n".join(["\t".join(map(str, row)) for row in cm])
    
    cm_label = tk.Label(window, text=cm_text, font=("Courier", 12), justify="left")
    cm_label.pack(pady=10)

    close_button = tk.Button(window, text="Close", command=window.destroy)
    close_button.pack(pady=10)

    window.mainloop()

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

        self.train_button = tk.Button(root, text="Get Results", command=self.train_test)
        self.train_button.grid(row=7, column=0, columnspan=2)

        self.result_label = tk.Label(root, text="Training Accuracy:")
        self.result_label.grid(row=8, column=0)
        self.result_output = tk.Label(root, text="")
        self.result_output.grid(row=8, column=1)

        self.test_status_label = tk.Label(root, text="Testing Accuracy:")
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

    def train_test(self):
        X = df.drop('bird category', axis=1).values
        Y = df['bird category'].values
        (X_train, y_train), (X_test, y_test) = TTS(X, Y)
        inputs = self.get_inputs()
        if inputs:
            hidden_layers, neurons, learning_rate, epochs, include_bias, activation_function = inputs
           
            self.NN = MNN(
                input_size=X.shape[1],
                hidden_layers=hidden_layers,
                neurons=neurons,
                output_size=3,
                learning_rate=learning_rate,
                include_bias=include_bias,
                activation_function=activation_function
            )
        
            self.NN.train(X_train, y_train, epochs)
            y_train_pred = self.NN.predict(X_train)
            y_train_true_class = np.argmax(y_train, axis=1)  # Convert one-hot to class labels
            train_accuracy = calculate_accuracy(y_train_true_class, y_train_pred)
            print(f"Training Accuracy: {train_accuracy}%")
            self.result_output.config(text=f"Training Accuracy: {train_accuracy:.2f}%", fg="green")

           # Calculate final accuracy on test data
            y_test_pred = self.NN.predict(X_test)
            y_test_true_class = np.argmax(y_test, axis=1)  # Convert one-hot to class labels
            test_accuracy = calculate_accuracy(y_test_true_class, y_test_pred)
            print(f"Test Accuracy: {test_accuracy}%")
            self.test_status_output.config(text=f"Testing Accuracy: {test_accuracy:.2f}%", fg="green")


            cm=confusion_matrix(y_test_true_class,y_test_pred)
            print(cm)
            show_confusion_matrix(cm)

            print(y_train_true_class)
            print(y_train_pred)

# Run the GUI
root = tk.Tk()
app = NNApp(root)
root.mainloop()
