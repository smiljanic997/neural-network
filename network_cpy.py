from activation import sigmoid, dsigmoid, rel, drel

import numpy as np
import pandas as pd
import os


class Network :

    max_error = 0.00001

    def __init__(self, number_of_inputs, number_of_outputs, hidden_layer_count, learning_rate):
        """
            Weights are divided into two matrices. One for input-to-hidden layer and one for
            hidden-to-output layer.
            Biases are represented as two vectors, one for hidden and one for output layer.
        """
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.hidden_layer_count = hidden_layer_count
        self.weight_matrix_ih = np.random.rand(hidden_layer_count, number_of_inputs)
        self.weight_matrix_ho = np.random.rand(number_of_outputs, hidden_layer_count)
        self.bias_hidden = np.random.rand(hidden_layer_count, 1)
        self.bias_out = np.random.rand(number_of_outputs, 1)
        self.learning_rate = learning_rate

    
    def feedforward(self, input_array):
        """
            Accepts two numbers as input array and returns network and hidden layer output.
        """
        input_matrix = np.reshape(input_array, (self.number_of_inputs, 1))
        hidden_layer_output = self.sigmoid_vectorized(np.add(np.dot(self.weight_matrix_ih, input_matrix), self.bias_hidden))
        network_output = np.add(np.dot(self.weight_matrix_ho, hidden_layer_output), self.bias_out)
        return self.rel_vectorized(network_output.flatten()), hidden_layer_output

    
    def train(self, input_array, target_array):
        """
            Accepts training pairs and changes weight matrices with backpropagation algorithm.
        """
        iter = 0
        while True:
            iter = iter + 1
            cumulative_error = 0
            for input, target in zip(input_array, target_array):
                net_outputs, hidden_output = self.feedforward(input)
                net_outputs = np.reshape(net_outputs, (self.number_of_outputs, 1))
                targets = np.reshape(target, (self.number_of_outputs, 1))
                inputs = np.reshape(input, (self.number_of_inputs, 1))
                output_errors = np.subtract(targets, net_outputs)
                E_p = np.sum(np.square(output_errors)) / 2 # privremena greska
                cumulative_error = cumulative_error + E_p
                if E_p > self.max_error:
                    real_iter = real_iter + 1
                    # Odredjivanje gradijenta za tezine od skrivenog sloja prema izlazu
                    # net_outputs = self.sigmoid_vectorized(net_outputs)
                    gradients = self.drel_vectorized(net_outputs)
                    gradients = np.multiply(gradients, output_errors) # if there is errors, look here!
                    gradients = gradients * self.learning_rate
                    # Odredjivanje delti za tezine od skrivenog sloja prema izlazu
                    hidden_to_out_deltas = np.dot(gradients, np.transpose(hidden_output))
                    self.weight_matrix_ho = np.add(self.weight_matrix_ho, hidden_to_out_deltas)
                    self.bias_out = np.add(self.bias_out, gradients)

                    # Odredjivanje gresaka za skriveni sloj
                    hidden_to_out_transposed = np.transpose(self.weight_matrix_ho)
                    hidden_errors = np.dot(hidden_to_out_transposed, output_errors)
                    # Odredjivanje gradijenta za skriveni sloj
                    hidden_gradient = self.dsigmoid_vectorized(hidden_output)
                    hidden_gradient = np.multiply(hidden_gradient, hidden_errors)
                    hidden_gradient = hidden_gradient * self.learning_rate
                    # Odredjivanje delti za ulazne tezine
                    input_transposed = np.transpose(inputs)
                    input_to_hidden_deltas = np.dot(hidden_gradient, input_transposed)
                    self.weight_matrix_ih = np.add(self.weight_matrix_ih, input_to_hidden_deltas)
                    self.bias_hidden = np.add(self.bias_hidden, hidden_gradient)
            if cumulative_error < self.max_error or iter > 50000:
                break
        return net_outputs

    
    def write_weights_to_csv(self):
        if not os.path.exists('weights'):
            os.makedirs('weights')
        df_for_ih = pd.DataFrame(data = self.weight_matrix_ih.astype(float))
        df_for_ih.to_csv('weights/ih_matrix.csv', index = False, header = False, float_format = '%.10f')
        df_for_ho = pd.DataFrame(data = self.weight_matrix_ho.astype(float))
        df_for_ho.to_csv('weights/ho_matrix.csv', index = False, header = False, float_format = '%.10f')
        df_for_hb = pd.DataFrame(data = self.bias_hidden.astype(float))
        df_for_hb.to_csv('weights/bias_hidden_matrix.csv', index = False, header = False, float_format = '%.10f')
        df_for_ob = pd.DataFrame(data = self.bias_out.astype(float))
        df_for_ob.to_csv('weights/bias_out_matrix.csv', index = False, header = False, float_format = '%.10f')

    
    def load_weights_and_biases(self):
        self.weight_matrix_ih = np.loadtxt(open("weights/ih_matrix.csv", "rb"), delimiter = ",")
        self.weight_matrix_ho = np.loadtxt(open("weights/ho_matrix.csv", "rb"), delimiter = ",")
        self.bias_hidden = np.reshape(np.loadtxt(open("weights/bias_hidden_matrix.csv", "rb"), delimiter = ","), (self.hidden_layer_count, 1))
        self.bias_out = np.reshape(np.loadtxt(open("weights/bias_out_matrix.csv", "rb"), delimiter = ","), (self.number_of_outputs, 1))
    
    
    def sigmoid_vectorized(self, vector_to_squish):
        out = np.empty((len(vector_to_squish), 1))
        i = 0
        for o in vector_to_squish:
            out[i] = sigmoid(o)
            i = i + 1
        return out
    
    def rel_vectorized(self, vector_to_squish):
        out = np.empty((len(vector_to_squish), 1))
        i = 0
        for o in vector_to_squish:
            out[i] = rel(o)
            i = i + 1
        return out
    
    def dsigmoid_vectorized(self, vector_to_desquish):
        out = np.empty((len(vector_to_desquish), 1))
        i = 0
        for o in vector_to_desquish:
            out[i] = dsigmoid(o)
            i = i + 1
        return out

    def drel_vectorized(self, vector_to_desquish):
        out = np.empty((len(vector_to_desquish), 1))
        i = 0
        for o in vector_to_desquish:
            out[i] = drel(o)
            i = i + 1
        return out
