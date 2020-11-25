import numpy as np
import math
from scipy.optimize import minimize

def relu(z):
    if np.isscalar(z):
        result = np.max((z, 0))
    else:
        zero_aux = np.zeros(z.shape)
        meta_z = np.stack((z, zero_aux), axis=-1)
        result = np.max(meta_z, axis=-1)
    return result

def relu_derivative( z):
    return 1 * (z > 0)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return (1 - np.tanh(z) ** 2)

def logistic(z):
    return 1 / (1 + np.exp(-2 * 0.3 * z))

def logistic_derivative(z):
    act = logistic(z)
    return 2 * 0.3 * act * (1 - act)


activation_functions = {
    "tanh": tanh,
    "logistic": logistic,
    "relu": relu
}

derivative_activation_functions = {
    "tanh": tanh_derivative,
    "logistic": logistic_derivative,
    "relu": relu_derivative
}

class MLP():
    def __init__(self, encoder_layer_sizes, latent_layer_size, decoder_layer_sizes, activation, solver, eta=0.0001, max_iterations=30000, iteration_minimize=20, adapt_eta=False, with_bias=True, verbose=False):
        self.encoder_layer = len(encoder_layer_sizes)
        self.latent_layer = self.encoder_layer + 1
        self.decoder_layer = self.latent_layer + len(decoder_layer_sizes) + 1
        aux = encoder_layer_sizes
        aux.append(latent_layer_size)
        self.layer_sizes = aux + decoder_layer_sizes
        self.n_layers = len(self.layer_sizes)
        self.activation = activation_functions[activation]
        self.derivate_activation = derivative_activation_functions[activation]
        self.solver = solver
        self.max_iterations = max_iterations
        self.eta = eta
        self.with_bias = with_bias
        self.adapt_eta = adapt_eta
        self.verbose = verbose
        self.prev_error = None
        self.iteration_minimize = iteration_minimize
        self.weights = self.initialize_weights()

    def train(self, training_input, training_output, reset=True):
        if reset:
            self.initialize_weights()
        for i in range(self.max_iterations):
            if self.verbose:
                print("Iteration: " + str(i))
            self.gradients = self.backpropagation(training_input, training_output)
            self.gradients_vector = self.unroll_weights(self.gradients)
            self.theta_vector = self.unroll_weights(self.weights)
            self.theta_vector = self.theta_vector - self.gradients_vector
            self.weights = self.roll_weights(self.theta_vector)
            if self.adapt_eta:
                self.eta = self.adapt_eta_f(training_input, training_output)
            if i % self.iteration_minimize == 0:
                self.mimize_weights_error(training_input, training_output)

    def adapt_eta_f(self, input_t, output_t):
        error = 0
        for inputs in input_t:
            error_calculated = self.calculate_error(input_t, output_t)
            error += error_calculated
        error = error / len(input_t)
        if self.prev_error is None:
            self.prev_error = error
            return self.eta + (error * 0.000001)
        else:
            if self.prev_error < error:
                self.prev_error = error
                return self.eta - (error * 0.000001)
            elif self.prev_error == error:
                self.prev_error = error
                return 0.01
            else:
                self.prev_error = error
                return self.eta + (error * 0.000001)

    def predict(self, input_t):
        A, Z = self.feedforward(input_t)
        Y_hat = A[-1]
        return Y_hat

    def initialize_weights(self):
        '''
        Initialize weights, initialization method depends
        on the Activation Function and the Number of Units in the current layer
        and the next layer.
        The weights for each layer as of the size [next_layer, current_layer + 1]
        '''
        self.weights = []
        size_next_layers = self.layer_sizes.copy()
        size_next_layers.pop(0)
        for size_layer, size_next_layer in zip(self.layer_sizes, size_next_layers):
            epsilon = 0.71
            # Weigts from Normal distribution mean = 0, std = epsion
            if self.with_bias:
                theta_tmp = epsilon * (np.random.normal(size= (size_next_layer, size_layer + 1), scale=epsilon))
            else:
                theta_tmp = epsilon * \
                    (np.random.randn(size_next_layer, size_layer))
            self.weights.append(theta_tmp)
        return self.weights

    def backpropagation(self, input_t, output_t):

        n_examples = input_t.shape[0]
        # Feedforward
        A, Z = self.feedforward(input_t)

        # Backpropagation
        deltas = [None] * self.n_layers
        deltas[-1] = A[-1] - output_t
        # For the second last layer to the second one
        for ix_layer in np.arange(self.n_layers - 1 - 1, -1, -1):
            theta_tmp = self.weights[ix_layer]
            if self.with_bias:
                # Removing weights for bias
                theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
            if ix_layer == 0:
                deltas[ix_layer] = np.matmul(np.matmul(theta_tmp.transpose(), deltas[ix_layer+1].transpose()), self.derivate_activation(Z[ix_layer])) * self.eta
            else:
                deltas[ix_layer] = (np.matmul(theta_tmp.transpose(), deltas[ix_layer + 1].transpose())).transpose() * self.derivate_activation(Z[ix_layer]) * self.eta

        # Compute gradients
        gradients = [None] * (self.n_layers - 1)
        for ix_layer in range(self.n_layers - 1):
            # print('ix layer segunda iteracion', ix_layer)
            grads_tmp = np.matmul(
                deltas[ix_layer + 1].transpose(), A[ix_layer])
            grads_tmp = grads_tmp / n_examples
            if self.with_bias:
                # Regularize weights, except for bias weigths
                grads_tmp[:, 1:] = grads_tmp[:, 1:] + \
                    (self.eta / n_examples) * \
                    self.weights[ix_layer][:, 1:]
            else:
                # Regularize ALL weights
                grads_tmp = grads_tmp + \
                    (self.eta / n_examples) * self.weights[ix_layer]
            gradients[ix_layer] = grads_tmp
        # print('gradients', gradients)
        return gradients

    def feedforward(self, input_t):
        A = [None] * self.n_layers
        Z = [None] * self.n_layers
        input_layer = input_t
        for ix_layer in range(self.n_layers - 1):
            n_examples = input_layer.shape[0]
            if self.with_bias:
                # Add bias element to every example in input_layer
                input_layer = np.concatenate(
                    (np.ones([n_examples, 1]), input_layer), axis=1)
                if ix_layer == 0:
                    Z[0] = np.matmul(input_layer, self.weights[0].T)

            A[ix_layer] = input_layer
            # Multiplying input_layer by weights for this layer
            Z[ix_layer + 1] = np.matmul(input_layer,
                                        self.weights[ix_layer].transpose())
            # Activation Function
            output_layer = self.activation(Z[ix_layer + 1])
            # Current output_layer will be next input_layer
            input_layer = output_layer
        A[self.n_layers - 1] = output_layer
        return A, Z

    def encode(self, data):
        return self.get_data_from_layer(data, self.latent_layer - 1, 0)

    def decode(self, data):
        return self.get_data_from_layer(data, self.decoder_layer, self.latent_layer - 1)

    def get_data_from_layer(self, data, to_layer, from_layer=0):
        input_layer = data
        if self.with_bias:
            input_layer = np.insert(input_layer, 0, 1)
        L = self.activation(np.matmul(input_layer, self.weights[from_layer].T))
        from_layer += 1
        if from_layer >= to_layer or from_layer >= len(self.weights):
            return L
        else:
            return self.get_data_from_layer(L, to_layer, from_layer=from_layer)

    def calculate_error(self, input_t, output_t):
        guesses = [self.predict(np.array([i])) for i in input_t]
        return np.sum(
            [(np.subtract(output_t[i], guesses[i]) ** 2).sum()
             for i in range(len(output_t))]
        ) / len(input_t)

    def cost(self, flat_weights, input_t, output_t):
        return self.calculate_error(input_t, output_t)

    def mimize_weights_error(self, input_t, output_t):
        prev_w = self.unroll_weights(self.weights)
        minimized_weights = minimize(fun=self.cost, x0=prev_w, args=(
            input_t, output_t), method=self.solver)
        after_w = self.roll_weights(minimized_weights.x)
        self.weights = after_w

    def print_weights(self, weights):
        for weight in weights:
            print(weight)

    def unroll_weights(self, rolled_data):
        '''
        Unroll a list of matrices to a single vector
        Each matrix represents the Weights (or Gradients) from one layer to the next
        '''
        unrolled_array = np.array([])
        for one_layer in rolled_data:
            unrolled_array = np.concatenate(
                (unrolled_array, one_layer.flatten("F")))
        return unrolled_array

    def roll_weights(self, unrolled_data):
        '''
        Unrolls a single vector to a list of matrices
        Each matrix represents the Weights (or Gradients) from one layer to the next
        '''
        size_next_layers = self.layer_sizes.copy()
        size_next_layers.pop(0)
        rolled_list = []
        if self.with_bias:
            extra_item = 1
        else:
            extra_item = 0
        for size_layer, size_next_layer in zip(self.layer_sizes, size_next_layers):
            n_weights = (size_next_layer * (size_layer + extra_item))
            data_tmp = unrolled_data[0: n_weights]
            data_tmp = data_tmp.reshape(
                size_next_layer, (size_layer + extra_item), order='F')
            rolled_list.append(data_tmp)
            unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])
        return rolled_list

    def get_latent_layer_position(self):
        return self.latent_layer