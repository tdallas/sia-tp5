import numpy as np
import math
from scipy.optimize import minimize
from fonts import get_inputs, get_inputs_with_bias


class Mlp():
    def __init__(self, size_layers, act_funct='relu', reg_lambda=0.9, with_bias=True):
        '''
        Constructor method. Defines the characteristics of the MLP
        Arguments:
            size_layers : List with the number of Units for:
                [Input, Hidden1, Hidden2, ... HiddenN, Output] Layers.
            act_funtc   : Activation function for all the Units in the MLP
                default = 'relu'
            reg_lambda: Value of the regularization parameter Lambda
                lambda = 0, i.e. no regularization
            bias: Indicates is the bias element is added for each layer, but the output
        '''
        self.size_layers = size_layers
        self.n_layers = len(size_layers)
        self.act_f = act_funct
        self.eta = reg_lambda
        self.with_bias = with_bias
        self.prev_error = None

        # Ramdomly initialize theta (MLP weights)
        self.initialize_theta_weights()

    def train(self, X, Y, iterations=1000, reset=True):
        '''
        Given X (feature matrix) and y (class vector)
        Updates the Theta Weights by running Backpropagation N tines
        Arguments:
            X          : Feature matrix [n_examples, n_features]
            Y          : Sparse class matrix [n_examples, classes]
            iterations : Number of times Backpropagation is performed
                default = 400
            reset      : If set, initialize Weights before training
                default = False
        '''
        if reset:
            self.initialize_theta_weights()
        for i in range(iterations):
            self.gradients = self.backpropagation(X, Y)
            self.gradients_vector = self.unroll_weights(self.gradients)
            self.theta_vector = self.unroll_weights(self.weights)
            self.theta_vector = self.theta_vector - self.gradients_vector
            self.weights = self.roll_weights(self.theta_vector)
            self.eta = self.adapt_eta()

    def adapt_eta(self):
        inputs_to_calculate_error = get_inputs()[:8]
        error = 0
        for inputs in inputs_to_calculate_error:
            error_calculated = self.calculate_error([inputs], [inputs])
            error += error_calculated
        error = error / len(inputs_to_calculate_error)
        # print('error', error)
        if self.prev_error is None:
            self.prev_error = error
            return self.eta + (error * 0.000001)
        else:
            if self.prev_error < error:
                self.prev_error = error
                return self.eta - (error * 0.000001)
            elif self.prev_error == error:
                self.prev_error = error
                return 0.9
            else:
                self.prev_error = error
                return self.eta + (error * 0.000001)

    def predict(self, X):
        '''
        Given X (feature matrix), y_hay is computed
        Arguments:
            X      : Feature matrix [n_examples, n_features]
        Output:
            y_hat  : Computed Vector Class for X
        '''
        # print('llamando a predict con', X)
        A, Z = self.feedforward(X)
        # print('A', A[-1])
        Y_hat = A[-1]
        return Y_hat

    def initialize_theta_weights(self):
        '''
        Initialize weights, initialization method depends
        on the Activation Function and the Number of Units in the current layer
        and the next layer.
        The weights for each layer as of the size [next_layer, current_layer + 1]
        '''
        self.weights = []
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            epsilon = np.sqrt(2.0 / (size_layer * size_next_layer))
            # Weigts from Normal distribution mean = 0, std = epsion
            if self.with_bias:
                theta_tmp = epsilon * \
                    (np.random.randn(size_next_layer, size_layer + 1))
            else:
                theta_tmp = epsilon * \
                    (np.random.randn(size_next_layer, size_layer))
            self.weights.append(theta_tmp)
        return self.weights

    def backpropagation(self, X, Y):
        '''
        Implementation of the Backpropagation algorithm with regularization
        '''
        def g_dz(x): return self.relu_derivative(x)

        n_examples = X.shape[0]
        # Feedforward
        A, Z = self.feedforward(X)

        # Backpropagation
        deltas = [None] * self.n_layers
        deltas[-1] = A[-1] - Y
        # For the second last layer to the second one
        for ix_layer in np.arange(self.n_layers - 1 - 1, 0, -1):
            theta_tmp = self.weights[ix_layer]
            if self.with_bias:
                # Removing weights for bias
                theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
            deltas[ix_layer] = (np.matmul(theta_tmp.transpose(
            ), deltas[ix_layer + 1].transpose())).transpose() * g_dz(Z[ix_layer])

        # Compute gradients
        gradients = [None] * (self.n_layers - 1)
        for ix_layer in range(self.n_layers - 1):
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
        return gradients

    def feedforward(self, X):
        '''
        Implementation of the Feedforward
        '''
        def g(x): return self.relu(x)

        A = [None] * self.n_layers
        Z = [None] * self.n_layers
        input_layer = X

        for ix_layer in range(self.n_layers - 1):
            n_examples = input_layer.shape[0]
            if self.with_bias:
                # Add bias element to every example in input_layer
                input_layer = np.concatenate(
                    (np.ones([n_examples, 1]), input_layer), axis=1)
            A[ix_layer] = input_layer
            # Multiplying input_layer by weights for this layer
            # print('np.shape(input_layer)', np.shape(input_layer))
            # print('np.shape(self.weights[ix_layer])', np.shape(self.weights[ix_layer]))
            Z[ix_layer + 1] = np.matmul(input_layer,
                                        self.weights[ix_layer].transpose())
            # Activation Function
            output_layer = g(Z[ix_layer + 1])
            # Current output_layer will be next input_layer
            input_layer = output_layer

        A[self.n_layers - 1] = output_layer
        return A, Z

    def calculate_error(self, test_data, test_exp):
        guesses = [self.predict(np.array([i])) for i in test_data]
        # print('guesses', guesses)
        return np.sum(
            [(np.subtract(test_exp[i], guesses[i]) ** 2).sum()
             for i in range(len(test_exp))]
        ) / len(test_data)

    def cost(self, flat_weights, test_data, test_exp):
        # print('flat_weights', flat_weights)
        # print('test_data', test_data)
        # print('test_exp', test_exp)
        return self.calculate_error(test_data, test_exp)

    def cost_denoise(self):
        return ''

    def train_minimize(self):
        prev_w = self.unroll_weights(self.weights)
        minimized_weights = minimize(fun=self.cost, x0=prev_w, args=(
            get_inputs(), get_inputs()), method='SLSQP')
        after_w = self.roll_weights(minimized_weights.x)
        # print('prev_w', self.weights, 'after_w', after_w)
        self.weights = after_w

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
        size_next_layers = self.size_layers.copy()
        size_next_layers.pop(0)
        rolled_list = []
        if self.with_bias:
            extra_item = 1
        else:
            extra_item = 0
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            n_weights = (size_next_layer * (size_layer + extra_item))
            data_tmp = unrolled_data[0: n_weights]
            data_tmp = data_tmp.reshape(
                size_next_layer, (size_layer + extra_item), order='F')
            rolled_list.append(data_tmp)
            unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])
        return rolled_list

    def relu(self, z):
        return self.tanh(z)
        '''
        Rectified Linear function
        z can be an numpy array or scalar
        '''
        if np.isscalar(z):
            result = np.max((z, 0))
        else:
            zero_aux = np.zeros(z.shape)
            meta_z = np.stack((z, zero_aux), axis=-1)
            result = np.max(meta_z, axis=-1)
        return result

    def relu_derivative(self, z):
        return self.tanh_deriv(z)
        '''
        Derivative for Rectified Linear function
        z can be an numpy array or scalar
        '''
        result = 1 * (z > 0)
        return result

    def tanh(self, x):
        return np.tanh(0.3 * x)

    def tanh_deriv(self, x):
        return 0.3 * (1 - np.tanh(x) ** 2)

    def logistic(self, x):
        return 1 / (1 + np.exp(-2 * 0.3 * x))

    def logistic_deriv(self, x):
        act = self.logistic(x)
        return 2 * 0.3 * act * (1 - act)
