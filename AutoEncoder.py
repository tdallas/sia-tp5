import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS

class AutoEncoder():
    def __init__(self, encoder_layer_sizes, latent_layer_size, decoder_layer_sizes, activation, solver, eta=0.0001, max_iterations=30000, tol=0.0000001, verbose=False):
        self.encoder_layer_sizes = encoder_layer_sizes
        self.latent_layer_size = latent_layer_size
        self.decoder_layer_sizes = decoder_layer_sizes
        self.encoder_layer = len(encoder_layer_sizes)
        self.latent_layer = self.encoder_layer + 1
        self.decoder_layer = self.latent_layer + len(decoder_layer_sizes) + 1
        self.activation = activation
        self.solver = solver
        self.eta = eta
        self.max_iterations = max_iterations
        aux = encoder_layer_sizes
        aux.append(latent_layer_size)
        self.hidden_layer_sizes = aux + decoder_layer_sizes
        self.mlp = MLPRegressor(hidden_layer_sizes = self.hidden_layer_sizes, 
                   activation = activation, 
                   solver = solver, 
                   learning_rate_init = eta, 
                   max_iter = max_iterations, 
                   tol = 0.0000001, 
                   verbose = verbose)

    def fit(self, training_input, training_output):
        return self.mlp.fit(training_input, training_output)

    def predict(self, data):
        return self.mlp.predict(data.reshape(-1,len(self.mlp.coefs_[0])))

    def encode(self, data):
        return self.get_data_from_layer(data, self.latent_layer, 0)

    def decode(self, data):
        return self.get_data_from_layer(data, self.decoder_layer, self.latent_layer)

    def get_data_from_layer(self, data, to_layer, from_layer=0):
        L = ACTIVATIONS[self.activation](np.matmul(data, self.mlp.coefs_[from_layer]) + self.mlp.intercepts_[from_layer])
        from_layer += 1
        if from_layer >= to_layer or from_layer >= len(self.mlp.coefs_):
            return L
        else:
            return self.get_data_from_layer(L, to_layer, from_layer=from_layer)

    def get_loss_curve(self):
        return self.mlp.loss_curve_
