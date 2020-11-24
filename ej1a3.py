import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from fonts import get_input, get_output

train_x = get_input(2)
train_y = get_output(2)

ae = MLP([35, 20], 2, [20, 35], activation='tanh', solver='bfgs', eta=0.01, max_iterations=10000, adaptive_eta=False, with_bias=True, verbose=True)
ae.train(train_x, train_x)

for i in range(32):
    A, Z = ae.feedforward(train_x[i].reshape(-1,35))
    plt.scatter(A[ae.get_latent_layer_position()-1][0][1], A[ae.get_latent_layer_position()-1][0][2], label=train_y[i])
plt.ylabel('Z1', fontsize=16)
plt.xlabel('Z2', fontsize=16)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.tight_layout()
plt.show()