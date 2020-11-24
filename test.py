import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.neural_network._base import ACTIVATIONS
from fonts import get_input, get_output

train_x = get_input(2)
train_y = get_output(2)

# idx = np.random.randint(train_x.shape[0])
# img = train_x[idx].reshape(7,5)

# plt.figure(figsize = (7,5))
# plt.imshow(img,'gray_r')
# plt.title("Letter : {}".format(train_y[idx]))
# plt.xticks([])
# plt.yticks([])
# plt.show()

# Shape of input and latent variable

n_input = 5*7

# Encoder structure
n_encoder1 = 20
n_encoder2 = 10

n_latent = 2

# Decoder structure
n_decoder2 = 10
n_decoder1 = 20

reg = MLPRegressor(hidden_layer_sizes = (n_encoder1, n_encoder2, n_latent, n_decoder2, n_decoder1), 
                   activation = 'tanh', 
                   solver = 'lbfgs', 
                   learning_rate_init = 0.0001, 
                   max_iter = 30000, 
                   tol = 0.0000001, 
                   verbose = True)

reg.fit(train_x, train_x)

idx = np.random.randint(train_x.shape[0])
x_reconst = reg.predict(train_x[idx].reshape(-1,35))
print(x_reconst)

# plt.figure(figsize = (10,8))
# plt.subplot(1,2,1)
# plt.imshow(train_x[idx].reshape(7,5), 'gray_r')
# plt.title("Input Letter : {}".format(train_y[idx]), fontsize = 15)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1,2,2)
# plt.imshow(x_reconst.reshape(7,5), 'gray_r')
# plt.title('Predicted', fontsize = 15)
# plt.xticks([])
# plt.yticks([])
# plt.show()

def get_data_from_layer(data, MLP, to_layer, from_layer=0):
    L = ACTIVATIONS['tanh'](np.matmul(data, MLP.coefs_[from_layer]) + MLP.intercepts_[from_layer])
    from_layer += 1
    if from_layer >= to_layer:
        return L
    else:
        return get_data_from_layer(L, MLP, to_layer, from_layer=from_layer)

L = get_data_from_layer(train_x[idx], reg, to_layer=6, from_layer=0)
print(L)


#GENERA DESDE LA CAPA LATENTE ACA
L = get_data_from_layer([0.5, 0.5], reg, to_layer=6, from_layer=3)
print(L)

