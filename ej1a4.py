import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from fonts import get_input, get_output

train_x = get_input(2)
train_y = get_output(2)

ae = MLP([35, 29, 17], 2, [17, 29, 35], activation='tanh',
         solver='bfgs', eta=0.01, max_iterations=200, adapt_eta=False, verbose=True)
ae.train(train_x, train_x)

latent_space_1 = np.array([-0.8, 0.8])
prediction_1 = ae.decode(latent_space_1)
latent_space_2 = np.array([0.7, -0.7])
prediction_2 = ae.decode(latent_space_2)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(latent_space_1.reshape(2,1), 'gray_r')
plt.title('Latent Space', fontsize = 15)
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
plt.imshow(prediction_1.reshape(7,5), 'gray_r')
plt.title('Predicted', fontsize = 15)
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
plt.imshow(latent_space_2.reshape(2,1), 'gray_r')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,4)
plt.imshow(prediction_2.reshape(7,5), 'gray_r')
plt.xticks([])
plt.yticks([])
plt.show()

for i in range(len(train_y)):
    latent_space = ae.encode(train_x[i])
    plt.scatter(latent_space[0], latent_space[1], label=train_y[i])
plt.xlabel('Z1', fontsize=16)
plt.ylabel('Z2', fontsize=16)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.tight_layout()
plt.show()

nx = 10
ny = 10
x_values = np.linspace(-1, 1, nx)
y_values = np.linspace(-1, 1, ny)
canvas = np.empty((7*ny, 5*nx))

for i, yi in enumerate(y_values):
    for j, xi in enumerate(x_values):
        latent = np.array([xi, yi])
        reconst = ae.decode(latent)
        canvas[(nx-i-1)*7:(nx-i)*7,j*5:(j+1)*5] = reconst.reshape(7, 5)

plt.imshow(canvas, 'gray_r')
plt.xlabel('Z1', fontsize = 16)
plt.ylabel('Z2', fontsize = 16)
plt.xticks([])
plt.yticks([])
plt.show()
