import numpy as np
import matplotlib.pyplot as plt

from autoencoder import AutoEncoder
from fonts import get_input, get_output

train_x = get_input(2)
train_y = get_output(2)

ae = AutoEncoder([20, 10], 2, [10, 20], activation='tanh', solver='lbfgs', eta=0.0001, max_iterations=30000, tol=0.0000001, verbose=True)
ae.fit(train_x, train_x)

i = np.random.randint(train_x.shape[0])
latent_space = ae.encode(train_x[i])
prediction = ae.decode(latent_space)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(train_x[i].reshape(7,5), 'gray_r')
plt.title("Input Letter: " + train_y[i], fontsize = 15)
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,2)
plt.imshow(latent_space.reshape(2,1), 'gray_r')
plt.title('Latent Space', fontsize = 15)
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,3)
plt.imshow(prediction.reshape(7,5), 'gray_r')
plt.title('Predicted', fontsize = 15)
plt.xticks([])
plt.yticks([])
plt.show()


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