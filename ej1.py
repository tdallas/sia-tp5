import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from fonts import get_input, get_output

train_x = get_input(2)
train_y = get_output(2)

ae = MLP([35, 29, 25], 5, [25, 29, 35], activation='tanh',
         solver='bfgs', eta=0.001, max_iterations=10000, adapt_eta=False, verbose=True)
ae.train(train_x, train_x)

for i in range(10):
    i = np.random.randint(train_x.shape[0])
    prediction = ae.predict(train_x[i].reshape(-1, 35))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(train_x[i].reshape(7, 5), 'gray_r')
    plt.title("Input Letter: " + train_y[i], fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(prediction.reshape(7, 5), 'gray_r')
    plt.title('Predicted', fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# latent_space_1 = np.array([-0.8, 0.8])
# prediction_1 = ae.decode(latent_space_1)
# latent_space_2 = np.array([0.7, -0.7])
# prediction_2 = ae.decode(latent_space_2)

# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(latent_space_1.reshape(2,1), 'gray_r')
# plt.title('Latent Space', fontsize = 15)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(2,2,2)
# plt.imshow(prediction_1.reshape(7,5), 'gray_r')
# plt.title('Predicted', fontsize = 15)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(2,2,3)
# plt.imshow(latent_space_2.reshape(2,1), 'gray_r')
# plt.xticks([])
# plt.yticks([])
# plt.subplot(2,2,4)
# plt.imshow(prediction_2.reshape(7,5), 'gray_r')
# plt.xticks([])
# plt.yticks([])
# plt.show()
