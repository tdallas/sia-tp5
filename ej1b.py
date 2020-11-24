import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from fonts import get_input, get_output, add_noise

train_x = get_input(2)
train_y = get_output(2)

train_x_with_noise = add_noise(train_x, 2)
test_x_with_noise = add_noise(train_x, 2)

ae = MLP([35, 25], 20, [25, 35], activation='tanh', solver='bfgs', eta=0.01, max_iterations=200, adapt_eta=False, verbose=True)
ae.train(train_x_with_noise, train_x)

for i in range(32):
    prediction_1 = ae.predict(train_x_with_noise[i].reshape(-1, 35))
    prediction_2 = ae.predict(train_x[i].reshape(-1, 35))
    prediction_3 = ae.predict(test_x_with_noise[i].reshape(-1, 35))

    plt.figure()
    plt.subplot(3,2,1)
    plt.imshow(train_x_with_noise[i].reshape(7,5), 'gray_r')
    plt.title("Train Input noise", fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,2,2)
    plt.imshow(prediction_1.reshape(7,5), 'gray_r')
    plt.title('Predicted noise', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,2,3)
    plt.imshow(train_x[i].reshape(7,5), 'gray_r')
    plt.title("Input no noise", fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,2,4)
    plt.imshow(prediction_2.reshape(7,5), 'gray_r')
    plt.title('Predicted no noise', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,2,5)
    plt.imshow(test_x_with_noise[i].reshape(7,5), 'gray_r')
    plt.title("Test Input no noise", fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,2,6)
    plt.imshow(prediction_3.reshape(7,5), 'gray_r')
    plt.title('Predicted no noise', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.show()