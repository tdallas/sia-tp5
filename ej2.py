import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from circunference_point_generator import get_points

input_len = len(get_points()[0])
output_len = input_len

inputs = get_points()
outputs = inputs

ae = MLP([input_len, 25, 15, 9, 3], 2, [3, 9, 15, 25, output_len], activation='tanh',
         solver='bfgs', eta=0.01, max_iterations=1000, adapt_eta=False, verbose=True, iteration_minimize=1000)

ae.train(inputs, outputs)

for i in range(0, 50, 10):
    prediction = ae.predict(np.array([inputs[i]]))
    matrix_of_points = np.array(prediction[0]).reshape(21, 2)
    matrix_of_points_input = np.array(inputs[i]).reshape(21, 2)
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(matrix_of_points_input[:, 0], matrix_of_points_input[:, 1])
    plt.title('Input', fontsize = 15)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.subplot(1,2,2)
    plt.title('Predicted', fontsize = 15)
    plt.scatter(matrix_of_points[:, 0], matrix_of_points[:, 1])
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.show()

for i in range(100):
    prediction = ae.predict(np.array([inputs[i]]))
    matrix_of_points = np.array(prediction[0]).reshape(21, 2)
    plt.scatter(matrix_of_points[:, 0], matrix_of_points[:, 1])
plt.ylabel('Eje x', fontsize=16)
plt.xlabel('Eje y', fontsize=16)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.tight_layout()
plt.show()

latent_space_1 = np.array([-0.8, 0.8])
prediction_1 = ae.decode(latent_space_1)
prediction_1 = np.array(prediction_1).reshape(21, 2)
latent_space_2 = np.array([0.7, -0.7])
prediction_2 = ae.decode(latent_space_2)
prediction_2 = np.array(prediction_2).reshape(21, 2)
latent_space_3 = np.array([0, 0])
prediction_3 = ae.decode(latent_space_3)
prediction_3 = np.array(prediction_3).reshape(21, 2)
latent_space_4 = np.array([0.2, -0.2])
prediction_4 = ae.decode(latent_space_4)
prediction_4 = np.array(prediction_4).reshape(21, 2)


plt.figure()
plt.subplot(2,2,1)
plt.imshow(latent_space_1.reshape(2,1), 'gray_r')
plt.title('Latent Space', fontsize = 15)
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
plt.scatter(prediction_1[:, 0], prediction_1[:, 1])
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title('Predicted', fontsize = 15)
plt.subplot(2,2,3)
plt.imshow(latent_space_2.reshape(2,1), 'gray_r')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,4)
plt.scatter(prediction_2[:, 0], prediction_2[:, 1])
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.show()

plt.figure()
plt.subplot(2,2,1)
plt.imshow(latent_space_3.reshape(2,1), 'gray_r')
plt.title('Latent Space', fontsize = 15)
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
plt.scatter(prediction_3[:, 0], prediction_3[:, 1])
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title('Predicted', fontsize = 15)
plt.subplot(2,2,3)
plt.imshow(latent_space_4.reshape(2,1), 'gray_r')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,4)
plt.scatter(prediction_4[:, 0], prediction_4[:, 1])
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.show()