import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from circunference_point_generator import get_points

input_len = len(get_points()[0])
output_len = input_len

inputs = get_points()
outputs = inputs

print(input_len)

ae = MLP([input_len, 40, 35, 30], 25, [30, 35, 40, output_len], activation='tanh',
         solver='bfgs', eta=0.01, max_iterations=5000, adapt_eta=False, verbose=True, iteration_minimize=5)

ae.train(inputs, outputs)

prediction = ae.predict(np.array([inputs[0]]))

print('prediction', prediction[0])
print('actual', inputs[0])

matrix_of_points = np.array(prediction[0]).reshape(21, 2)
print('matrix_of_points', matrix_of_points)

print('xs', matrix_of_points[:, 0])
print('ys',  matrix_of_points[:, 1])

plt.scatter(matrix_of_points[:, 0], matrix_of_points[:, 1])
plt.title('muchas capas, eta=0.01, it=5000, std=0.71, latent=25, it_min=5')
plt.ylabel('Eje x', fontsize=16)
plt.xlabel('Eje y', fontsize=16)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.tight_layout()
plt.show()
