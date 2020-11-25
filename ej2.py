import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from circunference_point_generator import get_points

input_len = len(get_points()[0])
output_len = input_len

inputs = get_points()
outputs = inputs

print(input_len)

ae = MLP([input_len], 25, [output_len], activation='tanh',
         solver='bfgs', eta=0.1, max_iterations=3000, adapt_eta=False, verbose=True, iteration_minimize=2)

ae.train(inputs, outputs)

prediction = ae.predict(np.array([inputs[0]]))

print('prediction', prediction[0])
print('actual', inputs[0])

matrix_of_points = np.array(prediction[0]).reshape(21, 2)
print('matrix_of_points', matrix_of_points)

print('xs', matrix_of_points[:, 0])
print('ys',  matrix_of_points[:, 1])

plt.scatter(matrix_of_points[:, 0], matrix_of_points[:, 1])
plt.title('eta=0.1, it=1950, std=0.65, latent=25, it_min=2')
plt.ylabel('Eje x', fontsize=16)
plt.xlabel('Eje y', fontsize=16)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.tight_layout()
plt.show()
