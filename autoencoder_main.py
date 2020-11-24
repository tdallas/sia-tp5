import numpy as np
import matplotlib.pyplot as plt

from autoencoder import AutoEncoder
from fonts import get_input, get_output, add_noise

#Ejercicio 1 a

train_x = get_input(2)
train_y = get_output(2)

# ae = AutoEncoder([20, 10], 2, [10, 20], activation='tanh', solver='lbfgs', eta=0.0001, max_iterations=30000, tol=0.0000001, verbose=True)
# ae.fit(train_x, train_x)

# i = np.random.randint(train_x.shape[0])
# latent_space = ae.encode(train_x[i])
# prediction = ae.decode(latent_space)

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(train_x[i].reshape(7,5), 'gray_r')
# plt.title("Input Letter: " + train_y[i], fontsize = 15)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1,3,2)
# plt.imshow(latent_space.reshape(2,1), 'gray_r')
# plt.title('Latent Space', fontsize = 15)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1,3,3)
# plt.imshow(prediction.reshape(7,5), 'gray_r')
# plt.title('Predicted', fontsize = 15)
# plt.xticks([])
# plt.yticks([])
# plt.show()


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

# for i in range(len(train_y)):
#     latent_space = ae.encode(train_x[i])
#     plt.scatter(latent_space[0], latent_space[1], label=train_y[i])
# plt.xlabel('Z1', fontsize=16)
# plt.ylabel('Z2', fontsize=16)
# plt.xlim(-1.1, 1.1)
# plt.ylim(-1.1, 1.1)
# plt.tight_layout()
# plt.show()

# nx = 10
# ny = 10
# x_values = np.linspace(-1, 1, nx)
# y_values = np.linspace(-1, 1, ny)
# canvas = np.empty((7*ny, 5*nx))

# for i, yi in enumerate(y_values):
#     for j, xi in enumerate(x_values):
#         latent = np.array([xi, yi])
#         reconst = ae.decode(latent)
#         canvas[(nx-i-1)*7:(nx-i)*7,j*5:(j+1)*5] = reconst.reshape(7, 5)

# plt.imshow(canvas, 'gray_r')
# plt.xlabel('Z1', fontsize = 16)
# plt.ylabel('Z2', fontsize = 16)
# plt.xticks([])
# plt.yticks([])
# plt.show()


#Ejercicio 1 b

train_x_with_noise = add_noise(train_x, 2)
test_x_with_noise = add_noise(train_x, 2)

ae = AutoEncoder([25, 15], 10, [10, 25], activation='tanh', solver='lbfgs', eta=0.0001, max_iterations=30000, tol=0.0000001, verbose=True)
ae.fit(train_x_with_noise, train_x)

for i in range(32):
    prediction_1 = ae.predict(train_x_with_noise[i])
    prediction_2 = ae.predict(train_x[i])
    prediction_3 = ae.predict(test_x_with_noise[i])

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


#Ejercicio 2

