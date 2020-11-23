from mlp import Mlp
from fonts import get_inputs, get_outputs
import numpy as np

etas = [0.1]
etas_iterations = []

for eta in etas:
    accuracy_predictions = [0] * len(get_inputs())
    for i in range(15):
        network = Mlp(size_layers=[35, 30, 20, 18, 10, 6, 2, 6, 10, 18, 20, 30, 35], reg_lambda=eta)
        network.train(get_inputs(), get_outputs())
        print('llamando al minimize')
        network.mimize_weights_error()
        print('despues del train minimize')
        for index_inputs, inputs in enumerate(get_inputs()):
            prediction = network.predict(np.array([inputs]))[0]
            count_ok = 0
            for index, num in enumerate(inputs):
                if abs(prediction[index] - num) < 0.5:
                    count_ok += 1
            print('Accuracy', count_ok / 35 * 100)
            accuracy_predictions[index_inputs] += (count_ok / 35 * 100)
    accuracy_predictions = list(map(lambda x: x / 50, accuracy_predictions))
    etas_iterations.append(accuracy_predictions)

print(etas_iterations)

outF = open("etas_iterations.txt", "w")
for index, iterations in enumerate(etas_iterations):
    outF.write('eta: ')
    outF.write(etas[index])
    outF.writelines(iterations)
outF.close()