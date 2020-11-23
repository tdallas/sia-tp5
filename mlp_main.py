from mlp import Mlp
from fonts import get_inputs, get_outputs
import numpy as np

# etas = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.005, 0.001, 0.0001]
# etas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
etas_iterations = []

etas = [0.05]

for eta in etas:
    accuracy_predictions = [0] * len(get_inputs())
    for i in range(15):
        print('nueva iteración', i)
        network = Mlp(size_layers=[35, 15, 2, 15, 35],
                      act_funct='relu', reg_lambda=eta)
        network.train(get_inputs(), get_outputs())
        for index_inputs, inputs in enumerate(get_inputs()):
            prediction = network.predict(np.array([inputs]))[0]
            count_ok = 0
            for index, num in enumerate(inputs):
                if abs(prediction[index] - num) < 0.5:
                    count_ok += 1
            print('Accuracy', count_ok / 35 * 100)
            accuracy_predictions[index_inputs] += (count_ok / 35 * 100)
        print('[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] =',network.predict(np.array([[0, 0, 0, 0, 0, 0, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])))
    accuracy_predictions = list(map(lambda x: x / 15, accuracy_predictions))
    etas_iterations.append(accuracy_predictions)

print(etas_iterations)

outF = open("etas_iterations.txt", "w")
for index, iterations in enumerate(etas_iterations):
    outF.write('eta: ')
    outF.write(etas[index])
    outF.writelines(iterations)
outF.close()
