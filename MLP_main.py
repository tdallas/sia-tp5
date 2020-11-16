from sklearn.neural_network import MLPClassifier
from bitstring import BitArray
import numpy as np
from multilayer_perceptron import Mlp
from fonts import get_inputs, get_outputs

clf = Mlp()

inputs_matrix_bits = []

clf.fit(np.array(get_inputs()), np.array(get_outputs()))

print(clf.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
