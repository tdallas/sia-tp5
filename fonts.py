from bitstring import BitArray
import pandas as pd
import numpy as np

df = pd.read_csv('fonts_2', delimiter="\n", header=None, dtype=str)

data = np.array(df).reshape(32, 7)

inputs_matrix_bits = data
outputs_matrix_bits = data

new_data = []
for i in range(len(data)):
    new_data.append("".join(np.squeeze(np.asarray(data[i]))))

for i in range(len(new_data)):
    new_data[i] = list(new_data[i])
    new_data[i] = [0 if j == '0' else int(j) for j in new_data[i]]

def get_inputs():
    return np.asarray(new_data)

print(new_data)

def get_outputs():
    return np.asarray(new_data)
