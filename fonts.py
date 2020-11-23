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
    return np.array(new_data)

def get_outputs():
    return np.asarray(new_data)

def concat_and_return(array):
    array.append(1)
    return array

def get_inputs_with_bias():
    return np.array(list(map(lambda x: concat_and_return(x), new_data)))

# print(get_inputs_with_bias())
