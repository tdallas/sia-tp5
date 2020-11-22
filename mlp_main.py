from mlp import Mlp
from fonts import get_inputs, get_outputs
import numpy as np

network = Mlp(size_layers=[35, 55, 35],
              act_funct='relu')

network.train(get_inputs(), get_outputs())

print(np.array([get_inputs()[5]]).T)

prediction = network.predict(np.array(
    [get_inputs()[5]]))

print(prediction.T)
