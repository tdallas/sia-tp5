from mlp import Mlp
from fonts import get_inputs, get_outputs
import numpy as np

network = Mlp(size_layers=[35, 25, 12, 2, 12, 25, 35],
              act_funct='relu')

network.train(get_inputs(), get_outputs())

prediction = network.predict(np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

print(prediction)
