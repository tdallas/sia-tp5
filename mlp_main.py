from mlp import Mlp
from fonts import get_inputs, get_outputs
import numpy as np

network = Mlp(size_layers = [35, 100, 35], 
                         act_funct   = 'relu',
                         reg_lambda  = 0,
                         bias_flag   = False)

network.train(get_inputs(), get_outputs())

prediction = network.predict(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

print(prediction)
