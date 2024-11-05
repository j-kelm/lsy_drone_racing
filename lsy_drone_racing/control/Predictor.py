import numpy as np
from copy import deepcopy

from lsy_drone_racing.control.mpc.model import PredictionModel, DeltaModel
from lsy_drone_racing.control.mpc.mpc_utils import rk_discrete


class SymbolicPredictor:
    def __init__(self, info):
        self.model = PredictionModel(info=info)

        self.dynamics_func = rk_discrete(self.model.symbolic.fc_func,
                                         self.model.symbolic.nx,
                                         self.model.symbolic.nu,
                                         self.model.dt)

    def predict(self, obs, info, inputs):
        x = obs
        if len(inputs):
            inputs = np.atleast_2d(inputs)
            info = deepcopy(info)

            for u in inputs:
                x = self.dynamics_func(x0=x, p=u)['xf']
                info['step'] += 1

        return np.array(x).flatten(), info