import numpy as np

from lsy_drone_racing.control.mpc.model import PredictionModel
from lsy_drone_racing.control.mpc.mpc_utils import rk_discrete


class SymbolicPredictor:
    def __init__(self):
        self.model = PredictionModel(info=None)

        self.dynamics_func = rk_discrete(self.model.symbolic.fc_func,
                                         self.model.symbolic.nx,
                                         self.model.symbolic.nu,
                                         self.model.dt)

    def predict(self, obs, info, inputs):
        if not len(inputs):
            inputs = np.atleast_2d(self.model.U_EQ)

        x = obs
        for u in inputs:
            # x = self.dynamics_func(x0=x, p=u)['xf']
            info['step'] += 1

        return np.array(x).flatten(), info