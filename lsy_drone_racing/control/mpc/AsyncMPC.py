import numpy as np

from lsy_drone_racing.control.AsyncControl import AsyncControl
from lsy_drone_racing.control.mpc.mpc_control import MPCControl
from lsy_drone_racing.control.mpc.mpc_utils import outputs_for_actions


class AsyncMPC(AsyncControl):
    def __init__(self, initial_info, initial_obs, mpc_config, *args, **kwargs):
        super().__init__(ratio=mpc_config['ratio'], *args, **kwargs)

        # set up controller
        self.ctrl = MPCControl(initial_info, initial_obs, mpc_config)

    def compute_control(self, obs, info):
        obs = np.concatenate([obs['pos'], obs['vel'], obs['rpy'], obs['ang_vel']])
        horizons = self.ctrl.compute_control(obs, info['reference'], info)

        out = {
            'actions': horizons['outputs'][outputs_for_actions].T,
            'outputs': horizons['outputs'].T,
            'states': horizons['states'].T,
        }

        return out