import numpy as np

from lsy_drone_racing.control.AsyncControl import AsyncControl
from lsy_drone_racing.control.buffered_diffusion import Controller as DiffusionController

class AsyncDiffusion(AsyncControl):
    def __init__(self, initial_info, initial_obs, mpc_config, *args, **kwargs):
        super().__init__(ratio=mpc_config['ratio'], *args, **kwargs)

        # set up controller
        DiffusionController(initial_obs=initial_obs, initial_info=initial_info)

    def compute_control(self, obs, info):
        obs = np.concatenate([obs['pos'], obs['vel'], obs['rpy'], obs['ang_vel']])

        return {'actions': actions}
