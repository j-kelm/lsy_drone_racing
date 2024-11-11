import numpy as np

from lsy_drone_racing.control.AsyncControl import AsyncControl
from lsy_drone_racing.control.mpc.mpc_control import MPCControl


class AsyncMPC(AsyncControl):
    def __init__(self, initial_info, initial_obs, mpc_config, *args, **kwargs):
        super().__init__(ratio=mpc_config['ratio'], *args, **kwargs)

        # set up controller
        self.ctrl = MPCControl(initial_info, initial_obs, mpc_config)

    def compute_control(self, obs, info):
        pos = obs['pos']
        rpy = obs['rpy']
        vel = obs['vel']
        body_rates = obs['ang_vel']
        obs = np.concatenate([pos, vel, rpy, body_rates])

        inputs, states, outputs = self.ctrl.compute_control(obs, info['reference'], info)

        out = dict()

        target_pos = outputs[:3]
        target_vel = outputs[3:6]
        target_acc = outputs[6:9]
        target_yaw = outputs[11:12]
        target_body_rates = outputs[12:15]

        out['inputs'] = outputs[-4:].T - inputs.T/2
        out['actions'] = np.vstack([target_pos, target_vel, target_acc, target_yaw, target_body_rates]).T
        out['outputs'] = outputs.T
        out['states'] = states.T

        return out