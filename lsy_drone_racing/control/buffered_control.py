"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations


import numpy as np
import numpy.typing as npt
import yaml
from munch import munchify

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.control.diffusion.horizon_diffusion import HorizonDiffusion
from lsy_drone_racing.control.mpc.horizon_mpc import HorizonMPC

class BufferedController(BaseController):
    """
    Provide a receding horizon controller using a buffered output


    """
    def __init__(self, initial_obs: dict[str, npt.NDArray[np.floating]], initial_info: dict):
        super().__init__(initial_obs=initial_obs, initial_info=initial_info)

        config_path = "config/mpc.yaml"
        with open(config_path, "r") as file:
            config = munchify(yaml.safe_load(file))

        base_controller = config.controller

        if base_controller == 'diffusion':
            self.ctrl = HorizonDiffusion(initial_obs, initial_info, config)
        elif base_controller == 'mpc':
            self.ctrl = HorizonMPC(initial_obs, initial_info, config)
        else:
            raise RuntimeError(f'Controller type {base_controller} not supported!')

        self.action_buffer = list()
        self.offset = config.buffered.offset

        self._tick = 0
        initial_info['step'] = self._tick

        # compute first with n_initial_actions, then set back
        self.n_actions = config.buffered.n_initial_actions
        self.compute_control(initial_obs, initial_info)
        self.n_actions = config.buffered.n_actions


    def compute_control(
        self, obs: dict, info: dict | None = None

    ) -> npt.NDArray[np.floating]:
        if not len(self.action_buffer):
            obs['ang_vel'] *= np.pi / 180  # TODO: fix
            info['step'] = self._tick
            actions = self.ctrl.compute_horizon(obs, info).squeeze()
            self.action_buffer += [action for action in actions[:, self.offset:self.n_actions+self.offset].T]

        action = self.action_buffer.pop(0)
        return action

    def step_callback(
        self,
        action: npt.NDArray[np.floating],
        obs: npt.NDArray[np.floating],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        self._tick += 1

    @property
    def unwrapped(self):
        return self.ctrl.unwrapped