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

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import numpy.typing as npt

import pybullet as p

from munch import munchify
import yaml

from lsy_drone_racing.control.mpc.mpc_control import MPCControl
from lsy_drone_racing.control.mpc.mpc_utils import outputs_for_actions

from lsy_drone_racing.control.mpc.planner import MinsnapPlanner
from lsy_drone_racing.control.utils import obs_from_dict


class HorizonMPC:
    """Template controller class."""

    def __init__(self, initial_obs: dict, initial_info: dict):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        config = initial_info['config']

        self.planner = MinsnapPlanner(initial_info=initial_info,
                                      initial_obs=initial_obs,
                                      speed=config.mpc.planner.speed,
                                      gate_time_constant=config.mpc.planner.gate_time_const,
                                      )


        self.mpc_ctrl = MPCControl(initial_info=initial_info,initial_obs=initial_obs)

        if p.isConnected():
            for i in range(self.planner.ref.shape[1]-10):
                if not i % 10:
                    p.addUserDebugLine(self.planner.ref[0:3, i], self.planner.ref[0:3, i+10], lineColorRGB=[1,0,0])


    def compute_horizon(self, obs: dict, info: dict) -> npt.NDArray[np.floating]:
        obs = obs_from_dict(obs)

        info['reference'] = self.planner.ref
        info['gate_prox'] = self.planner.gate_prox

        horizons = self.mpc_ctrl.compute_control(obs, info['reference'], info)

        outputs = horizons['outputs']
        actions = outputs[outputs_for_actions]

        return actions

    def reset(self):
        self.mpc_ctrl.reset()

    @property
    def unwrapped(self):
        return self.mpc_ctrl.unwrapped