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

from pathlib import Path

import numpy as np
import numpy.typing as npt

from munch import munchify
import yaml

from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.wrapper import ObsWrapper

from examples.planner import Planner, FilePlanner, LinearPlanner, PointPlanner
from examples.mpc_controller import MPC
from examples.model import Model


class Controller(BaseController):
    """Template controller class."""

    def __init__(self, initial_obs: npt.NDArray[np.floating], initial_info: dict):
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
        super().__init__(initial_obs, initial_info)

        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.initial_info = initial_info

        self.episode_reset()

        path = "config/mpc.yaml"
        with open(path, "r") as file:
            config = munchify(yaml.safe_load(file))

        # initialize planner
        self.planner = LinearPlanner(initial_info=initial_info, CTRL_FREQ=self.CTRL_FREQ)
        self.ref = self.planner.plan(initial_obs=self.initial_obs,
                                     gates=None,
                                     speed=1.0)

        # initialize mpc controller
        self.model = Model(info=None)
        self.ctrl = MPC(model=self.model, horizon=int(config.mpc.horizon_sec * self.CTRL_FREQ),
                                   q_mpc=config.mpc.q, r_mpc=config.mpc.r)

        self.state = None
        self.state_history = []
        self.action_history = []

        # TODO: draw reference

        self._take_off = False
        self._setpoint_land = False
        self._land = False

    def compute_control(
        self, obs: npt.NDArray[np.floating], info: dict | None = None
    ) -> npt.NDarray[np.floating]:
        """Compute the next desired position and orientation of the drone.

        INSTRUCTIONS:
            Re-implement this method to return the target pose to be sent from Crazyswarm to the
            Crazyflie using the `cmdFullState` call.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone pose [x_des, y_des, z_des, yaw_des] as a numpy array.
        """

        remaining_ref = self.ref[:, info['step']:]
        self.state = obs[:12]

        action, next_state = self.ctrl.select_action(obs=self.state, info={"ref": remaining_ref})
        target_pos = next_state[:3]
        target_vel = next_state[3:6]

        target_yaw = next_state[8]

        # calculate target acc
        y = np.array(self.model.symbolic.g_func(x=next_state, u=action)['g']).flatten()
        target_acc = y[6:9]
        target_rpy_rates = y[12:15]

        self.state_history.append(self.state)
        self.action_history.append(action)

        action = np.hstack([target_pos, target_vel, target_acc, target_yaw, target_rpy_rates])

        return action

    def episode_reset(self):
        self.action_history = []
        self.state_history = []

    def episode_learn(self):
        pass

    def reset(self):
        pass
