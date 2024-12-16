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

from munch import munchify
import yaml

import time

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.control.control_process import ControlProcess
import multiprocessing as mp


class Controller(BaseController):
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
        super().__init__(initial_obs, initial_info)

        config_path = "config/mpc.yaml"
        with open(config_path, "r") as file:
            config = munchify(yaml.safe_load(file))

        # Save environment and control parameters.
        self.CTRL_FREQ = initial_info['env_freq']
        self.CTRL_TIMESTEP = 1 / self.CTRL_FREQ
        self.config = config

        # diffusion (torch) only works with spawn, MPC (casadi) only works with fork
        if config.controller == 'diffusion':
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass

        self._tick = 0
        initial_info['step'] = self._tick - config.n_actions

        self.ctrl = ControlProcess(initial_obs=initial_obs, initial_info=initial_info, config=config, daemon=False)
        self.ctrl.start()

        # start precomputing first actions
        self.ctrl.put_obs(obs=initial_obs, info=initial_info, block=False)

        # wait for first actions to be computed
        self.ctrl.wait_tasks()


    def compute_control(
        self, obs: dict, info: dict | None = None
    ) -> npt.NDArray[np.floating]:
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
        info['step'] = self._tick

        # set funky body_rate obs to zero (good guesstimate)
        obs['ang_vel'] *= np.pi / 180 # = np.zeros(3)  # TODO: fix!

        # only put new obs and retrieve action to minimize control delay
        self.ctrl.put_obs(obs, info, block=False)

        action, step_idx = self.ctrl.get_action(block=True, timeout=self.CTRL_TIMESTEP * self.config.wait_time_ratio)
        assert self._tick == step_idx, f'Action was provided for step {step_idx}, should be {self._tick}'

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

    def episode_callback(self):
        # use this function to plot episode data instead of learning
        # file_path = "output/states.npz"
        # self.save_episode(file_path)
        # history = np.load(file_path)
        # plot_3d(history)

        # self.async_ctrl.join(timeout=2)
        pass

    def reset(self):
        pass