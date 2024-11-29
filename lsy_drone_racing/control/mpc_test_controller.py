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
from numpy._typing import NDArray

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.control.mpc.mpc_control import MPCControl

from lsy_drone_racing.control.mpc.planner import MinsnapPlanner



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

        path = "config/mpc.yaml"
        with open(path, "r") as file:
            mpc_config = munchify(yaml.safe_load(file))

        # Save environment and control parameters.
        self.CTRL_FREQ = initial_info['env_freq']
        self.CTRL_TIMESTEP = 1 / self.CTRL_FREQ
        self.initial_obs = initial_obs
        self.initial_info = initial_info
        self.config = mpc_config

        self.initial_info['nominal_physical_parameters'] = mpc_config.drone_params

        self.episode_reset()

        self.planner = MinsnapPlanner(initial_info=self.initial_info,
                                      initial_obs=self.initial_obs,
                                      speed=mpc_config.planner.speed,
                                      gate_time_constant=mpc_config.planner.gate_time_const,
                                      )
        self.mpc_ctrl = MPCControl(initial_info=initial_info,initial_obs=initial_obs, config=mpc_config)

        if p.isConnected():
            for i in range(self.planner.ref.shape[1]-10):
                if not i % 10:
                    p.addUserDebugLine(self.planner.ref[0:3, i], self.planner.ref[0:3, i+10], lineColorRGB=[1,0,0])


    def compute_control(
        self, obs: dict, info: dict | None = None
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
        obs['ang_vel'] *= np.pi/180
        actions = self.compute_horizon(obs, info)

        return actions[:, 2]

    def compute_horizon(self, obs: dict, info: dict):
        info['step'] = self._tick
        info['reference'] = self.planner.ref
        info['gate_prox'] = self.planner.gate_prox

        obs = np.concatenate([obs['pos'], obs['vel'], obs['rpy'], obs['ang_vel']])
        inputs, states, outputs = self.mpc_ctrl.compute_control(obs, info['reference'], info)

        target_pos = outputs[:3]
        target_vel = outputs[3:6]
        target_acc = outputs[6:9]
        target_yaw = outputs[11:12]
        target_body_rates = outputs[12:15]

        actions = np.vstack([target_pos, target_vel, target_acc, target_yaw, target_body_rates])

        return actions

    def episode_reset(self):
        print('episode_reset')
        self._tick = 1

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: NDArray[np.floating],
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
        print('episode_callback')
        pass

    def reset(self):
        print('reset')
        pass