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

from lsy_drone_racing.control import BaseController

from lsy_drone_racing.control.mpc.planner import MinsnapPlanner

from lsy_drone_racing.control.mpc.AsyncMPC import AsyncMPC


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
        self.async_ctrl = AsyncMPC(initial_info=initial_info,initial_obs=initial_obs, mpc_config=mpc_config, daemon=True)

        if 'step' not in self.initial_info:
            self.initial_info['step'] = -mpc_config.ratio
        self.initial_info['reference'] = self.planner.ref
        self.initial_info['gate_prox'] = self.planner.gate_prox

        # compute solutions once with unlimited time to set internal MPC state for a good warm start
        self.initial_info['step'] = 0
        self.async_ctrl.ctrl.ctrl.horizon_skip = 0
        self.async_ctrl.compute_control(self.initial_obs, self.initial_info)
        self.initial_info['step'] = -mpc_config.ratio
        self.async_ctrl.ctrl.ctrl.horizon_skip = mpc_config.ratio

        # switch back solver options BEFORE starting the solver worker
        self.async_ctrl.ctrl.ctrl.setup_optimizer(
            solver='ipopt',
            max_iter=mpc_config.max_iter,
            max_wall_time=self.CTRL_TIMESTEP * mpc_config.ratio * mpc_config.wall_time_ratio,
        )

        self.async_ctrl.start()

        # start precomputing first actions
        self.async_ctrl.put_obs(obs=self.initial_obs, info=self.initial_info, block=False)

        if p.isConnected():
            for i in range(self.planner.ref.shape[1]-10):
                if not i % 10:
                    p.addUserDebugLine(self.planner.ref[0:3, i], self.planner.ref[0:3, i+10], lineColorRGB=[1,0,0])

        # wait for first actions to be computed
        self.async_ctrl.wait_tasks()


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
        info['reference'] = self.planner.ref
        info['gate_prox'] = self.planner.gate_prox

        # set funky body_rate obs to zero (good guesstimate)
        obs['ang_vel'] = np.zeros(3)  # TODO: fix!

        # only put new obs and retrieve action to minimize control delay
        self.async_ctrl.put_obs(obs, info, block=False)

        action, step_idx = self.async_ctrl.get_action(block=True, timeout=self.CTRL_TIMESTEP * self.config.wait_time_ratio)
        assert self._tick == step_idx, f'Action was provided for step {step_idx}, should be {self._tick}'

        return action

    def episode_reset(self):
        print('episode_reset')
        self._tick = 0

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
        print('episode_callback')
        pass

    def reset(self):
        print('reset')
        pass