""" Asynchronous controller class that performs calculations in a separate process for both MPC and diffusion.
"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import numpy.typing as npt

from munch import munchify
import yaml

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.control.control_process import ControlProcess
import multiprocessing as mp


class AsyncController(BaseController):
    """ Asynchronous controller class shared for MPC and diffusion.

    This class implements a controller that spawns a control process for either MPC or diffusion to perform controller
    computations off-process. The class loads a config to determine times to wait for control outputs and spawns a
    process with diffusion or MPC controller depending on the value set in the config.
    """

    def __init__(self, initial_obs: dict, initial_info: dict):
        """Initialization of the controller and spawning control process.

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
        initial_info['config'] = config

        # diffusion (torch) only works with spawn, MPC (casadi) only works with fork
        if config.controller == 'diffusion':
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass

        self._tick = 0
        initial_info['step'] = self._tick - config.n_actions

        self.ctrl = ControlProcess(initial_obs=initial_obs, initial_info=initial_info, daemon=False)
        self.ctrl.start()

        # start precomputing first actions
        self.ctrl.put_obs(obs=initial_obs, info=initial_info, block=False)

        # wait for first actions to be computed
        self.ctrl.wait_tasks()

    def compute_control(
        self, obs: dict, info: dict | None = None
    ) -> npt.NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Store the current observation in a queue and fetch the current action from the control process.
        In case the control process did not finish computation yet, wait for an amount according to config.

        Args:
            obs: The current observation of the environment. See the environment's observation space for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, prate, qrate, rrate] as a numpy
            array.
        """
        info['step'] = self._tick

        # set funky body_rate obs to zero (good guesstimate)
        # obs['ang_vel'] *= np.pi / 180 # TODO: fix!
        # obs['ang_vel'] = np.zeros(3)

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
        pass

    def episode_reset(self):
        print('[MAIN] Joining worker...')
        self.ctrl.join()
        self.ctrl.close()
        print('[MAIN] Worker joined.')