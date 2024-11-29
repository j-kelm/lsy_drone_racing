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

import time

import numpy as np
import numpy.typing as npt

from numpy._typing import NDArray

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.control.mpc_test_controller import Controller as MPC_Controller
from lsy_drone_racing.control.diffusion_controller import Controller as Diffusion_Controller
from lsy_drone_racing.control.utils import to_local_obs



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

        self.mpc = MPC_Controller(initial_obs=initial_obs, initial_info=initial_info)
        self.diffusion = Diffusion_Controller(initial_obs=initial_obs, initial_info=initial_info)

        self.mpc_actions = list()
        self.diffusion_actions = list()
        self.states = list()


    def compute_control(
        self, obs: dict, info: dict | None = None,
    ) -> npt.NDarray[np.floating]:
        obs['ang_vel'] *= np.pi/180 # TODO: fix

        state = to_local_obs(pos=obs['pos'],
                             vel=obs['vel'],
                             rpy=obs['rpy'],
                             ang_vel=obs['ang_vel'],
                             obstacles_pos=obs['obstacles_pos'].T,
                             gates_pos=obs['gates_pos'].T,
                             gates_rpy=obs['gates_rpy'].T,
                             target_gate=obs['target_gate'],
                             )

        t_start = time.perf_counter()
        diffusion_actions = self.diffusion.compute_horizon(obs, 15)
        t_diffusion = time.perf_counter()
        mpc_actions = self.mpc.compute_horizon(obs, info)
        t_mpc = time.perf_counter()

        print(f'MPC time: {t_mpc - t_diffusion} Diffusion time: {t_diffusion - t_start}')

        self.states.append(state)
        self.mpc_actions.append(mpc_actions)
        self.diffusion_actions.append(diffusion_actions)

        # return diffusion_actions[0, :, 2]
        return mpc_actions[:, 2]

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        self.mpc.step_callback(action, obs, reward, terminated, truncated, info)

    def episode_reset(self):
        if __debug__:
            self.save_episode("output/logs/sim_diffusion.npz")
        self.mpc_actions = list()
        self.diffusion_actions = list()
        self.states = list()

    def save_episode(self, file):
        states = np.array(self.states)
        mpc_actions = np.array(self.mpc_actions)
        diffusion_actions = np.array(self.diffusion_actions)

        np.savez(file, states=states, mpc_actions=mpc_actions, diffusion_actions=diffusion_actions)

    def episode_learn(self):
        pass

    def reset(self):
        pass