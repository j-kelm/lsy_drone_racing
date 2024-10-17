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

import torch

from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils.utils import draw_trajectory, draw_segment_of_traj

from examples.nn_model import NeuralNetwork


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
        self.initial_obs = initial_obs
        self.initial_info = initial_info

        self.device = 'cpu' # 'cuda:0'

        self.model = NeuralNetwork()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load("output/modality.pth", weights_only=True))
        self.model.eval()

        self.state_history = []
        self.action_history = []


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

        pos = obs[0:3]
        rpy = obs[3:6]
        vel = obs[6:9]
        body_rates = obs[9:12]
        state = np.hstack([pos, vel, rpy, body_rates, info['target_gate'], np.hstack(info['obstacles.pos']), np.hstack(info['gates.pos']), np.hstack(info['gates.rpy'])])
        obs = torch.tensor(state, device=self.device, dtype=torch.float32)

        if len(self.state_history):
            draw_segment_of_traj(self.initial_info, self.state_history[-1][0:3], pos, [0, 1, 0, 1])

        actions = self.model(obs).detach().cpu().numpy()
        actions = actions.reshape(13, 5).T
        action = actions[0]

        self.state_history.append(state)
        self.action_history.append(action)

        return action

    def episode_reset(self):
        self.action_history = []
        self.state_history = []

    def save_episode(self, file):
        pass

    def episode_learn(self):
        pass

    def reset(self):
        pass