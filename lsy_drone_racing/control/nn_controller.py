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
from wandb.cli.cli import local

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.utils.utils import draw_segment_of_traj

from lsy_drone_racing.control.nn.nn_model import NeuralNetwork
from lsy_drone_racing.control.utils import to_local_obs, to_global_action, to_local_action


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

        self.device = 'cuda:0'

        self.model = NeuralNetwork(23, hidden_size=300)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load("output/modality.pth", weights_only=True))
        self.model.eval()

        self.state_history = []
        self.action_history = []


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

        if __debug__:
            state = np.hstack([obs['pos'], obs['vel'], obs['rpy'], obs['ang_vel']])
            actions = self.compute_horizon(obs)

            self.state_history.append(state)
            self.action_history.append(actions)
        else:
            actions = self.compute_horizon(obs)

        return actions[:, 1]

    def compute_horizon(self, obs: dict) -> npt.NDarray[np.floating]:
        local = True
        if local:
            state = to_local_obs(pos=obs['pos'],
                                 vel=obs['vel'],
                                 rpy=obs['rpy'],
                                 ang_vel=obs['ang_vel'],
                                 obstacles_pos=obs['obstacles_pos'].T,
                                 gates_pos=obs['gates_pos'].T,
                                 gates_rpy=obs['gates_rpy'].T,
                                 target_gate=obs['target_gate'],
                                 )
        else:
            pos = obs['pos']
            vel = obs['vel']
            rpy = obs['rpy']
            body_rates = obs['ang_vel']
            state = np.hstack([pos, vel, rpy, body_rates, obs['target_gate'], np.hstack(obs['obstacles_pos']), np.hstack(obs['gates_pos']), np.hstack(obs['gates_rpy'])]).reshape((1, 1, -1))

        state = torch.tensor(state, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            actions = self.model(state).detach().cpu().numpy()
        actions = actions.reshape(14, -1)

        if local:
            actions = to_global_action(actions, obs['rpy'], obs['pos'])

        return actions[0]

    def episode_reset(self):
        self.action_history = []
        self.state_history = []

    def save_episode(self, file):
        pass

    def episode_learn(self):
        pass

    def reset(self):
        pass