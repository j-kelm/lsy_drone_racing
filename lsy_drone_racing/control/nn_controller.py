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

        self.model = NeuralNetwork(19, hidden_size=250)
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

        pos = obs['pos']
        rpy = obs['rpy']
        vel = obs['vel']
        ang_vel = obs['ang_vel']
        gate_index = info['target_gate']
        obstacles_pos = np.hstack(info['obstacles.pos'])
        gates_pos = np.hstack(info['gates.pos'])
        gates_rpy = np.hstack(info['gates.rpy'])

        # Transform observation into local frame (h, vel, rp, body_rates)
        local_obs = to_local_obs(pos=pos,
                                 vel=vel,
                                 rpy=rpy,
                                 ang_vel=ang_vel,
                                 obstacles_pos=obstacles_pos,
                                 gates_pos=gates_pos,
                                 gates_rpy=gates_rpy,
                                 target_gate=gate_index,
                                 )

        obs = torch.tensor(local_obs, device=self.device, dtype=torch.float32)

        if len(self.state_history):
            draw_segment_of_traj(self.initial_info, self.state_history[-1][0:3], pos, [0, 1, 0, 1])

        local_actions = self.model(obs).detach().cpu().numpy()
        local_actions = local_actions.reshape(13, -1)
        local_action = local_actions

        # Transform action back to global frame
        global_action = to_global_action(local_action, rpy, pos)
        back_to_local = to_local_action(global_action, rpy, pos)

        diff = local_action - back_to_local
        print(diff)


        return global_action[0, :, 0]

    def episode_reset(self):
        self.action_history = []
        self.state_history = []

    def save_episode(self, file):
        pass

    def episode_learn(self):
        pass

    def reset(self):
        pass