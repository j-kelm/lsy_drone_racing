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

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.utils.utils import draw_segment_of_traj

import hydra
import dill
from lsy_drone_racing.control.diffusion.base_workspace import BaseWorkspace
from lsy_drone_racing.control.diffusion.pytorch_util import dict_apply

from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace

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

        self.n_obs_steps = 1
        self.n_latency_steps = 0

        self.device = torch.device('cuda:0')
        checkpoint = 'models/diffusion/drone/checkpoints/latest.ckpt'
        output_dir = 'output/diffusion_eval_output'

        # load checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model

        self.policy.to(self.device)
        self.policy.eval()

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
        body_rates = obs['ang_vel']
        state = np.hstack([pos, vel, rpy, body_rates, info['target_gate'], np.hstack(info['obstacles.pos']), np.hstack(info['gates.pos']), np.hstack(info['gates.rpy'])])

        if len(self.state_history):
            draw_segment_of_traj(self.initial_info, self.state_history[-1][0:3], pos, [0, 1, 0, 1])

        # create obs dict
        np_obs_dict = {
            # handle n_latency_steps by discarding the last n_latency_steps
            'obs': state.astype(np.float32).reshape((1, 1, -1)),
        }

        # device transfer
        obs_dict = dict_apply(np_obs_dict,
                              lambda x: torch.from_numpy(x).to(
                                  device=self.device))

        # run policy
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)

        # device_transfer
        np_action_dict = dict_apply(action_dict,
                                    lambda x: x.detach().to('cpu').numpy())

        # handle latency_steps, we discard the first n_latency_steps actions
        # to simulate latency
        action = np_action_dict['action'][:, 0, :].flatten()

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