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

import hydra
import dill
from lsy_drone_racing.control.diffusion.base_workspace import BaseWorkspace
from lsy_drone_racing.control.diffusion.pytorch_util import dict_apply
from lsy_drone_racing.control.utils import to_local_obs, to_global_action


class HorizonDiffusion:
    """Template controller class."""

    def __init__(self, initial_obs: dict, initial_info: dict, config: dict | None = None):
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

        self.device = torch.device(config.diffusion.device)
        checkpoint = 'models/diffusion/latest.ckpt'
        output_dir = 'output/diffusion_eval_output'

        # load checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cfg['policy']['num_inference_steps'] = 5
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

        torch.manual_seed(initial_info['run_id'])

    def compute_horizon(self, obs: dict, info: dict, samples=1) -> npt.NDArray[np.floating]:
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

        actions = self.sample_actions(state, samples)
        if local:
            actions = to_global_action(actions, obs['rpy'], obs['pos'])

        return actions

    def sample_actions(self, obs, samples=1):
        state = np.tile(obs, (samples, 1, 1))

        # create obs dict
        np_obs_dict = {
            # handle n_latency_steps by discarding the last n_latency_steps
            'obs': state.astype(np.float32),
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
        actions = np_action_dict['action'].swapaxes(1, 2)
        return actions  # (B, S, T)

    def reset(self):
        pass

    @property
    def unwrapped(self):
        return self