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

import hydra
import dill
from lsy_drone_racing.control.diffusion.base_workspace import BaseWorkspace
from lsy_drone_racing.control.diffusion.pytorch_util import dict_apply
from lsy_drone_racing.control.utils import to_local_obs, to_global_action


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
        self.initial_obs = initial_obs
        self.initial_info = initial_info

        self.n_obs_steps = 1
        self.n_latency_steps = 0

        self.device = torch.device('cuda:0')
        checkpoint = 'models/diffusion/latest.ckpt'
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

        self.action_buffer = list()


    def compute_control(
        self, obs: dict, info: dict | None = None
    ) -> npt.NDarray[np.floating]:
        n_actions = 20
        if not len(self.action_buffer):
            if __debug__:
                state = np.hstack([obs['pos'], obs['vel'], obs['rpy'], obs['ang_vel']])
                actions = self.compute_horizon(obs, 15)[:, :, :n_actions]

                self.state_history.append(state)
                self.action_history.append(actions)
            else:
                actions = self.compute_horizon(obs)

            self.action_buffer += [action for action in actions[0, :, :n_actions].T]

        return self.action_buffer.pop(0)


    def compute_horizon(self, obs: dict, samples=1) -> npt.NDarray[np.floating]:
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

        state = np.tile(state, (samples, 1, 1))

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
        actions = np_action_dict['action'].swapaxes(1,2)

        if local:
            actions = to_global_action(actions, obs['rpy'], obs['pos'])

        return actions # (B, S, T)

    def episode_reset(self):
        if __debug__:
            self.save_episode("output/logs/sim_diffusion.npz")
        self.action_history = []
        self.state_history = []

    def save_episode(self, file):
        states = np.array(self.state_history)
        actions = np.array(self.action_history)

        np.savez(file, states=states, actions=actions)

    def episode_learn(self):
        pass

    def reset(self):
        pass