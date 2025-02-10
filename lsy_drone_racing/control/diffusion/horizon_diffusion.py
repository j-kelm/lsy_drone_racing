"""Receding horizon controller using a diffusion policy for computing action sequences.
"""

from __future__ import annotations  # Python 3.10 type hints

import time

import numpy as np
import numpy.typing as npt

import torch

import hydra
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from lsy_drone_racing.control.utils import to_local_obs, to_global_action, state_from_dict


class HorizonDiffusion:
    """Class to compute action sequences using a diffusion policy."""

    def __init__(self, initial_obs: dict, initial_info: dict):
        """Initialization of the controller.

        Prepare results dict and load model. Set seed of diffusion policy.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Augmented environment information also containing the controller config.
        """
        config = initial_info['config']

        self.n_actions = config['n_actions']
        self.offset = config['offset']
        self.n_samples = config['diffusion']['n_samples']

        self.device = torch.device(config.diffusion.device)
        self.logs = True
        self.results_dict = {'horizon_states': [],
                             'horizon_actions': [],
                             't_wall': [],
                             't_solver': [],
                             'horizon_samples': [],
                             'gates_pos': initial_obs['gates_pos'],
                             'gates_rpy': initial_obs['gates_rpy'],
                             'obstacles_pos': initial_obs['obstacles_pos'],
                             'env_freq': initial_info['env_freq'],
                             }

        checkpoint = 'models/diffusion/latest.ckpt'
        output_dir = 'output/diffusion_eval_output'

        # load checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cfg['policy']['num_inference_steps'] = config['diffusion']['n_inference_steps']
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

        if 'run_id' in initial_info:
            torch.manual_seed(initial_info['run_id'])
        else:
            torch.manual_seed(config['diffusion']['seed'])

        self.results_dict['seed'] = torch.random.initial_seed()
        print(f"Seed: {self.results_dict['seed']}")

    def compute_horizon(self, obs: dict, info: dict) -> npt.NDArray[np.floating]:
        """Compute action sequence from diffusion policy.

        :param obs: Observation dict.
        :param info: Augmented info dict containing config.
        :return: Action sequence.
        """

        # start timer
        start_t = time.perf_counter()

        # transform into local frame
        state = to_local_obs(pos=obs['pos'],
                             vel=obs['vel'],
                             rpy=obs['rpy'],
                             ang_vel=obs['ang_vel'],
                             obstacles_pos=obs['obstacles_pos'].T,
                             gates_pos=obs['gates_pos'].T,
                             gates_rpy=obs['gates_rpy'].T,
                             target_gate=obs['target_gate'],
                             )
        
        start_solve_t = time.perf_counter()
        samples = self.sample_actions(state, self.n_samples)
        end_solve_t = time.perf_counter()

        samples = to_global_action(samples, obs['rpy'], obs['pos'])

        # Find action most similar to last action
        if len(self.results_dict['horizon_actions']) and self.n_samples > 1:
            differences = samples[:, :, :self.n_actions] - self.results_dict['horizon_actions'][-1][:, self.offset:self.offset+self.n_actions]

            # normalize differences
            differences_max = differences.max(axis=0)
            differences_min = differences.min(axis=0)
            differences_scaled = (differences - differences_min)/(differences_max - differences_min + 1.0e-8)
            differences_norm = np.linalg.norm(differences_scaled, axis=(1,2))

            closest_index = np.argmin(differences_norm)
            actions = samples[closest_index]
        else:
            actions = samples[0]

        end_t = time.perf_counter()
        if self.logs:
            self.results_dict['horizon_states'].append(state_from_dict(obs)[:, None])
            self.results_dict['horizon_actions'].append(actions)
            self.results_dict['horizon_samples'].append(samples)
            self.results_dict['t_wall'].append(end_t - start_t)
            self.results_dict['t_solver'].append(end_solve_t - start_solve_t)

        return actions

    def sample_actions(self, obs, n_samples=1):
        """ Batch sample from the diffusion policy.

        :param obs: Observation dict.
        :param n_samples: Size of action batch to be sampled.
        :return: Action sequences (n_samples, State, Timestep)
        """

        state = np.tile(obs, (n_samples, 1, 1))

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

        actions = np_action_dict['action'].swapaxes(1, 2)
        return actions  # (B, S, T)

    def reset(self):
        pass

    @property
    def unwrapped(self):
        return self