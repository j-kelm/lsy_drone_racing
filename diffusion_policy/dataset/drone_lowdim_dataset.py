from typing import Dict
import torch
import numpy as np
from torch.utils.data import TensorDataset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class DictLowdimDataset(BaseLowdimDataset):
    def __init__(self, obs, actions):#
        super().__init__()

        assert len(obs) == len(actions)
        self.data = {'obs': obs, 'action': actions}

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(data=self.data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_validation_dataset(self):
        raise NotImplementedError

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.data['action'])

    def __len__(self) -> int:
        return len(self.data['action'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch_data = {
            'obs': self.data['obs'][idx],
            'action': self.data['action'][idx],
        }
        return torch_data

class DroneLowdimDataset(DictLowdimDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 obs_key='obs',
                 action_key='action',
                 val_percentage=0.15,
                 ):


        obs_key = obs_key
        action_key = action_key

        data = np.load(zarr_path, allow_pickle=True)
        obs = torch.from_numpy(data[obs_key])
        actions = torch.from_numpy(data[action_key])

        assert len(obs) == len(actions)
        assert 0 < horizon <= actions.shape[2]

        actions = actions[:, :, :horizon].swapaxes(1, 2)
        obs = obs[:, None, :]

        assert 0 <= val_percentage < 1
        train_val_cut = int((1 - 0.15) * len(obs))

        train_obs = obs[:train_val_cut]
        train_actions = actions[:train_val_cut]

        val_obs = obs[train_val_cut:]
        val_actions = actions[train_val_cut:]

        self.val_set = DictLowdimDataset(obs=val_obs, actions=val_actions)
        super().__init__(obs=train_obs, actions=train_actions)

    def get_validation_dataset(self):
        return self.val_set