from typing import Dict
import torch
import numpy as np

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class DroneLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            obs_key='obs',
            action_key='action',
            ):
        super().__init__()
        self.obs_key = obs_key
        self.action_key = action_key
        self.horizon = horizon

        self.data = np.load(zarr_path, allow_pickle=True)
        self.data = {'action': self.data[action_key], 'obs': self.data[self.obs_key]}
        self.data = dict_apply(self.data, torch.from_numpy)

        self.data['action'] = self.data['action'][:, :, :self.horizon].swapaxes(1, 2)
        self.data['obs'] = self.data['obs'][:, None, :]

    def get_validation_dataset(self):
        raise NotImplementedError

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(data=self.data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.data['action'])

    def __len__(self) -> int:
        return len(self.data['action'])

    def _sample_to_data(self, sample):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch_data = {
            'obs': self.data['obs'][idx],
            'action': self.data['action'][idx],
        }
        return torch_data