from __future__ import annotations

from abc import abstractmethod

import numpy as np
import numpy.typing as npt

from lsy_drone_racing.control import BaseController

class BufferedController(BaseController):
    """
    Provide a receding horizon controller using a buffered output


    """
    def __init__(self, initial_obs: dict[str, npt.NDArray[np.floating]], initial_info: dict, n_actions: int = 1, n_init_actions: int = 1, offset: int = 0):
        super().__init__(initial_obs=initial_obs, initial_info=initial_info)

        self.action_buffer = list()
        self.n_actions = n_init_actions
        self.offset = offset

        self.compute_control(initial_obs, initial_info)
        self.n_actions = n_actions


    def compute_control(
        self, obs: dict, info: dict | None = None

    ) -> npt.NDArray[np.floating]:
        if not len(self.action_buffer):
            obs['ang_vel'] *= np.pi / 180  # TODO: fix
            actions = self.compute_horizon(obs, info).squeeze()
            self.action_buffer += [action for action in actions[:, self.offset:self.n_actions+self.offset].T]

        action = self.action_buffer.pop(0)
        return action

    @abstractmethod
    def compute_horizon(self, obs: dict, info: dict) -> npt.NDArray[np.floating]:
        raise NotImplementedError