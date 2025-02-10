"""Controller used for testing different coordinate frames in actions and observations.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import BaseController

if TYPE_CHECKING:
    from numpy.typing import NDArray

from lsy_drone_racing.control.utils import state_from_dict


class TrajectoryController(BaseController):
    """Controller that executes a predetermined sequence of actions to test coordinate frames in actions and observations."""

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], initial_info: dict):
        """Initialization of the controller.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """

        self.results_dict = {
            'horizon_states': list(),
            'horizon_actions': list(),
        }

        self._tick = 0

        super().__init__(initial_obs, initial_info)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """

        if self._tick < 100:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      0,
                      0, 0, 0,]
        elif self._tick < 105:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      0,
                      4, 0, 0,]
        elif self._tick < 200:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      0,
                      0, 0, 0,]
        elif self._tick < 205:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      0,
                      0, 4, 0,]
        elif self._tick < 300:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      0,
                      0, 0, 0,]
        elif self._tick < 305:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      0,
                      0, 0, 4,]
        elif self._tick < 400:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      0,
                      0, 0, 0,]
        elif self._tick < 500:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      np.pi/2,
                      0, 0, 0,]
        elif self._tick < 505:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      np.pi/2,
                      4, 0, 0,]
        elif self._tick < 600:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      np.pi/2,
                      0, 0, 0,]
        elif self._tick < 605:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      np.pi/2,
                      0, 4, 0,]
        
        else:
            action = [1, 1, 0.2,
                      0, 0, 0,
                      0, 0, 0,
                      np.pi/2,
                      0, 0, 0,]
            
        obs = state_from_dict(obs)
        self.results_dict['horizon_states'].append(obs)    
        action = np.array(action)
        self.results_dict['horizon_actions'].append(action)
        
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """Increment the time step counter."""
        self._tick += 1

    def episode_reset(self):
        """Reset the time step counter."""
        self._tick = 0

    def episode_callback(self):
        np.savez_compressed("output/logs/mpc.npz", **self.results_dict)
        print(f'[WORK] Saved controller logs to: {"output/logs/mpc.npz"}')
