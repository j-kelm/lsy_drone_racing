""" Buffered controller class that periodically executes MPC or diffusion controller and buffers actions.
"""

from __future__ import annotations


import numpy as np
import numpy.typing as npt
import yaml
from munch import munchify

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.control.diffusion.horizon_diffusion import HorizonDiffusion
from lsy_drone_racing.control.mpc.horizon_mpc import HorizonMPC

class BufferedController(BaseController):
    """
    Provide a receding horizon controller using a buffered output.

    The controller buffers the computed action sequence and applies multiple actions from that sequence before
    recomputing a new action sequence. Computations run in the same process. This controller can also be used for
    vanilla receding horizon control if only the first action from the buffer is applied. Select type of controller
    in config.
    """

    def __init__(self, initial_obs: dict[str, npt.NDArray[np.floating]], initial_info: dict):
        """Initialization of the controller.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """

        super().__init__(initial_obs=initial_obs, initial_info=initial_info)

        config_path = "config/mpc.yaml"
        with open(config_path, "r") as file:
            config = munchify(yaml.safe_load(file))

        base_controller = config.controller
        initial_info['config'] = config

        if base_controller == 'diffusion':
            self.ctrl = HorizonDiffusion(initial_obs, initial_info)
        elif base_controller == 'mpc':
            self.ctrl = HorizonMPC(initial_obs, initial_info)
        else:
            raise RuntimeError(f'Controller type {base_controller} not supported!')

        self.action_buffer = list()
        self.offset = config.offset

        self._tick = 0
        initial_info['step'] = self._tick

        # compute first with n_initial_actions, then set back
        self.n_actions = config.n_initial_actions
        self.compute_control(initial_obs, initial_info)
        self.n_actions = config.n_actions


    def compute_control(
        self, obs: dict, info: dict | None = None

    ) -> npt.NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Compute a new action sequence if the action buffer is empty, otherwise fetch next action from action buffer.

        Args:
            obs: The current observation of the environment. See the environment's observation space for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, prate, qrate, rrate] as a numpy
            array.
        """

        if not len(self.action_buffer):
            # obs['ang_vel'] *= np.pi / 180  # TODO: fix
            info['step'] = self._tick
            actions = self.ctrl.compute_horizon(obs, info).squeeze()
            self.action_buffer += [action for action in actions[:, self.offset:self.n_actions+self.offset].T]

        action = self.action_buffer.pop(0)
        return action

    def step_callback(
        self,
        action: npt.NDArray[np.floating],
        obs: npt.NDArray[np.floating],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        self._tick += 1

    @property
    def unwrapped(self):
        return self.ctrl.unwrapped