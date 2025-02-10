"""Control process to be spawned by the AsyncController for both MPC and diffusion.
"""

import multiprocessing as mp
import numpy as np
import queue

from lsy_drone_racing.control.diffusion.horizon_diffusion import HorizonDiffusion
from lsy_drone_racing.control.mpc.horizon_mpc import HorizonMPC


class ControlProcess(mp.Process):
    """Wrap a controller to fetch actions asynchronously in a separate process.

    Uses MPC or diffusion depending on configuration.
    """

    def __init__(self, initial_info, initial_obs, *args, **kwargs):
        """ Initialize the control process.

        :param initial_info:    Additional environment information from the reset, config dict.
        :param initial_obs:     The initial observation of the environment's state. See the environment's
                                observation space for details.
        :param args:            mp.Process args
        :param kwargs:          mp.Process kwargs
        """
        super().__init__(*args, **kwargs)
        config = initial_info['config']
        base_controller = config.controller

        if base_controller == 'diffusion':
            self.ctrl = HorizonDiffusion(initial_obs, initial_info)
        elif base_controller == 'mpc':
            self.ctrl = HorizonMPC(initial_obs, initial_info)
        else:
            raise RuntimeError(f'Controller type {base_controller} not supported!')

        self._obs_queue = mp.JoinableQueue()
        self._action_queue = mp.Queue()

        self._action_queue.cancel_join_thread()
        self._obs_queue.cancel_join_thread()

        self._n_actions = config.n_actions
        self._delay = config.offset

        self._logging_path = config.logging_path if 'logging_path' in config and config.logging_path != 'None' else None

        assert self._n_actions > 0, "Amount of environment steps per control step must be at least 1"
        assert self._delay >= 0, "Action delay cannot be negative"

    def put_obs(self, obs, info, *args, **kwargs):
        """ Put observation into queue.

        The control process expects the observations of all time steps and discards unnecessary observations itself.
        Entries of the dicts should be serializable for mp.Queue.

        :param obs: Observation dict.
        :param info: Info dict.
        :param args: args for mp.Queue.put().
        :param kwargs: kwargs for mp.Queue.put().
        """
        self._obs_queue.put((obs, info), *args, **kwargs)

    def get_action(self, *args, **kwargs):
        """Get next action from queue.

        :param args: args for mp.Queue.get().
        :param kwargs: kwargs for mp.Queue.get().
        """
        return self._action_queue.get(*args, **kwargs)

    def wait_tasks(self):
        """Block until all tasks in the observation queue are processed.
        """
        self._obs_queue.join()

    def run(self):
        """Run worker loop that waits for observations and computes actions.

        If no new observation is received within 1 second, the worker terminates.
        """
        while True:
            # blocking, wait until new obs is available
            try:
                obs, info = self._obs_queue.get(block=True, timeout=1) # crash self if no obs
            except queue.Empty:
                if self._logging_path is not None:
                    # write logs on termination
                    for key in self.ctrl.unwrapped.results_dict:
                        self.ctrl.unwrapped.results_dict[key] = np.array(self.ctrl.unwrapped.results_dict[key])

                    self.ctrl.unwrapped.results_dict['n_actions'] = self._n_actions
                    self.ctrl.unwrapped.results_dict['offset'] = self._delay

                    np.savez_compressed(self._logging_path, **self.ctrl.unwrapped.results_dict)
                    print(f'[WORK] Saved controller logs to: {self._logging_path}')

                return

            # compute new action every ratio steps
            if not info['step'] % self._n_actions:
                info['step'] += self._n_actions

                actions = self.ctrl.compute_horizon(obs, info).squeeze()
                assert len(actions) >= self._n_actions + self._delay, "Controller must return at least delay + n_action steps"

                actions = actions[:, self._delay:self._n_actions+self._delay].T
                for step_offset, action in enumerate(actions):
                    self._action_queue.put((action, info['step'] + step_offset))

            # indicate that all currents obs are processed
            self._obs_queue.task_done()








