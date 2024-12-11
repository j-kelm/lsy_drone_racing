import multiprocessing as mp
import queue
import time

from lsy_drone_racing.control.diffusion.horizon_diffusion import HorizonDiffusion
from lsy_drone_racing.control.mpc.horizon_mpc import HorizonMPC


class ControlProcess(mp.Process):
    """
    Wrap a controller to fetch actions asynchronously in a separate process

    """

    def __init__(self, initial_info, initial_obs, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_controller = config.controller

        if base_controller == 'diffusion':
            self.ctrl = HorizonDiffusion(initial_obs, initial_info, config)
        elif base_controller == 'mpc':
            self.ctrl = HorizonMPC(initial_obs, initial_info, config)
        else:
            raise RuntimeError(f'Controller type {base_controller} not supported!')

        self._obs_queue = mp.JoinableQueue()
        self._action_queue = mp.Queue()

        self._action_queue.cancel_join_thread()
        self._obs_queue.cancel_join_thread()

        self._n_actions = config.mp.n_actions
        self._delay = config.mp.offset
        assert self._n_actions > 0, "Amount of environment steps per control step must be at least 1"
        assert self._delay >= 0, "Action delay cannot be negative"

    def put_obs(self, obs, info, *args, **kwargs):
        self._obs_queue.put((obs, info), *args, **kwargs)

    def get_action(self, *args, **kwargs):
        return self._action_queue.get(*args, **kwargs)

    def wait_tasks(self):
        self._obs_queue.join()

    def run(self):
        while True:
            # blocking, wait until new obs is available
            try:
                obs, info = self._obs_queue.get(block=True, timeout=5) # crash self if no obs
            except queue.Empty:
                print('Worker is done.')
                return 0

            # compute new action every ratio steps
            if not info['step'] % self._n_actions:
                start_time = time.perf_counter()

                info['step'] += self._n_actions

                actions = self.ctrl.compute_horizon(obs, info).squeeze()
                assert len(actions) >= self._n_actions + self._delay, "Controller must return at least delay + n_action steps"

                actions = actions[:, self._delay:self._n_actions+self._delay].T
                for step_offset, action in enumerate(actions):
                    self._action_queue.put((action, info['step'] + step_offset))
                print(f'Controller took {time.perf_counter() - start_time}')

            # indicate that all currents obs are processed
            self._obs_queue.task_done()







