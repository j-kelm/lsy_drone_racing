import multiprocessing as mp
import queue
import time
from turtledemo.forest import start

from lsy_drone_racing.control.diffusion.diffusion import HorizonDiffusion


class AsyncControl(mp.Process):
    """
    Wrap a controller to fetch actions asynchronously in a separate process

    """

    def __init__(self, initial_info, initial_obs, config, ratio: int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert ratio >= 0, "Amount of environment steps per control step must be at least 1"

        base_controller = "diffusion"

        if base_controller == 'diffusion':
            self.ctrl = HorizonDiffusion(initial_obs, initial_info, config)
        elif base_controller == 'mpc':
            self.ctrl = None
        else:
            raise RuntimeError(f'Controller type {base_controller} not supported!')

        self._obs_queue = mp.JoinableQueue()
        self._action_queue = mp.Queue()

        self._action_queue.cancel_join_thread()
        self._obs_queue.cancel_join_thread()

        self._terminate = mp.Event()

        self._ratio = ratio

    def put_obs(self, obs, info, *args, **kwargs):
        self._obs_queue.put((obs, info), *args, **kwargs)

    def get_action(self, *args, **kwargs):
        return self._action_queue.get(*args, **kwargs)

    def wait_tasks(self):
        self._obs_queue.join()

    def quit(self):
        self._terminate.set()

    def run(self):
        while True:
            # blocking, wait until new obs is available
            try:
                obs, info = self._obs_queue.get(block=True, timeout=1) # crash self if no obs for 30s
            except queue.Empty:
                print('Worker is done.')
                break

            # compute new action every ratio steps
            if not info['step'] % self._ratio:
                start_time = time.perf_counter()
                # adjust step into future
                info['step'] += self._ratio

                actions = self.ctrl.compute_horizon(obs, info).squeeze()
                actions = actions[:, self._ratio:].T

                assert len(actions) >= self._ratio, "Controller must return at least ratio steps"
                for step_offset, action in enumerate(actions[:self._ratio]):
                    self._action_queue.put((action, info['step'] + step_offset))
                print(f'Diffusion took {time.perf_counter() - start_time}')

            # indicate that all currents obs are processed
            self._obs_queue.task_done()







