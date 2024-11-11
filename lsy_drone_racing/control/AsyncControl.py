import multiprocessing as mp
import numpy as np



class AsyncControl(mp.Process):
    """
    Wrap a controller to fetch actions asynchronously in a separate process


    """
    def __init__(self, ratio: int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert ratio >= 0, "Amount of environment steps per control step must be at least 1"

        self._obs_queue = mp.JoinableQueue()
        self._action_queue = mp.Queue()

        self._ratio = ratio

    def put_obs(self, obs, info, *args, **kwargs):
        self._obs_queue.put((obs, info), *args, **kwargs)

    def get_action(self, *args, **kwargs):
        return self._action_queue.get(*args, **kwargs)

    def wait_tasks(self):
        self._obs_queue.join()

    def compute_control(self, obs, info):
        raise NotImplementedError

    def run(self):
        while True:
            # blocking, wait until new obs is available
            obs, info = self._obs_queue.get()

            # compute new action every ratio steps
            if not info['step'] % self._ratio:
                # predict into future

                out = self.compute_control(obs, info)
                actions = out['actions'][self._ratio:]

                assert len(actions) >= self._ratio, "Controller must return at least ratio steps"
                for step_offset, action in enumerate(actions[:self._ratio]):
                    self._action_queue.put((action, info['step'] + step_offset))

            # indicate that all currents obs are processed
            self._obs_queue.task_done()


