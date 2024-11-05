import multiprocessing as mp
import numpy as np

from lsy_drone_racing.control.Predictor import SymbolicPredictor


class AsyncControl(mp.Process):
    """
    Wrap a controller to fetch actions asynchronously in a separate process


    """
    def __init__(self, initial_info, initial_inputs=None, ratio: int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert ratio >= 0, "Amount of environment steps per control step must be at least 1"

        self._obs_queue = mp.JoinableQueue()
        self._action_queue = mp.Queue()

        self._ratio = ratio
        self.last_inputs = np.atleast_2d(initial_inputs) if initial_inputs is not None else list()

        self._predictor = SymbolicPredictor(initial_info)

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
                pos = obs['pos']
                rpy = obs['rpy']
                vel = obs['vel']
                body_rates = obs['ang_vel']
                obs = np.concatenate([pos, vel, rpy, body_rates])

                obs_predicted, info_predicted = self._predictor.predict(obs=obs, info=info, inputs=self.last_inputs[:self._ratio])
                obs_predicted[-3:] = 0
                out = self.compute_control(obs_predicted, info_predicted)
                actions = out['actions']
                self.last_inputs = out['inputs']

                assert len(actions) >= self._ratio, "Controller must return at least ratio steps"
                for step_offset, action in enumerate(actions[:self._ratio]):
                    self._action_queue.put((action, info_predicted['step'] + step_offset))

            # indicate that all currents obs are processed
            self._obs_queue.task_done()


