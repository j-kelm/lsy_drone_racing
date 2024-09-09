import numpy as np
from scipy import interpolate


class Planner:
    def __init__(self, initial_info, CTRL_FREQ):
        self.initial_info = initial_info
        self.CTRL_FREQ = CTRL_FREQ

    def plan(self, gates, initial_obs, duration=13):
        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled
        waypoints = []
        waypoints.append([initial_obs[0], initial_obs[2], 0.3])

        gates = [  # x, y, z, r, p, y, type (0: `tall` obstacle, 1: `low` obstacle)
            [0.45, -1.0, 0, 0, 0, 2.35, 1],
            [1.0, -1.55, 0, 0, 0, -0.78, 0],
            [0.0, 0.5, 0, 0, 0, 0, 1],
            [-0.5, -0.5, 0, 0, 0, 3.14, 0]
        ]

        z_low = self.initial_info["gate_dimensions"]["low"]["height"]
        z_high = self.initial_info["gate_dimensions"]["tall"]["height"]
        waypoints.append([1, 0, z_low])
        waypoints.append([gates[0][0] + 0.2, gates[0][1] + 0.1, z_low])
        waypoints.append([gates[0][0] + 0.1, gates[0][1], z_low])
        waypoints.append([gates[0][0] - 0.1, gates[0][1], z_low])
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.7,
                (gates[0][1] + gates[1][1]) / 2 - 0.3,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.5,
                (gates[0][1] + gates[1][1]) / 2 - 0.6,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append([gates[1][0] - 0.3, gates[1][1] - 0.2, z_high])
        waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.2, z_high])
        waypoints.append([gates[2][0], gates[2][1] - 0.4, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.1, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.1, z_high + 0.2])
        waypoints.append([gates[3][0], gates[3][1] + 0.1, z_high])
        waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.1])
        waypoints.append(
            [
                self.initial_info["x_reference"][0],
                self.initial_info["x_reference"][2],
                self.initial_info["x_reference"][4],
            ]
        )
        waypoints = np.array(waypoints)
        tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
        self.waypoints = waypoints

        steps = int(duration * self.CTRL_FREQ)
        self.traj = np.zeros(shape=(12, steps))

        t = np.linspace(0, 1, steps)
        self.traj[:3] = np.array(interpolate.splev(t, tck))
        assert max(self.traj[2, :]) < 2.5, "Drone must stay below the ceiling"
        return self.traj


class PointPlanner:
    def __init__(self, initial_info, CTRL_FREQ):
        self.initial_info = initial_info
        self.CTRL_FREQ = CTRL_FREQ

    def plan(self, gates, initial_obs, speed=2.5):
        start = np.array([1.0, 1.0, 0.3])
        end = start + np.array([0.0, 0.0, 3.0])
        self.waypoints = [start, end]

        distance = np.linalg.norm(end - start)
        steps = np.rint(distance/speed * self.CTRL_FREQ).astype(int)
        self.traj = np.zeros(shape=(12, steps))

        pos = np.linspace(start, end, steps).T
        self.traj[:3] = pos

        return self.traj

class FilePlanner:
    def __init__(self, initial_info, CTRL_FREQ):
        self.initial_info = initial_info
        self.CTRL_FREQ = CTRL_FREQ

    def plan(self, **kwargs):
        self.waypoints = []

        # load state history
        with open('examples/state_history.csv', 'rb') as f:
            self.traj = np.loadtxt(f, delimiter=',')

        return self.traj


class LinearPlanner:
    def __init__(self, initial_info, CTRL_FREQ):
        self.initial_info = initial_info
        self.CTRL_FREQ = CTRL_FREQ

    def plan(self, gates, initial_obs, duration=3, speed=2.5, acc=4.0):
        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled
        waypoints = []
        waypoints.append(initial_obs[:3])

        gates_pos = self.initial_info['gates.pos']
        gates_rpy = self.initial_info['gates.rpy']

        for gate_pos, gate_rpy in zip(gates_pos, gates_rpy):
            theta = gate_rpy[2]
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            offset = np.zeros(3)
            offset[:2] = rotation @ np.array([0, 0.15])

            waypoints.append(gate_pos - offset)
            waypoints.append(gate_pos + offset)

        waypoints = np.array(waypoints).T

        # calculate intermediary waypoint to allow linear acceleration
        i_speed = np.linspace(0, speed, np.round(speed/acc * self.CTRL_FREQ).astype(int))
        direction = (waypoints[:, 1] - waypoints[:, 0])/np.linalg.norm(waypoints[:, 1] - waypoints[:, 0])
        deltas = direction[:, np.newaxis] * i_speed[np.newaxis, :] / self.CTRL_FREQ
        spacing = waypoints[:, 0, np.newaxis] + deltas.cumsum(axis=1)

        traj_list = [spacing[:, :-1].T]
        waypoints[:, 0] = spacing[:, -1]

        # compute distances between waypoints for time allocation
        distances = np.linalg.norm((waypoints[:, 1:] - waypoints[:,:-1]), axis=0)
        steps_per_line = np.rint(distances*self.CTRL_FREQ/speed).astype(int)

        for i, steps in enumerate(steps_per_line):
            traj_list.append(np.linspace(waypoints[:,i], waypoints[:,i+1], steps, endpoint=False))

        # append last point to allow deceleration
        i_speed = np.linspace(speed, 0, np.round(speed / (acc/2) * self.CTRL_FREQ).astype(int))
        direction = (waypoints[:, -1] - waypoints[:, -2]) / np.linalg.norm(waypoints[:, -1] - waypoints[:, -2])
        deltas = direction[:, np.newaxis] * i_speed[np.newaxis, :] / self.CTRL_FREQ
        spacing = waypoints[:, -1, np.newaxis] + deltas.cumsum(axis=1)
        traj_list.append(spacing[:, 1:].T)

        traj = np.vstack(traj_list).T

        self.traj = np.zeros((12, np.shape(traj)[1]))
        self.traj[:3] = traj
        self.waypoints = waypoints

        assert max(self.traj[2, :]) < 2.5, "Drone must stay below the ceiling"
        return self.traj

