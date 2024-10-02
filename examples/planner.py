import numpy as np
from scipy import interpolate
import minsnap_trajectories as ms

class Planner:
    def __init__(self, initial_info, CTRL_FREQ):
        self.initial_info = initial_info
        self.CTRL_FREQ = CTRL_FREQ

    def plan(self, gates, initial_obs, speed=2.5, acc=4.0):
        gates_pos = self.initial_info['gates.pos']
        gates_rpy = self.initial_info['gates.rpy']

        waypoints = list()

        time = 0.0

        waypoints.append(ms.Waypoint(
            time=time,
            position=initial_obs[0:3],
            velocity=initial_obs[6:9],
        ))

        for gate_pos, gate_rpy in zip(gates_pos, gates_rpy):
            theta = gate_rpy[2]
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            offset = np.zeros(3)
            offset[:2] = rotation @ np.array([0, 0.15])

            gate_front = gate_pos - offset
            time += np.linalg.norm(gate_front - waypoints[-1].position) / speed
            waypoints.append(ms.Waypoint(
                time=time,
                position=gate_front,
            ))

            time += 0.3 / speed
            gate_back = gate_pos + offset
            waypoints.append(ms.Waypoint(
                time=time,
                position=gate_back,
            ))

        time += 1
        waypoints.append(ms.Waypoint(
            time=time,
            position=waypoints[-1].position + 2 * offset,
            velocity=np.zeros(3),
        ))

        polys = ms.generate_trajectory(
            waypoints,
            degree=8,
            idx_minimized_orders=(3, 4),
            num_continuous_orders=3,
            algorithm="closed-form"
        )

        t = np.linspace(0, time, np.round(time * self.CTRL_FREQ).astype(int))
        pv = ms.compute_trajectory_derivatives(polys, t, 2)

        self.traj = np.zeros((12, np.shape(pv)[1]))
        self.traj[:3] = pv[0, ...].T
        self.traj[3:6] = pv[1, ...].T
        self.waypoints = waypoints

        assert max(self.traj[2, :]) < 2.5, "Drone must stay below the ceiling"
        return self.traj, None


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

    def plan(self, initial_obs, speed=2.5, acc=5.0, gates=None):
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
        next_gate_list = [np.ones((1, np.shape(traj_list[0])[0]))]

        waypoints[:, 0] = spacing[:, -1]

        # compute distances between waypoints for time allocation
        distances = np.linalg.norm((waypoints[:, 1:] - waypoints[:,:-1]), axis=0)
        steps_per_line = np.rint(distances*self.CTRL_FREQ/speed).astype(int)

        for i, steps in enumerate(steps_per_line):
            traj_list.append(np.linspace(waypoints[:,i], waypoints[:,i+1], steps, endpoint=False))
            next_gate_list.append(np.ones((1, steps)) * ((i+1)//2+1))

        # append last point to allow deceleration
        i_speed = np.linspace(speed, 0, np.round(speed / (acc/2) * self.CTRL_FREQ).astype(int))
        direction = (waypoints[:, -1] - waypoints[:, -2]) / np.linalg.norm(waypoints[:, -1] - waypoints[:, -2])
        deltas = direction[:, np.newaxis] * i_speed[np.newaxis, :] / self.CTRL_FREQ
        spacing = waypoints[:, -1, np.newaxis] + deltas.cumsum(axis=1)
        traj_list.append(spacing[:, 1:].T)
        next_gate_list.append(np.ones((1, np.shape(traj_list[-1])[0])) * ((i+1)//2+1))

        traj = np.vstack(traj_list).T
        next_gate = np.hstack(next_gate_list)

        self.traj = np.zeros((12, np.shape(traj)[1]))
        self.traj[:3] = traj
        self.waypoints = waypoints

        assert max(self.traj[2, :]) < 2.5, "Drone must stay below the ceiling"
        return self.traj, next_gate


class PolynomialPlanner:
    def __init__(self, initial_info, CTRL_FREQ):
        self.initial_info = initial_info
        self.CTRL_FREQ = CTRL_FREQ
        self.waypoints = list()

    def get_time_from_last_waypoint(self, waypoint, speed):
        return np.linalg.norm(waypoint - self.waypoints[-1].position) / speed

    def plan(self, initial_obs, speed=2.5, gates=None):
        gates_pos = self.initial_info['gates.pos']
        gates_rpy = self.initial_info['gates.rpy']

        time = 0.0

        self.waypoints = list()

        self.waypoints.append(ms.Waypoint(
            time=time,
            position=initial_obs[0:3],
            velocity=initial_obs[6:9],
        ))

        for gate_pos, gate_rpy in zip(gates_pos, gates_rpy):
            theta = gate_rpy[2]
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            offset = np.zeros(3)
            offset[:2] = rotation @ np.array([0, 0.15])

            time += self.get_time_from_last_waypoint(gate_pos, speed)
            self.waypoints.append(ms.Waypoint(
                time=time,
                position=gate_pos,
            ))

        final_pos = self.waypoints[-1].position + 10 * offset
        time += self.get_time_from_last_waypoint(final_pos, speed/2)
        self.waypoints.append(ms.Waypoint(
            time=time,
            position=final_pos,
            velocity=np.zeros(3),
        ))

        polys = ms.generate_trajectory(
            self.waypoints,
            degree=8,
            idx_minimized_orders=(3, 4,),
            num_continuous_orders=3,
            algorithm="closed-form"
        )

        self.waypoint_times = [waypoint.time for waypoint in self.waypoints]
        self.waypoint_pos = [waypoint.position for waypoint in self.waypoints]

        progress_f = interpolate.interp1d(self.waypoint_times, range(len(self.waypoints)))


        t = np.linspace(0, time, np.round(time*self.CTRL_FREQ).astype(int))

        pv = ms.compute_trajectory_derivatives(polys, t, 2)

        progress = progress_f(t)

        next_gate_idx = np.floor(progress)
        gate_prox = np.empty((len(self.waypoints), t.shape[0]))
        for i, time in enumerate(self.waypoint_times):
            gate_prox[i, :] = np.exp(-((t-time)/0.25)**2)

        gate_prox = 1 + 4 * np.max(gate_prox, axis=0)

        self.traj = np.zeros((14, np.shape(pv)[1]))
        self.traj[:3] = pv[0, ...].T
        self.traj[3:6] = pv[1, ...].T
        self.traj[12] = next_gate_idx
        self.traj[13] = gate_prox

        assert max(self.traj[2, :]) < 2.5, "Drone must stay below the ceiling"
        return self.traj
