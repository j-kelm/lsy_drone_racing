import numpy as np
from scipy import interpolate
import minsnap_trajectories as ms

class MinsnapPlanner:
    def __init__(self,
                 initial_info,
                 initial_obs,
                 speed=1.5,
                 gate_index=0,
                 gate_time_constant=0.25):

        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.gates_pos = initial_info['gates.pos']
        self.gates_rpy = initial_info['gates.rpy']

        self.waypoints = list()  # without helper waypoints
        time = 0.0

        self.waypoints.append(ms.Waypoint(
            time=time,
            position=initial_obs[0:3],
            velocity=initial_obs[6:9],
        ))

        for gate_pos, gate_rpy in zip(self.gates_pos[gate_index:], self.gates_rpy[gate_index:]):
            time += self.get_time_from_last_waypoint(gate_pos, self.waypoints, speed)
            self.waypoints.append(ms.Waypoint(
                time=time,
                position=gate_pos,
            ))

            theta = gate_rpy[2]
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            offset = np.zeros(3)
            offset[:2] = rotation @ np.array([0, 1.0])

            gate_front_pos = gate_pos - offset
            gate_back_pos = gate_pos + offset

        final_pos = self.waypoints[-1].position + 1.0 * np.sqrt(speed) * offset
        time += self.get_time_from_last_waypoint(final_pos, self.waypoints, speed) * 2.5
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

        progress_f = interpolate.interp1d(self.waypoint_times, range(gate_index, gate_index + len(self.waypoints)))

        t = np.linspace(0, time, np.round(time*self.CTRL_FREQ).astype(int))
        pv = ms.compute_trajectory_derivatives(polys, t, 2)

        # time steps close to gate are more important
        progress = progress_f(t)
        next_gate_idx = np.floor(progress)
        gate_prox = np.empty((len(self.waypoints), t.shape[0]))
        for i, time in enumerate(self.waypoint_times):
            gate_prox[i, :] = np.exp(-((t-time)/gate_time_constant)**2)

        gate_prox = np.max(gate_prox, axis=0)

        self.ref = np.zeros((16, np.shape(pv)[1]))
        self.ref[:3] = pv[0, ...].T
        self.ref[3:6] = pv[1, ...].T
        self.next_gate_idx = next_gate_idx
        self.gate_prox = gate_prox

        # assert max(self.ref[2, :]) < 2.5, "Drone must stay below the ceiling"
        # assert min(self.ref[2, :]) > 0.0, "Drone must stay above the ground"


    @staticmethod
    def get_time_from_last_waypoint(waypoint, prev_list, speed):
        return np.linalg.norm(waypoint - prev_list[-1].position) / speed