import numpy as np
from scipy.spatial.transform import Rotation as R

def transform(points, orientations, origins=None):
    """
    transform n points into m coordinate systems

    points: (N, 3, M)
    origins: (N, 3)
    orientations: (N, 3)

    return: (N, 3, M)
    """



    assert(len(points) == len(orientations) or len(points) == 1)
    if origins is None:
        origins = np.zeros_like(orientations)
    else:
        assert(len(orientations) == len(origins))

    r = R.from_euler('xyz', orientations, degrees=True)

    points = points - origins[:, :, None]

    transformed = np.empty_like(points)
    for n in range(len(orientations)):
        transformed[n] = r[n] @ points[n]

    return transformed

def deform(points, orientations, origins=None):
    """
    transform n points into m coordinate systems

    points: (N, 3, M)
    origins: (N, 3)
    orientations: (N, 3)

    return: (N, 3, M)
    """

    assert(len(points) == len(orientations) or len(points) == 1)
    if origins is None:
        origins = np.zeros_like(orientations)
    else:
        assert(len(orientations) == len(origins))

    r = R.from_euler('xyz', orientations, degrees=True).inv().as_matrix()

    transformed = np.empty_like(points)
    for n in range(len(orientations)):
        transformed[n] = r[n] @ points[n]

    transformed = transformed + origins[:, :, None]

    return transformed

def to_so3(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.hstack([s, c])

def to_local_obs(pos, vel, rpy, ang_vel, obstacles_pos, gates_pos, gates_rpy, target_gate):
    pos, vel, rpy, ang_vel, obstacles_pos, gates_pos, gates_rpy, target_gate = np.atleast_2d(pos, vel, rpy, ang_vel, obstacles_pos.T, gates_pos.T, gates_rpy.T, target_gate)
    snippet_length = pos.shape[0]

    rot = np.zeros_like(rpy)
    rot[:, 2:3] = rpy[:, 2:3]

    obstacles_pos_obs = transform(obstacles_pos.T[None, :, :], rot, pos)
    gates_pos_obs = transform(gates_pos.T[None, :, :], rot, pos)
    gates_rpy_obs = transform(gates_rpy.T[None, :, :], rot)

    obstacles_pos_obs = obstacles_pos_obs.reshape((snippet_length, -1))
    gates_pos_obs = gates_pos_obs.reshape((snippet_length, -1))
    gates_rpy_obs = gates_rpy_obs.reshape((snippet_length, -1))

    vels_obs = transform(vel[:, :, None], rot).reshape((snippet_length, -1))
    rpy_obs = transform(rpy[:, :, None], rot).reshape((snippet_length, -1))

    obs_states = np.hstack([pos[:, 2:], vels_obs, rpy_obs[:, 0:2], ang_vel]) # todo maybe fix this

    # transform angles into s, c (see hitchhikers guide to SO(3))

    return np.hstack([obs_states, target_gate, obstacles_pos_obs, gates_pos_obs, gates_rpy_obs])

def to_local_action(actions, rpys, positions):
    actions, rpys, positions = np.atleast_2d(actions, rpys, positions)
    rpys[:, 0:2] = 0

    pos_des = transform(actions[:, 0:3], rpys, positions)
    vel_des = transform(actions[:, 3:6], rpys)
    acc_des = transform(actions[:, 6:9], rpys)
    yaw_des = actions[:, 9:10] - rpys[:, 2:3, None]
    body_rates_des = actions[:, 10:13]

    return np.concatenate([pos_des, vel_des, acc_des, yaw_des, body_rates_des], axis=1)

def to_global_action(actions, rpys, positions):
    actions = np.atleast_3d(actions.T).T
    rpys, positions = np.atleast_2d(rpys, positions)
    rpys[:, 0:2] = [0, 0]

    pos_des = deform(actions[:, 0:3], rpys)
    pos_des = deform(pos_des[:, 0:3], np.zeros((1, 3)), positions)
    vel_des = deform(actions[:, 3:6], rpys)
    acc_des = deform(actions[:, 6:9], rpys)
    yaw_des = actions[:, 9:10] + rpys[:, 2:3, None]
    body_rates_des = actions[:, 10:13, :]

    return np.concatenate([pos_des, vel_des, acc_des, yaw_des, body_rates_des], axis=1)



