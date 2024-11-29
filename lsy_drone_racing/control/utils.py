import numpy as np
from scipy.spatial.transform import Rotation as R


def np_rot_x(phi):

    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def np_rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def np_rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def np_rot_xyz(phi, theta, psi) -> np.ndarray:
    """Rotation matrix from euler angles.

    This represents the extrinsic X-Y-Z (or quivalently the intrinsic Z-Y-X (3-2-1)) euler angle
    rotation.

    Args:
        phi: roll (or rotation about X).
        theta: pitch (or rotation about Y).
        psi: yaw (or rotation about Z).

    Returns:
        R: numpy Rotation matrix
    """

    return np_rot_z(psi) @ np_rot_y(theta) @ np_rot_x(phi)


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

    r = R.from_euler('xyz', orientations, degrees=False).as_matrix()

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

    r = R.from_euler('xyz', orientations, degrees=False).inv().as_matrix()

    transformed = np.empty_like(points)
    for n in range(len(orientations)):
        transformed[n] = r[n] @ points[n]

    transformed = transformed + origins[:, :, None]

    return transformed

def to_so2(angle):
    """
    Convert angle into sin, cos

    Args:
        angle: Angle in radians

    Return:
        sin and cos of the angle. For multiple angles, the output is [s, s , ..., c, c, ...]
    """
    s, c = np.sin(angle), np.cos(angle)
    return np.hstack([s, c])

def from_so2(sc):
    """
    Convert sin, cos output into angle

    Args:
        sc: [sin, cos]

    Returns:
        The corresponding angle in radians
    """
    angle = np.arctan2(sc[:, 0:1, :], sc[:, 1:2, :])
    return angle

def to_local_obs(pos, vel, rpy, ang_vel, obstacles_pos, gates_pos, gates_rpy, target_gate):
    use_so2 = True

    pos, vel, rpy, ang_vel, obstacles_pos, gates_pos, gates_rpy, target_gate = np.atleast_2d(pos, vel, rpy, ang_vel, obstacles_pos.T, gates_pos.T, gates_rpy.T, target_gate)
    snippet_length = pos.shape[0]

    ref_pos = np.zeros_like(pos)
    ref_rot = np.zeros_like(rpy)

    ref_pos[:, 0:3] = pos[:, 0:3]
    ref_rot[:, 2:3] = rpy[:, 2:3]

    vels_obs = transform(vel[:, :, None], ref_rot).reshape((snippet_length, -1))

    obstacles_pos_obs = transform(obstacles_pos.T[None, :, :], ref_rot, ref_pos).reshape((snippet_length, -1))
    gates_pos_obs = transform(gates_pos.T[None, :, :], ref_rot, ref_pos).reshape((snippet_length, -1))
    gates_rpy_obs = transform(gates_rpy.T[None, :, :], ref_rot)[:, 2:3, :].reshape((snippet_length, -1))

    if use_so2:
        obs_states = np.hstack([pos[:, 2:], vels_obs, to_so2(rpy[:, 0:2]), ang_vel])
        gates_rpy_obs = to_so2(gates_rpy_obs)
    else:
        obs_states = np.hstack([pos[:, 2:], vels_obs, rpy[:, 0:2], ang_vel])

    return np.hstack([obs_states, target_gate, obstacles_pos_obs, gates_pos_obs, gates_rpy_obs])


def to_local_action(actions, rpy, pos):
    use_so2 = True
    actions, rpy, pos = np.atleast_2d(actions, rpy, pos)

    ref_pos = np.zeros_like(pos)
    ref_rot = np.zeros_like(rpy)

    ref_pos[:, 0:3] = pos[:, 0:3]
    ref_rot[:, 2:3] = rpy[:, 2:3]

    pos_des = transform(actions[:, 0:3], ref_rot, ref_pos)
    vel_des = transform(actions[:, 3:6], ref_rot)
    acc_des = transform(actions[:, 6:9], ref_rot)
    yaw_des = actions[:, 9:10] - ref_rot[:, 2:3, None]
    body_rates_des = actions[:, 10:13]

    if use_so2:
        return np.concatenate([pos_des, vel_des, acc_des, to_so2(yaw_des), body_rates_des], axis=1)
    else:
        return np.concatenate([pos_des, vel_des, acc_des, yaw_des, body_rates_des], axis=1)

def to_global_action(actions, rpy, pos):
    use_so2 = True
    actions = np.atleast_3d(actions.T).T
    rpy, pos = np.atleast_2d(rpy, pos)

    ref_pos = np.zeros_like(pos)
    ref_rot = np.zeros_like(rpy)

    ref_pos[:, 0:3] = pos[:, 0:3]
    ref_rot[:, 2:3] = rpy[:, 2:3]

    ref_rot = np.tile(ref_rot, (len(actions), 1))
    ref_pos = np.tile(ref_pos, (len(actions), 1))

    pos_des = deform(actions[:, 0:3], ref_rot)
    pos_des = deform(pos_des[:, 0:3], np.zeros((len(actions), 3)), ref_pos)
    vel_des = deform(actions[:, 3:6], ref_rot)
    acc_des = deform(actions[:, 6:9], ref_rot)

    if use_so2:
        yaw_des = from_so2(actions[:, 9:11]) + rpy[:, 2:3, None]
        body_rates_des = actions[:, 11:14, :]

    else:
        yaw_des = actions[:, 9:10] + rpy[:, 2:3, None]
        body_rates_des = actions[:, 10:13, :]

    return np.concatenate([pos_des, vel_des, acc_des, yaw_des, body_rates_des], axis=1)



