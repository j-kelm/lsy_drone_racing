import numpy as np

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

