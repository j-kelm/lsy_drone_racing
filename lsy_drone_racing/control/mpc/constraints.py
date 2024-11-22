import numpy as np
import casadi as cs
import pybullet as p

from lsy_drone_racing.control.utils import np_rot_z

from scipy.spatial.transform import Rotation as R


def draw_elliptic_sphere(
        pos, scale, quat=(0, 0, 0, 1), rgbaColor=(1, 0, 0, 0.2)
):
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName="sphere_smooth.obj",
        meshScale=scale,
        # visualFramePosition=shift,
        rgbaColor=rgbaColor,
    )

    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=pos,
        baseOrientation=quat,
    )

def vblock_constraint(obstacle_center, length, r=0.15):
    def g(x):
        return ((x[0] - obstacle_center[0]) / r) ** 2 + ((x[1] - obstacle_center[1]) / r) ** 2 + (
            (2 * (x[2] - obstacle_center[2]) / length)) ** 2 # TODO: Change back to 4

    if __debug__:
        draw_elliptic_sphere(obstacle_center, [r, r, length/2])

    return g

def hblock_constraint(obstacle_center, width, yaw, r=0.15):
    def g(x):
        x_rel = np_rot_z(yaw) @ (x[0:3] - obstacle_center[:, None])
        return  (2*x_rel[0]/width)**2 + (x_rel[1]/r)**2 + (x_rel[2]/r)**2

    if __debug__:
        quat = R.from_euler("xyz", [0, 0, yaw], degrees=False).as_quat()
        draw_elliptic_sphere(obstacle_center, [width/2, r, r], quat=quat)

    return g

def obstacle_constraints(obstacle_pos, r=0.15, s=1.5):
    obstacle_center = np.zeros_like(obstacle_pos)

    obstacle_height = obstacle_pos[2] * s
    obstacle_center[0:2] = obstacle_pos[0:2]
    obstacle_center[2] = obstacle_pos[2]/2
    return [vblock_constraint(obstacle_center, obstacle_height, r)]

def gate_constraints(gate_pos, gate_yaw, r=0.15, s=1.75):
    # Gate:
    # ----- <- 5
    # I   I <- 3/4
    # ----- <- 2
    #   I   <- 1

    gate_pos = np.array(gate_pos)

    constraints = []
    gate_size = 0.48

    # pole 1
    pos = np.zeros(3)
    pos[0:2] = gate_pos[0:2]
    pos[2] = (gate_pos[2] - gate_size/2)/2
    constraints.append(vblock_constraint(pos, (gate_pos[2] - gate_size / 2), r))

    # pole 3/4
    x_offset = np_rot_z(gate_yaw) @ [gate_size/2, 0, 0]
    constraints.append(vblock_constraint(gate_pos + x_offset, gate_size*s, r))
    constraints.append(vblock_constraint(gate_pos - x_offset, gate_size*s, r))

    # pole 2/5
    z_offset = [0, 0, -gate_size/2]
    constraints.append(hblock_constraint(gate_pos + z_offset, gate_size*s, gate_yaw, r))
    constraints.append(hblock_constraint(gate_pos - z_offset, gate_size*s, gate_yaw, r))

    return constraints

def rbf(x, sigma):
    return cs.exp(-x/sigma)

def to_rbf_potential(constraints: list):
    def g(x):
        res = -rbf(1, 0.25)
        # res = -np.exp(-1/0.25)
        for constraint in constraints:
            res += rbf(constraint(x), 0.25)
        return res

    return g