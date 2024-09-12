import numpy as np

from examples.utils import np_rot_z

def vblock_constraint(obstacle_foot, length, r=0.15):
    return lambda x: 1 - ((x[0] - obstacle_foot[0]) / r) ** 2 - ((x[1] - obstacle_foot[1]) / r) ** 2 - (
                (2 * (x[2] - obstacle_foot[2]) / length) - 1) ** 2

def hblock_constraint(obstacle_center, width, yaw, r=0.15):
    def g(x):
        x_rel = np_rot_z(yaw) @ (x[0:3] - obstacle_center)
        return 1 - (2*x_rel[0]/width)**2 - (x_rel[1]/r)**2 - (x_rel[2]/r)**2

    return g

def obstacle_constraints(obstacle_pos, r=0.15):
    obstacle_pos = np.array(obstacle_pos)

    pos = np.zeros(3)
    pos[0:2] = obstacle_pos[0:2]
    return [vblock_constraint(pos, obstacle_pos[2], r)]

def gate_constraints(gate_pos, gate_yaw, r=0.15):
    # Gate:
    # ----- <- 5
    # I   I <- 3/4
    # ----- <- 2
    #   I   <- 1

    gate_pos = np.array(gate_pos)

    constraints = []
    gate_size = 0.5

    # pole 1
    pos = np.zeros(3)
    pos[0:2] = gate_pos[0:2]
    constraints.append(vblock_constraint(pos, gate_pos[2] - gate_size / 2, r))

    # pole 3/4
    x_offset = np_rot_z(gate_yaw) @ [gate_size/2, 0, 0]
    z_offset = [0, 0, -gate_size/2]
    constraints.append(vblock_constraint(gate_pos + z_offset + x_offset, gate_size, r))
    constraints.append(vblock_constraint(gate_pos + z_offset - x_offset, gate_size, r))

    # pole 2/5
    constraints.append(hblock_constraint(gate_pos + z_offset, gate_size, gate_yaw, r))
    constraints.append(hblock_constraint(gate_pos - z_offset, gate_size, gate_yaw, r))

    return constraints