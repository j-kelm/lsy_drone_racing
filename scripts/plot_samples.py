import yaml
import toml
from munch import munchify
import matplotlib.pyplot as plt
import numpy as np

from lsy_drone_racing.control.diffusion_controller import Controller

path = "config/mpc.yaml"
with open(path, "r") as file:
    mpc_config = munchify(yaml.safe_load(file))

with open('config/multi_modality.toml', "r") as file:
    track_config = munchify(toml.load(file))

mpc_config['ctrl_timestep'] = 1 / track_config.env.freq
mpc_config['env_freq'] = track_config.env.freq

gates = track_config.env.track.gates
gates_pos = [gate.pos for gate in gates]
gates_rpy = [gate.rpy for gate in gates]
obstacles_pos = [obstacle.pos for obstacle in track_config.env.track.obstacles]

initial_info = {
    'env_freq': track_config.env.freq,
    'nominal_physical_parameters': mpc_config.drone_params,
}

initial_obs = {
    'gates_pos': gates_pos,
    'gates_rpy': gates_rpy,
    'obstacles_pos': obstacles_pos,
    'pos': track_config.env.track.drone.pos,
    'vel': track_config.env.track.drone.vel,
    'rpy': track_config.env.track.drone.rpy,
    'ang_vel': track_config.env.track.drone.ang_vel,
    'target_gate': 0,
}

ctrl = Controller(initial_obs=initial_obs, initial_info=initial_info)

fig = plt.figure()
ax = plt.axes(projection="3d")

actions = ctrl.compute_horizon(obs=initial_obs, samples=25)

for action in actions:
    ax.plot3D(action[0], action[1], action[2])

ax.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()

