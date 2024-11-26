import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

HORIZON = 3

states = np.load("output/race_data.npz", allow_pickle=True)
outputs = states['actions']

rng = default_rng()
numbers = rng.choice(len(outputs), size=2500, replace=False)

fig = plt.figure()
ax = plt.axes(projection="3d")

for index in numbers:
    snippet = outputs[index]
    ax.plot3D(snippet[0, :HORIZON], snippet[1, :HORIZON], snippet[2, :HORIZON])

ax.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()