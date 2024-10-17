import matplotlib.pyplot as plt
import numpy as np

states = np.load("output/training.npz")
outputs = states['outputs']

fig = plt.figure()
ax = plt.axes(projection="3d")

for snippet in outputs:
    ax.plot3D(snippet[0], snippet[1], snippet[2])

ax.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()