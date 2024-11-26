import matplotlib.pyplot as plt
import numpy as np

history = np.load("output/logs/sim_diffusion.npz")

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot3D(history['states'][:, 0], history['states'][:, 1], history['states'][:, 2], color='r', alpha=0.5)

for i, step in enumerate(history['actions']):
    if not i % 1:
        ax.scatter3D(history['states'][i, 0], history['states'][i, 1], history['states'][i, 2], color='r')
        for sample in step:
            ax.plot3D(sample[0], sample[1], sample[2])

ax.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()
