import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

action_labels = ['$x$', '$y$', '$z$',
                 '$\dot{x}$', '$\dot{y}$', '$\dot{z}$',
                 '$\ddot{x}$', '$\ddot{y}$', '$\ddot{z}$',
                 '$\\psi$',
                 '$p$', '$q$', '$r$']
action_groups = [('Position', (0, 1, 2), '$m$'),
                 ('Velocity', (3, 4, 5), '$\\frac{m}{s}$'),
                 ('Acceleration', (6, 7, 8), '$\\frac{m}{s^2}$'),
                 ('Yaw', (9,), 'rad'),
                 ('Rate', (10, 11, 12), '$\\frac{rad}{s}$')]

state_labels = ['$x$', '$y$', '$z$',
                '$\dot{x}$', '$\dot{y}$', '$\dot{z}$',
                '$\\phi$', '$\\theta$', '$\\psi$',
                '$p$', '$q$', '$r$',
                '$F_1$', '$F_2$', '$F_2$', '$F_4$',]
state_groups = [('Position', (0, 1, 2), '$m$'),
                ('Velocity', (3, 4, 5), '$\\frac{m}{s}$'),
                ('Attitude', (6, 7, 8), 'rad'),
                ('Rate', (9, 10, 11), '$\\frac{rad}{s}$'),]

input_labels = ['$\\dot{F}_1$', '$\\dot{F}_2$', '$\\dot{F}_3$', '$\\dot{F}_4$']
input_groups = [('Thrust Change', (0, 1, 2, 3), '$\\frac{N}{s}$'),]

def plot_groups(data, timesteps, index_groups, labels, title, save=False):
    # plot states
    fig, axs = plt.subplots(len(index_groups), sharex=True, figsize=(20, 15), squeeze=False)
    axs = axs.squeeze(1)
    if not save:
        fig.suptitle(title)

    for ax, (group_label, index_group, group_units) in zip(axs, index_groups):
        ax.set_title(group_label)
        ax.set_ylabel(group_units)  # , rotation=0
        for i in index_group:
            ax.plot(timesteps, data[:, i, 0], label=labels[i])
            ax.legend(loc='lower right')

    axs[-1].set_xlabel('s')


def plot_trajectories3d(state_data, index_group):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # plot continuous lines
    for i, episode in enumerate(state_data):
        # ax.plot(episode[:, 0, 0], episode[:, 1, 0], episode[:, 2, 0], c='gray', alpha=0.5)

        state_data[i] = episode[..., 0:6, 0].reshape(-1, 6)

    state_data = np.concatenate(state_data)

    img = ax.scatter(state_data[:, 0], state_data[:, 1], state_data[:, 2], c=np.linalg.norm(state_data[:, 3:6], axis=1), cmap='turbo', s=2.5, rasterized=True)  # , alpha=0.5)
    ax.set_aspect('equal', 'box')
    ax.view_init(15,45,0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    cbar = fig.colorbar(img, fraction=0.015, pad=0.05)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(r'\SI{%.1f}{\meter\per\second}'))
    fig.tight_layout()

    return fig, ax

def plot_trajectories2d(state_data, index_group):
    fig = plt.figure(dpi=100)
    ax = plt.axes()

    # plot continuous lines
    for i, episode in enumerate(state_data):
        ax.plot(episode[:, 1, 0], -episode[:, 0, 0], c='gray', alpha=0.25)

        state_data[i] = episode[..., 0:6, 0].reshape(-1, 6)

    state_data = np.concatenate(state_data)

    img = ax.scatter(state_data[:, 1], -state_data[:, 0], c=np.linalg.norm(state_data[:, 3:6], axis=1), cmap='turbo', s=5)  # , alpha=0.5)
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(img)
    cbar.ax.set_ylabel(index_group[1][2], rotation=0)
    fig.tight_layout()

    return fig, ax
